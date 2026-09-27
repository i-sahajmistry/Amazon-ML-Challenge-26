#!/bin/bash
# End to end: student_resource/ -> output/{matching_results,candidate_pairs}.tsv.
# Blocking (fine-tuned e5, HNSW, shortlist with text similarities) -> pair features -> two cross-encoders -> stages 1-2
# (+ a bge cross-encoder for the countries with training labels) -> LLM judge on the unsure records -> for the countries
# without training labels, three self-training rounds (each a veto) and the same-address fixes -> recall fixes ->
# threshold 0.70. Steps that do not depend on each other run at the same time on up to three GPUs. Each step logs to
# work/logs/<step>.log and leaves work/logs/<step>.done, so a rerun resumes where it stopped.
#   AMLC_ROOT=/folder/with/student_resource GPU_A=0 GPU_B=1 GPU_C=2 PY=/env/bin/python bash reproduce.sh
# Fewer GPUs: GPU_C defaults to GPU_B, GPU_B to GPU_A (one 80 GB GPU works). work/logs/gpu_A (B, C) holding a device
# id moves a lane's later steps. Offline nodes: E5=/path/multilingual-e5-small, LLM=/path/Qwen3-Reranker-4B,
# BGE=/path/bge-reranker-v2-m3.
set -u
cd "$(dirname "$0")/src"
export AMLC_ROOT=${AMLC_ROOT:-$(dirname "$PWD")} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
PY=${PY:-python}; W=$AMLC_ROOT/work; X=$W/x; L=$W/logs; mkdir -p "$L" "$X"

gpu() { case $1 in A) echo "${GPU_A:-0}";; B) echo "${GPU_B:-${GPU_A:-0}}";; C) echo "${GPU_C:-${GPU_B:-${GPU_A:-0}}}";; esac; }
s() {   # s NAME LANE [VAR=value ...] -- python-args
    local n=$1 g; shift
    g=$(cat "$L/gpu_$1" 2>/dev/null || gpu "$1"); shift
    local e=(); while [ "$1" != -- ]; do e+=("$1"); shift; done; shift
    [ -e "$L/$n.done" ] && return 0
    [ -e "$L/FAILED" ] && exit 1
    echo "$(date '+%F %T') start $n (GPU $g)"
    if env ${e[@]+"${e[@]}"} CUDA_VISIBLE_DEVICES="$g" "$PY" -u "$@" > "$L/$n.log" 2>&1; then
        touch "$L/$n.done"; echo "$(date '+%F %T') done  $n"
    else
        echo "$n" >> "$L/FAILED"; echo "$(date '+%F %T') FAILED $n: $L/$n.log"; exit 1
    fi
}
ok() { wait; [ ! -e "$L/FAILED" ] || exit 1; }

D=DFOLD=1; SELF="CE_N=3000000 SELF_N=4000000 CE_LR=2e-5"
R10="$D CE_TEXT=raw CE_DIR=$X/ce_r10 CE_TAG=_r10"; N10="$D CE_DIR=$X/ce_n10 CE_TAG=_n10"
VETO="unlabelled:_a1pw:min,unlabelled:_a2pw:min,unlabelled:_a3pw:min"   # unlabelled: test minus train countries

# blocking: bi-encoder, HNSW retrieval of both splits, lexicon learned from pseudo-matches, shortlist model
s selfcheck A -- common.py
s train_embed A -- train_embed.py
( s retrieve_train A ANN=hnsw -- retrieve.py train ) & ( s retrieve_test B ANN=hnsw -- retrieve.py test ) & ok
s lexicon A -- lexicon.py
s shortlist A -- x_shortlist2.py                                                # work/shortlist_text.txt
# pair features of the shortlisted pairs (they are candidate_pairs.tsv); stage 1 without cross-encoders, which the
# cross-encoder fits' fold-9 check compares against
( s features_train A SHORTLIST=text:0.001 -- -c "from match import features; features('train')" ) &
( s features_test B SHORTLIST=text:0.001 -- -c "from match import features; features('test')" ) & ok
( s extra_train A -- -c "import x_feats; x_feats.main('train')" ) &
( s extra_test B -- -c "import x_feats; x_feats.main('test')" ) & ok
s dfold_cache A $D -- -c "from harness import truth_arrays; truth_arrays()"   # x/imitated_train.npy
s stage1v4 A -- stage1_cv.py
# two cross-encoders from the fine-tuned e5 (raw / normalised text), each distractor in the fold of the S1 it imitates
( s ce_r_fit A $R10 CE_N=12000000 -- x_ce3.py train
  s ce_r_train A $R10 -- x_ce3.py score train; s ce_r_test A $R10 -- x_ce3.py score test ) &
( s ce_n_fit B $N10 CE_N=12000000 -- x_ce3.py train
  s ce_n_train B $N10 -- x_ce3.py score train; s ce_n_test B $N10 -- x_ce3.py score test ) & ok
s stage1 A $D CE_TAGS=_n10,_r10 TAG=_v10 S1K=3 -- x_stage_multi.py s1
s stage2 A $D CE_TAGS=_n10,_r10 TAG=_v10 BAG=5 PEERS=1 -- x_nocopy.py test                        # x/test_q_v10pw

# lane C: the LLM judge (Qwen3-Reranker-4B + LoRA) trained on the unsure train records and scoring the unsure ones.
# Lanes A / B meanwhile: three self-training rounds for the countries without training labels. Both cross-encoders
# continue on the confident decisions so far (round 1: stage 2's; round k: stage 2 vetoed by rounds < k), stages 1-2
# rerun; each round's stage 2 becomes a veto.
# Lane C then adds bge-reranker-v2-m3 as a third cross-encoder and reruns stages 1-2 with it: the base for the
# countries with training labels (validation +0.00011 over the two e5 cross-encoders).
B="$D CE_BASE=${BGE:-BAAI/bge-reranker-v2-m3} CE_TEXT=orig CE_DIR=$X/ce_b10 CE_TAG=_b10"
( s llm_train C QT=_v10p -- x_llm.py train
  s llm_val C QT=_v10p -- x_llm.py val; s llm_test C QT=_v10p -- x_llm.py test
  s b_fit C $B CE_N=3000000 CE_BS=256 CE_LR=2e-5 -- x_ce3.py train
  s b_train C $B -- x_ce3.py score train; s b_test C $B -- x_ce3.py score test
  s b_s1 C $D CE_TAGS=_n10,_r10,_b10 TAG=_v10b S1K=3 -- x_stage_multi.py s1
  s b_s2 C $D CE_TAGS=_n10,_r10,_b10 TAG=_v10b BAG=5 PEERS=1 -- x_nocopy.py test ) &                # x/test_q_v10bpw
round() {   # round K SEED_FILE
    local k=$1 seed=$2 a b RA="$D CE_TEXT=raw CE_DIR=$X/ce_ra$1 CE_TAG=_ra$1" NA="$D CE_DIR=$X/ce_na$1 CE_TAG=_na$1"
    ( s a${k}_ra_fit A $RA CE_INIT=$X/ce_r10 SELF=$seed $SELF -- x_ce3.py train
      s a${k}_ra_train A $RA -- x_ce3.py score train; s a${k}_ra_test A $RA -- x_ce3.py score test ) & a=$!
    ( s a${k}_na_fit B $NA CE_INIT=$X/ce_n10 SELF=$seed $SELF -- x_ce3.py train
      s a${k}_na_train B $NA -- x_ce3.py score train; s a${k}_na_test B $NA -- x_ce3.py score test ) & b=$!
    wait $a $b; [ ! -e "$L/FAILED" ] || exit 1
    s a${k}_s1 A $D CE_TAGS=_na$k,_ra$k TAG=_a$k S1K=3 -- x_stage_multi.py s1
    s a${k}_s2 A $D CE_TAGS=_na$k,_ra$k TAG=_a$k BAG=5 PEERS=1 -- x_nocopy.py test               # x/test_q_a{k}pw
}
round 1 "$X/test_q_v10pw.parquet"
s a2_seed A FROM=unlabelled:_a1pw:min SAVE_Q=_a2seed -- x_final.py - _v10pw
round 2 "$X/test_q_a2seed.parquet"
s a3_seed A FROM=unlabelled:_a1pw:min,unlabelled:_a2pw:min SAVE_Q=_a3seed -- x_final.py - _v10pw
round 3 "$X/test_q_a3seed.parquet"
ok

# the LLM judge blended into stage 2's unsure records (the e5 stack: base for the countries without training labels;
# the bge stack: base for the others); the three vetoes; same-address fixes for the countries without training labels
# (x_ddfix.py; structural candidates from rule_fr.py + wstat.py); the recall fixes (x_recall.py); threshold 0.70
s llm_stack A -- x_llmstack.py _v10p _v10p                                                        # x/test_q_v10plw
s llm_stack_b A -- x_llmstack.py _v10p _v10bp                                                     # x/test_q_v10bplw
s llm2_seed A FROM=$VETO SAVE_Q=_v10fr3l -- x_final.py - _v10plw
s wstat_test A -- wstat.py test
s rule_fr A -- rule_fr.py _v10plw
s x_ddfix A -- x_ddfix.py _v10plw _v10fr3l
F="unlabelled:_v10plw,$VETO"
s fin_q A FROM=$F SAVE_Q=_fin -- x_final.py - _v10bplw
s x_recall A -- x_recall.py _fin _v10p restore_dd reject_dd _v10plw
s llm_sn C ROWS=sn_rows -- x_llm.py rows          # the judge on recall candidates outside its band (x/sn_rows)
s x_recall2 A -- x_recall.py _fin _v10p restore_dd reject_dd _v10plw                    # + their scores
s final A FROM=$F RESTORE=restore_dd,restore_empty,restore_nafr,restore_nacore,restore_samename,restore_vetona,restore_exact REJECT=reject_dd,reject_decword -- x_final.py "$AMLC_ROOT/output" _v10bplw 0.70
O=$AMLC_ROOT/output   # absolute: "../" would leave AMLC_ROOT when student_resource is a symlink
cd "$AMLC_ROOT/student_resource" && "$PY" utils/validate_submission.py --check-ids --test-dir dataset/test \
    --matching "$O/matching_results.tsv" --candidate "$O/candidate_pairs.tsv"
