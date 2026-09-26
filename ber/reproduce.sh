#!/bin/bash
# End to end: student_resource/ -> output_v10_fr3_llm_dd/{matching_results,candidate_pairs}.tsv (leaderboard 0.990282).
# Same steps and environment as the x_chain.py plans v8a, v8b, v9b, v9p, v9s, v9s2, v10, v10s3, the LLM judge and the
# same-address fixes (README "Reproduce"); steps that do not depend on each other run at the same time on two GPUs.
# Each step logs to work/logs/<step>.log and leaves work/logs/<step>.done, so a rerun resumes where it stopped.
#   AMLC_ROOT=/folder/with/student_resource GPU_A=0 GPU_B=1 PY=/env/bin/python bash reproduce.sh
# GPU_A = GPU_B works on one 80 GB GPU. work/logs/gpu_A (or gpu_B) holding a device id moves a lane's later steps.
# Offline nodes: E5=/path/multilingual-e5-small, LLM=/path/Qwen3-Reranker-4B.
# LLM_ELSEWHERE=1: the three x_llm.py steps run on another machine; this script waits for work/logs/llm_test.done.
set -u
cd "$(dirname "$0")/src"
export AMLC_ROOT=${AMLC_ROOT:-$(dirname "$PWD")} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
PY=${PY:-python}; W=$AMLC_ROOT/work; X=$W/x; L=$W/logs; mkdir -p "$L" "$X"

s() {   # s NAME LANE [VAR=value ...] -- python-args
    local n=$1 g; shift
    g=$(cat "$L/gpu_$1" 2>/dev/null || { [ "$1" = A ] && echo "${GPU_A:-0}" || echo "${GPU_B:-${GPU_A:-0}}"; }); shift
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
w() { for n; do until [ -e "$L/$n.done" ]; do [ -e "$L/FAILED" ] && exit 1; sleep 30; done; done; }
ok() { wait; [ ! -e "$L/FAILED" ] || exit 1; }
feats() {   # pair features and x_feats extras of both splits, for the shortlist in $2
    ( s $1_features_train A $2 -- -c "from match import features; features('train')" ) &
    ( s $1_features_test B $2 -- -c "from match import features; features('test')" ) & ok
    ( s $1_extra_train A -- -c "import x_feats; x_feats.main('train')" ) &
    ( s $1_extra_test B -- -c "import x_feats; x_feats.main('test')" ) & ok
}

RAW="CE_TEXT=raw CE_DIR=$X/ce_raw CE_TAG=_raw"; V9="CE_TAGS=,_raw TAG=_v9"
SELF="CE_N=3000000 SELF_N=4000000 CE_LR=2e-5"; D=DFOLD=1
RS="CE_TEXT=raw CE_DIR=$X/ce_rs CE_TAG=_rs"; NS="CE_DIR=$X/ce_ns CE_TAG=_ns"
RS2="CE_TEXT=raw CE_DIR=$X/ce_rs2 CE_TAG=_rs2"; NS2="CE_DIR=$X/ce_ns2 CE_TAG=_ns2"
R10="$D CE_TEXT=raw CE_DIR=$X/ce_r10 CE_TAG=_r10"; N10="$D CE_DIR=$X/ce_n10 CE_TAG=_n10"
RS3="$D CE_TEXT=raw CE_DIR=$X/ce_rs3 CE_TAG=_rs3"; NS3="$D CE_DIR=$X/ce_ns3 CE_TAG=_ns3"
VETO="unlabelled:_v9spw:min,unlabelled:_v9s2pw:min,unlabelled:_v10s3pw:min"   # unlabelled: test minus train countries

# v8a: bi-encoder, HNSW retrieval of both splits, learned lexicon, shortlist model
s selfcheck A -- common.py
s train_embed A -- train_embed.py
( s v8_retrieve_train A ANN=hnsw -- retrieve.py train ) & ( s v8_retrieve_test B ANN=hnsw -- retrieve.py test ) & ok
s v8_lexicon A -- lexicon.py
s v8_shortlist A -- shortlist.py
# v8b: features on the calibrated shortlist, both cross-encoders trained and scored, stages -> x/test_q_v8w
feats v8 SHORTLIST=model:0.002
( s v8_stage1v4 A -- stage1_cv.py; s v9_shortlist2 A -- x_shortlist2.py ) &
( s v8_ce_train A CE_N=12000000 -- x_ce.py train
  s v8_ce_score_train A -- x_ce.py score train; s v8_ce_score_test A -- x_ce.py score test ) &
( s v8_raw_train B CE_TEXT=raw CE_DIR=$X/ce_raw CE_N=12000000 -- x_ce3.py train
  s v8_raw_score_train B $RAW -- x_ce3.py score train; s v8_raw_score_test B $RAW -- x_ce3.py score test ) & ok
s v8_s1 A TAG=_v8 CE_TAGS=,_raw -- x_stage_multi.py s1
s v8_s2 A TAG=_v8 CE_TAGS=,_raw -- x_nocopy.py test
# From here four lanes run at once. CPU: v9b's pair caches (features() reuses an existing feats2, so v8's are set
# aside first), then every stage as its cross-encoder scores arrive. GPU lanes A (raw-text CEs) and B (normalised CEs):
# self-training round 1 (v9s: both CEs continued on v8's confident decisions for the countries without training
# labels, x/test_q_v8w), v8's CEs re-scoring
# the text shortlist (v9b), v10 (both CEs retrained from scratch with distractors in their imitated S1's fold),
# rounds 2 (v9s2, seeded by v9p_frand) and 3 (v10s3, seeded by v10_fr2). The LLM judge needs v9's stage 1 and v9p.
# V10_ELSEWHERE=1 / LLM_ELSEWHERE=1: those fits (the LLM: all three steps) run on another machine; the lanes wait for
# their .done files.
( if [ ! -e "$L/v9_archive_v8.done" ]; then
      mkdir -p "$W/archive_v8/x" && mv "$W"/feats2_{train,test}.parquet "$W/oof_train.parquet" "$W/p1_test.parquet" "$W/archive_v8/" \
          && mv "$X"/ce_{train,test}.npy "$X"/ce_raw_{train,test}.npy "$X"/extra_{train,test}.parquet "$W/archive_v8/x/" \
          && touch "$L/v9_archive_v8.done" || { echo v9_archive_v8 >> "$L/FAILED"; exit 1; }
  fi
  s dfold_cache A $D -- -c "from harness import truth_arrays; truth_arrays()"   # x/imitated_train.npy for the v10 lanes
  feats v9 SHORTLIST=text:0.001
  s v9_stage1v4 A -- stage1_cv.py
  w v9_ce_score_test v9_raw_score_test
  s v9_s1 A $V9 S1K=3 -- x_stage_multi.py s1
  s v9_s2p A $V9 BAG=5 PEERS=1 -- x_nocopy.py test                                              # v9p: x/test_q_v9pw
  w v9s_rs_test v9s_ns_test
  s v9s_s1 A CE_TAGS=_ns,_rs TAG=_v9s S1K=3 -- x_stage_multi.py s1
  s v9s_s2p A CE_TAGS=_ns,_rs TAG=_v9s BAG=5 PEERS=1 -- x_nocopy.py test                        # x/test_q_v9spw
  s v9s2_seed A FROM=unlabelled:_v9spw:min SAVE_Q=_v9pfr -- x_final.py - _v9pw                        # v9p_frand's q
  w v10_r_test v10_n_test
  s v10_s1 A $D CE_TAGS=_n10,_r10 TAG=_v10 S1K=3 -- x_stage_multi.py s1
  s v10_s2p A $D CE_TAGS=_n10,_r10 TAG=_v10 BAG=5 PEERS=1 -- x_nocopy.py test                   # x/test_q_v10pw
  w v9s2_rs_test v9s2_ns_test
  s v9s2_s1 A CE_TAGS=_ns2,_rs2 TAG=_v9s2 S1K=3 -- x_stage_multi.py s1
  s v9s2_s2p A CE_TAGS=_ns2,_rs2 TAG=_v9s2 BAG=5 PEERS=1 -- x_nocopy.py test                    # x/test_q_v9s2pw
  s v10s3_seed A FROM=unlabelled:_v9spw:min,unlabelled:_v9s2pw:min SAVE_Q=_v10fr -- x_final.py - _v10pw  # v10_fr2's q
  w v10s3_rs_test v10s3_ns_test
  s v10s3_s1 A $D CE_TAGS=_ns3,_rs3 TAG=_v10s3 S1K=3 -- x_stage_multi.py s1
  s v10s3_s2p A $D CE_TAGS=_ns3,_rs3 TAG=_v10s3 BAG=5 PEERS=1 -- x_nocopy.py test ) &          # x/test_q_v10s3pw
( s v9s_rs_fit A $RS CE_INIT=$X/ce_raw SELF=$X/test_q_v8w.parquet $SELF -- x_ce3.py train
  w v9_extra_train v9_extra_test
  s v9_raw_score_train A $RAW -- x_ce3.py score train; s v9_raw_score_test A $RAW -- x_ce3.py score test
  s v9s_rs_train A $RS -- x_ce3.py score train; s v9s_rs_test A $RS -- x_ce3.py score test
  w dfold_cache v9_stage1v4   # a v10 fit's fold-9 check reads feats2_train / oof_train
  if [ -n "${V10_ELSEWHERE:-}" ]; then w v10_r_fit; else s v10_r_fit A $R10 CE_N=12000000 -- x_ce3.py train; fi
  s v10_r_train A $R10 -- x_ce3.py score train; s v10_r_test A $R10 -- x_ce3.py score test
  w v9s2_seed
  s v9s2_rs_fit A $RS2 CE_INIT=$X/ce_raw SELF=$X/test_q_v9pfr.parquet $SELF -- x_ce3.py train
  s v9s2_rs_train A $RS2 -- x_ce3.py score train; s v9s2_rs_test A $RS2 -- x_ce3.py score test
  w v10s3_seed
  s v10s3_rs_fit A $RS3 CE_INIT=$X/ce_r10 SELF=$X/test_q_v10fr.parquet $SELF -- x_ce3.py train
  s v10s3_rs_train A $RS3 -- x_ce3.py score train; s v10s3_rs_test A $RS3 -- x_ce3.py score test ) &
( s v9s_ns_fit B $NS CE_INIT=$X/ce SELF=$X/test_q_v8w.parquet $SELF -- x_ce3.py train
  w v9_extra_train v9_extra_test
  s v9_ce_score_train B -- x_ce.py score train; s v9_ce_score_test B -- x_ce.py score test
  s v9s_ns_train B $NS -- x_ce3.py score train; s v9s_ns_test B $NS -- x_ce3.py score test
  w dfold_cache v9_stage1v4
  if [ -n "${V10_ELSEWHERE:-}" ]; then w v10_n_fit; else s v10_n_fit B $N10 CE_N=12000000 -- x_ce3.py train; fi
  s v10_n_train B $N10 -- x_ce3.py score train; s v10_n_test B $N10 -- x_ce3.py score test
  w v9s2_seed
  s v9s2_ns_fit B $NS2 CE_INIT=$X/ce SELF=$X/test_q_v9pfr.parquet $SELF -- x_ce3.py train
  s v9s2_ns_train B $NS2 -- x_ce3.py score train; s v9s2_ns_test B $NS2 -- x_ce3.py score test
  w v10s3_seed
  s v10s3_ns_fit B $NS3 CE_INIT=$X/ce_n10 SELF=$X/test_q_v10fr.parquet $SELF -- x_ce3.py train
  s v10s3_ns_train B $NS3 -- x_ce3.py score train; s v10s3_ns_test B $NS3 -- x_ce3.py score test ) &
( if [ -n "${LLM_ELSEWHERE:-}" ]; then w llm_test; else   # LLM judge (Qwen3-Reranker-4B LoRA) on v9p's unsure records
      w v9_s1; s llm_train B -- x_llm.py train; w v9_s2p; s llm_val B -- x_llm.py val; s llm_test B -- x_llm.py test; fi ) &
ok

# v10_fr3_llm: the LLM judge blended into v10's unsure records, the three self-training vetoes; then the same-address
# fixes for the countries without training labels (x_ddfix.py; structural candidates from rule_fr.py + wstat.py)
s llm_stack A -- x_llmstack.py _v9p _v10p                                                        # x/test_q_v10plw
s v10_fr3_llm A FROM=$VETO -- x_final.py "$AMLC_ROOT/output_v10_fr3_llm" _v10plw 0.70
s llm2_seed A FROM=$VETO SAVE_Q=_v10fr3l -- x_final.py - _v10plw
s wstat_test A -- wstat.py test
s rule_fr A -- rule_fr.py _v10plw
s x_ddfix A -- x_ddfix.py _v10plw _v10fr3l
s v10_fr3_llm_dd A FROM=$VETO RESTORE=restore_dd REJECT=reject_dd -- x_final.py "$AMLC_ROOT/output_v10_fr3_llm_dd" _v10plw 0.70
cd "$AMLC_ROOT/student_resource" && "$PY" utils/validate_submission.py --check-ids --test-dir dataset/test \
    --matching ../output_v10_fr3_llm_dd/matching_results.tsv --candidate ../output_v10_fr3_llm_dd/candidate_pairs.tsv
