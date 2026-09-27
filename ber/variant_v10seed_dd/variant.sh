#!/bin/bash
# Simplified France chain on the original run's v10 base (one leaderboard probe): the three self-training rounds are
# seeded from v10's own decisions and start from v10's cross-encoders, instead of v8 -> v9p_frand -> v10_fr2 with v8's.
# Same round recipe as v9s / v9s2 / v10s3 (x_ce3 SELF fit, 3M train + 4M pseudo pairs, lr 2e-5, stages with peers).
# The LLM blend uses the judge's scores on every unsure v10 row (x/llm_{val,test}_v10p). Then the dd fixes on top.
# Inputs are symlinked read-only from ~/scratch/AmazonMLChallenge/work; everything written stays in this sandbox.
set -u
R=$HOME/scratch/amlc_variant; W=$R/work; X=$W/x; L=$W/logs
PY=$HOME/scratch/miniconda3/envs/amlc/bin/python
cd $R/ber/src
export AMLC_ROOT=$R HF_HUB_OFFLINE=1 HF_HOME=/scratch/scai/mtech/aib262144/hf OMP_NUM_THREADS=24
G=${GPU:-GPU-366eff51-004b-3fd5-2cb1-a25c42617f3e}
s() {   # s NAME [VAR=value ...] -- python-args
    local n=$1; shift; local e=(); while [ "$1" != -- ]; do e+=("$1"); shift; done; shift
    [ -e $L/$n.done ] && return 0
    [ -e $L/FAILED ] && exit 1
    echo "$(date '+%F %T') start $n"
    if env ${e[@]+"${e[@]}"} CUDA_VISIBLE_DEVICES=$G $PY -u "$@" > $L/$n.log 2>&1; then touch $L/$n.done; echo "$(date '+%F %T') done  $n"
    else echo $n >> $L/FAILED; echo "$(date '+%F %T') FAILED $n"; exit 1; fi
}
SELF="DFOLD=1 CE_N=3000000 SELF_N=4000000 CE_LR=2e-5"
round() {   # round K SEED_FILE
    local k=$1 seed=$2 RA="DFOLD=1 CE_TEXT=raw CE_DIR=$X/ce_ra$1 CE_TAG=_ra$1" NA="DFOLD=1 CE_DIR=$X/ce_na$1 CE_TAG=_na$1"
    ( s a${k}_ra_fit $RA CE_INIT=$X/ce_r10 SELF=$seed $SELF -- x_ce3.py train ) &
    ( s a${k}_na_fit $NA CE_INIT=$X/ce_n10 SELF=$seed $SELF -- x_ce3.py train ) & wait
    ( s a${k}_ra_train $RA -- x_ce3.py score train; s a${k}_ra_test $RA -- x_ce3.py score test ) &
    ( s a${k}_na_train $NA -- x_ce3.py score train; s a${k}_na_test $NA -- x_ce3.py score test ) & wait
    [ -e $L/FAILED ] && exit 1
    s a${k}_s1 DFOLD=1 CE_TAGS=_na$k,_ra$k TAG=_a$k S1K=3 -- x_stage_multi.py s1
    s a${k}_s2p DFOLD=1 CE_TAGS=_na$k,_ra$k TAG=_a$k BAG=5 PEERS=1 -- x_nocopy.py test
}
round 1 $X/test_q_v10pw.parquet
s a2_seed FROM=France:_a1pw:min SAVE_Q=_a2seed -- x_final.py - _v10pw
round 2 $X/test_q_a2seed.parquet
s a3_seed FROM=France:_a1pw:min,France:_a2pw:min SAVE_Q=_a3seed -- x_final.py - _v10pw
round 3 $X/test_q_a3seed.parquet
VETO=France:_a1pw:min,France:_a2pw:min,France:_a3pw:min
s llm_stack -- x_llmstack.py _v10p _v10p
s fr3_llm FROM=$VETO -- x_final.py $R/output_variant_fr3_llm _v10plw 0.70
s llm2_seed FROM=$VETO SAVE_Q=_v10fr3l -- x_final.py - _v10plw
s wstat_test -- wstat.py test
s rule_fr -- rule_fr.py _v10plw
s x_rule1 -- x_rule1.py
s x_wordlists -- x_wordlists.py
s dd FROM=$VETO RESTORE=restore_fr_dd REJECT=reject_fr_dd -- x_final.py $R/output_variant_dd _v10plw 0.70
cd $AMLC_ROOT/student_resource && $PY utils/validate_submission.py --check-ids --test-dir dataset/test \
    --matching ../output_variant_dd/matching_results.tsv --candidate ../output_variant_dd/candidate_pairs.tsv
md5sum $R/output_variant_dd/*.tsv
cd $R && AMLC_ROOT=$R $PY $HOME/scratch/amlc_final/compare.py $R/output_variant_dd/matching_results.tsv \
    $HOME/scratch/AmazonMLChallenge/output_v10_fr3_llm_dd/matching_results.tsv
echo "$(date '+%F %T') all done"
