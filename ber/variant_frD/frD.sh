#!/bin/bash
# variant_frD = Sarvesh's E17 + the France fixes of variant_frB. US / India = E17; France = variant_frB.
# E17 (branch sarvesh-exp: e10_prep.py, e10_run.pbs, exp10.py): US / India from the bge + LLM stack retrained on a
# smaller US / India candidate set (shortlist P >= 0.005; France keeps all its pairs: 12,020,996 pairs, 6.94 per S1,
# -5.8%, validation unchanged), with an empty-S1 rescue at q >= 0.5 for the countries with labels.
# France: variant_v10seed_dd's chain (main + LLM stack, three self-training vetoes, same-address fixes) + x_recall.py's
# no-address fix and empty-S1 rule, both for the countries without labels only.
# AMLC_ROOT's work/x must hold test_q_resc17 (exp10.py) and p5_test_resc17 (= the E10 run's p5_test_v10b), the
# variant's test_q_v10plw, test_q_a{1,2,3}pw, restore_fr_dd, reject_fr_dd, and llm_test_v10p (the judge's margins).
# Output: $AMLC_ROOT/output_variant_frD (matching_results md5 76ef7eea, candidate_pairs md5 15fb51cc).
set -e
cd "$(dirname "$0")/../src"
PY=${PY:-python}
FR=unlabelled:_v10plw,unlabelled:_a1pw:min,unlabelled:_a2pw:min,unlabelled:_a3pw:min   # France: the variant's chain
FROM=$FR SAVE_Q=_e17fr $PY x_final.py - _resc17 0.70           # E17 for US / India, the variant's France q
EMPTY_C=unlabelled $PY x_recall.py _e17fr _v10p restore_fr_dd reject_fr_dd   # France: restore_empty, restore_nafr
O="$AMLC_ROOT/output_variant_frD"                               # absolute: student_resource may be a symlink
FROM=$FR RESTORE=restore_fr_dd,restore_empty,restore_nafr REJECT=reject_fr_dd $PY x_final.py "$O" _resc17 0.70
cd "$AMLC_ROOT/student_resource"
$PY utils/validate_submission.py --check-ids --matching "$O/matching_results.tsv" --candidate "$O/candidate_pairs.tsv" \
  --test-dir dataset/test
md5sum "$O"/*.tsv
