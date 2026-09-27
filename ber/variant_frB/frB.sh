#!/bin/bash
# variant_frB = variant_v10seed_dd (leaderboard 0.990349; built by ../variant_v10seed_dd/variant.sh on the original
# run's v10 files) + the two recall lists of x_recall.py. Runs from ber/src with AMLC_ROOT = the variant's root, whose
# work/x holds test_q_v10plw, test_q_a{1,2,3}pw, p5_test_v10plw, restore_fr_dd, reject_fr_dd and llm_test_v10p.
# Output: $AMLC_ROOT/output_variant_frB (matching_results md5 73b4a5ea, candidate_pairs md5 1a8b4f5c = the variant's).
set -e
cd "$(dirname "$0")/../src"
PY=${PY:-python}
VETO=unlabelled:_a1pw:min,unlabelled:_a2pw:min,unlabelled:_a3pw:min
FROM=$VETO SAVE_Q=_varfr $PY x_final.py - _v10plw 0.70       # the variant's q after the three self-training vetoes
$PY x_recall.py _varfr _v10p restore_fr_dd reject_fr_dd       # -> x/restore_empty, x/restore_nafr
O="$AMLC_ROOT/output_variant_frB"                              # absolute: student_resource may be a symlink
FROM=$VETO RESTORE=restore_fr_dd,restore_empty,restore_nafr REJECT=reject_fr_dd $PY x_final.py "$O" _v10plw 0.70
cd "$AMLC_ROOT/student_resource"
$PY utils/validate_submission.py --check-ids --matching "$O/matching_results.tsv" --candidate "$O/candidate_pairs.tsv" \
  --test-dir dataset/test
md5sum "$O"/*.tsv
