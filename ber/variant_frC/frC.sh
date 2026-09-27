#!/bin/bash
# variant_frC = variant_frB with US / India from the bge + LLM stack (Sarvesh's E1: US / India validation 0.99294 vs
# 0.99282). France keeps variant_v10seed_dd's chain (main + LLM stack, three self-training vetoes, same-address fixes)
# and frB's no-address fix, so France is identical to variant_frB.
# The bge + LLM stack x/test_q_v10bplw: `x_chain.py v10b` (branch v11: bge-reranker-v2-m3, Apache-2.0, 568M, as a third
# cross-encoder, then stages 1-2), then `x_llmstack.py _v10p _v10bp` (the LLM judge's v10 scores reused for the pairs).
# Runs from ber/src with AMLC_ROOT = the variant's root (see ../variant_frB/frB.sh), whose work/x also holds
# test_q_v10bplw and p5_test_v10bplw (= p5_test_v10: the same 12,760,925 pairs).
# Output: $AMLC_ROOT/output_variant_frC (matching_results md5 7d66d4ea, candidate_pairs md5 1a8b4f5c).
set -e
cd "$(dirname "$0")/../src"
PY=${PY:-python}
FR=unlabelled:_v10plw,unlabelled:_a1pw:min,unlabelled:_a2pw:min,unlabelled:_a3pw:min   # France: the variant's chain
FROM=$FR SAVE_Q=_e1fr $PY x_final.py - _v10bplw 0.70          # bge + LLM for US / India, the variant's France q
$PY x_recall.py _e1fr _v10p restore_fr_dd reject_fr_dd        # -> x/restore_empty (on these q), x/restore_nafr
O="$AMLC_ROOT/output_variant_frC"                              # absolute: student_resource may be a symlink
FROM=$FR RESTORE=restore_fr_dd,restore_empty,restore_nafr REJECT=reject_fr_dd $PY x_final.py "$O" _v10bplw 0.70
cd "$AMLC_ROOT/student_resource"
$PY utils/validate_submission.py --check-ids --matching "$O/matching_results.tsv" --candidate "$O/candidate_pairs.tsv" \
  --test-dir dataset/test
md5sum "$O"/*.tsv
