#!/bin/bash
# The problem statement's "Final Submission Package": $AMLC_ROOT/submission/<TEAM>_submission/ and a zip of its contents.
#   AMLC_ROOT=/folder/with/student_resource PY=python bash package.sh TEAM [DIR with matching_results.tsv, candidate_pairs.tsv]
# DIR defaults to $AMLC_ROOT/output, which reproduce.sh writes. Documentation_template.md and pipeline.png come from this
# folder when present, else the provided blank template.
set -e
TEAM=${1:?team name}; R=${AMLC_ROOT:?set AMLC_ROOT}; OUT=${2:-$R/output}; PY=${PY:-python}
B=$(cd "$(dirname "$0")" && pwd); S=$R/submission/${TEAM}_submission; C=$S/code/business_entity_resolution
rm -rf "$S"; mkdir -p "$S/output" "$C"
cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$S/output/"
cp -r "$B/src" "$B/README.md" "$B/requirements.txt" "$B/reproduce.sh" "$C/"
rm -rf "$C/src/__pycache__"
if [ -e "$B/Documentation_template.md" ]; then cp "$B/Documentation_template.md" "$S/"
else cp "$R/student_resource/Documentation_template.md" "$S/"; fi
[ -e "$B/pipeline.png" ] && cp "$B/pipeline.png" "$S/"
cd "$R/student_resource" && "$PY" utils/validate_submission.py --check-ids --test-dir dataset/test \
    --matching "$S/output/matching_results.tsv" --candidate "$S/output/candidate_pairs.tsv"
"$PY" -c "import shutil, sys; print(shutil.make_archive(sys.argv[1], 'zip', sys.argv[2]))" "$R/submission/${TEAM}_submission" "$S"
