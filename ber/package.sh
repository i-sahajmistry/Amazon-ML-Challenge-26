#!/bin/bash
# Build <team>_submission.zip in $AMLC_ROOT/submission. usage: package.sh TEAM_NAME
set -e
TEAM=${1:?team name}
R=${AMLC_ROOT:-$HOME/scratch/AmazonMLChallenge}
S=$R/submission/stage; rm -rf $S; mkdir -p $S/output $S/code/business_entity_resolution
cp $R/output/matching_results.tsv $R/output/candidate_pairs.tsv $S/output/
cp -r $R/ber/src $R/ber/README.md $S/code/business_entity_resolution/
rm -rf $S/code/business_entity_resolution/src/__pycache__
$HOME/scratch/miniconda3/envs/amlc/bin/pip freeze | grep -iE "^(torch|numpy|numba|llvmlite|faiss-cpu|pandas|pyarrow|lightgbm|rapidfuzz|anyascii|sentence-transformers|transformers|datasets|accelerate|peft|scikit-learn)==" \
  > $S/code/business_entity_resolution/requirements.txt
cp $R/ber/Documentation_template.md $S/ 2>/dev/null || cp $R/student_resource/Documentation_template.md $S/
cd $R/student_resource && python3 utils/validate_submission.py --matching $S/output/matching_results.tsv \
  --candidate $S/output/candidate_pairs.tsv --test-dir dataset/test
cd $S && rm -f ../${TEAM}_submission.zip && zip -qr ../${TEAM}_submission.zip . && ls -la ../${TEAM}_submission.zip
