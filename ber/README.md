# Business Entity Resolution — reproduction

Pipeline: **fine-tuned bi-encoder blocking → pairwise features → LightGBM → one-S1-per-record assignment**.

## Setup
```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt
```
Base encoder: `intfloat/multilingual-e5-small` (MIT, 118M params), downloaded once from the Hugging Face Hub.
Point `E5=/path/to/snapshot` at a local copy if the machine is offline.

Data layout (set `AMLC_ROOT`, default `~/scratch/AmazonMLChallenge`):
```
$AMLC_ROOT/student_resource/dataset/{train,test}/*.tsv
```
Intermediate files go to `$AMLC_ROOT/work`, outputs to `$AMLC_ROOT/output` (override with `OUT=`).

## Run (from `src/`, one A100 GPU, ~32 CPU cores, ~200 GB RAM)
```bash
python train_embed.py        # fine-tune e5 on S1 folds 0-3            (~8 min)
python retrieve.py train     # embed + top-20 same-country S1 per S2/S3 record
python retrieve.py test
python match.py train        # features, LightGBM on S1 folds 4-8, threshold tuned on fold 9
python match.py test         # writes output/matching_results.tsv and output/candidate_pairs.tsv
```
`common.py` holds loading, text normalisation and the F0.5 scorer (`python common.py` runs its self-checks).

No external data or lookup services are used; country is treated as an open set of labels
(blocking only requires equal country strings, so unseen countries such as France are handled the same way).
