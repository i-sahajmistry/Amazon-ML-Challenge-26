# Business Entity Resolution

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.

## Algorithm

Key observation from the training ground truth: **every S2/S3 record belongs to at most one S1 entity**
(7.6M matched ids, none reused), ~26% of S2/S3 records match nothing, and only 5.6% of S1 entities are singletons.
So we solve it record-by-record: *for each S2/S3 record, which S1 entity (if any) is it?*

1. **Normalise** (`common.py`) — transliterate every script to ASCII with `anyascii`
   (`व्हाइट बिल्डर्स` → `vhait bildrs`), lowercase, expand abbreviations (`pvt→private`, `st→street`, `TX→texas`),
   strip website suffixes, and derive a "core" name without legal words (LLC, Pvt Ltd, SARL…).
2. **Blocking with a fine-tuned bi-encoder** (`train_embed.py`, `retrieve.py`) —
   `intfloat/multilingual-e5-small` (MIT, 118M) is fine-tuned for one epoch with in-batch-negative contrastive loss
   on (S2/S3 record, its S1 entity) pairs, using only S1 folds 0–3. Every record is embedded as `name | address`,
   and each S2/S3 record takes its top-20 S1 records **with the same country label** by exact GPU cosine search.
   Country is an open set of strings, so unseen countries (France) are blocked the same way.
   Recall of the true S1 on unseen folds: 97.8% @1, 99.0% @5, 99.5% @20.
3. **Pairwise matcher** (`match.py`) — the top-5 candidates per record (≈50M test pairs) get ~30 features:
   embedding score, rank, gap to the best / next candidate, how contested the S1 is; rapidfuzz ratio / token-set /
   token-sort / partial / Jaro-Winkler on full and core names, a no-space name comparison (website names);
   address ratio / token-set, street-number and PIN/ZIP agreement; empty-field and non-Latin-script flags.
   LightGBM (binary) is trained on S1 folds 4–8.
4. **Assignment** — each S2/S3 record goes to its highest-probability S1 candidate if that probability ≥ a threshold;
   the threshold is chosen by macro F0.5 on held-out S1 fold 9. `candidate_pairs.tsv` is the top-5 set fed to the
   matcher; `matching_results.tsv` is the accepted subset.

Train S1 ids are split into 10 folds by `crc32(id) % 10`: 0–3 embedder, 4–8 matcher, 9 validation — so the
matcher and the reported score never see pairs the embedder was trained on.

## Results (validation = S1 fold 9, macro F0.5)

| Version | Change | F0.5 |
|---|---|---|
| v1 | pipeline above, LightGBM lr 0.1, threshold 0.30 | 0.9884 |
| v2 | lr 0.05, threshold sweep from 0.05 → best 0.20 | **0.9890** |
| ceiling | perfect matcher on the same candidates | 0.9970 |

## Reproduce

### Setup
```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt   # or: torch sentence-transformers lightgbm rapidfuzz anyascii pandas pyarrow
```
Base encoder `intfloat/multilingual-e5-small` (MIT licence) is fetched from the Hugging Face Hub on first use; on an
offline node, download it first and set `E5=/path/to/local/model`.

Data: place (or symlink) the provided `student_resource/` folder next to `src/`, i.e. inside this folder:
```
business_entity_resolution/
├── src/
└── student_resource/dataset/{train,test}/*.tsv     # + utils/validate_submission.py
```
To keep data elsewhere, set `AMLC_ROOT` to the folder that contains `student_resource/`.
Caches, embeddings, models and logs go to `$AMLC_ROOT/work`; outputs to `$AMLC_ROOT/output` (override with `OUT=`).

### Run (from `src/`; one A100 80GB, ~32 cores, ~200 GB RAM)
```bash
python common.py             # self-checks for normalisation and the F0.5 scorer
python prep_norm.py          # normalise all 24M records once (~1 min, cached)
python train_embed.py        # fine-tune e5 on S1 folds 0-3            (~8 min)
python retrieve.py train     # embed + same-country top-20 search      (~25 min)
python retrieve.py test      #                                         (~25 min)
python match.py train        # features, LightGBM, threshold on fold 9 (~20 min)
python match.py test         # -> output/matching_results.tsv, output/candidate_pairs.tsv
python match.py tune         # optional: re-sweep the threshold from cached fold-9 predictions
```
Validate the outputs (run from this folder):
```bash
cd student_resource && python3 utils/validate_submission.py \
  --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

No external data or lookup services are used at any stage.
