# Improvements to merge into `ber-pipeline`

Branch `ber-improvements` = `origin/ber-pipeline` (5bc64a6, v4 without the state table) plus **only changes that
tested better**. Each change is either a pure fix or an opt-in switch: with the switches unset, the pipeline behaves
exactly as before. Experiments and diagnostics behind these numbers live on our branch `v4-analysis`
(`ber/experiments/`); ask if you want them.

All scores below are the v4 stage-2 validation (entities in S1 folds 8-9, distractors at 39%, `stage2.py cv`).

## How to merge

```bash
git fetch origin
git switch ber-pipeline
git merge origin/ber-improvements
```

Files touched: `ber/src/match.py`, `ber/requirements.txt`, this file. No conflict expected unless `_features` in
`match.py` changed on your side.

## 1. Adaptive shortlist: `SHORTLIST=gap:0.1` (tested, better)

**What.** Retrieval gives every S2/S3 record its top-20 S1 by cosine. Before, the top 5 always went to the
matcher. Now, with `SHORTLIST=gap:0.1`, every candidate with `cosine >= best cosine - 0.1` goes (1 to 20 per
record). A clear winner sends 1 pair; a near-tie (chain branches, similar names) keeps the whole tied group, so the
true S1 is no longer cut off at rank 6+. `SHORTLIST=softmax:T:c` also exists (keep candidates until the softmax of
score/T sums to c) but was not run through the full pipeline.

**Evidence.**

| | top-5 (current) | gap 0.1 |
|---|---|---|
| true S1 kept (offline, records of folds 4-9) | 99.03% | **99.43%** |
| candidates per record (average) | 5 | **1.98** |
| stage-2 validation F0.5 (thr 0.70) | 0.98799 | **0.98906** (+0.0011) |
| loss from missed matches | 0.00934 | 0.00844 |
| loss from merged distractors | 0.00146 | 0.00137 |
| test records accepted | 5.60M | 5.89M (3.40 per S1; train truth 3.46) |
| `candidate_pairs.tsv` | 635 MB | **306 MB** |

**How to use.**
1. Delete the cached features and everything built from them (the cache name does not include `KF`/`SHORTLIST`,
   so a stale top-5 cache would be reused silently):
   `work/feats2_train.parquet work/feats2_test.parquet work/oof_train.parquet work/p1_test.parquet
   work/gbm_A.txt work/gbm_B.txt work/gbm2.txt work/gbm2_cfg.json`.
   Keep `work/pq/norm2_*` and `work/cand_*` (normalisation and retrieval are unchanged).
2. Run with the switch set for the whole run:
   ```bash
   export SHORTLIST=gap:0.1
   python stage1_cv.py && python stage2.py cv && python stage2.py test
   ```
3. `candidate_pairs.tsv` then lists exactly the shortlisted pairs (what the model saw), as the README requires.

Once reproduced, consider making `gap:0.1` the default in `match.py`.

## 2. `requirements.txt`: add `numba` and `llvmlite` (fix)

`decide.py` imports `numba`, and `stage2.py` imports `decide`, so stage 2 crashes on a clean environment
(`ModuleNotFoundError: numba`). Added `numba==0.67.0` and `llvmlite==0.49.0`; a pip dry run confirmed they keep
`numpy==2.5.3`. Needed for the final submission zip to reproduce.

## 3. Pending: stage 2 without duplicated distractors (`DISTRACTORS=weight`)

**Finding.** `stage2.train_rows` reaches the 39% distractor share by drawing about 4.9M distractor rows **with
replacement** from about 2.7M distinct ones (about 1.8 copies each; most distractors appear 2+ times). The copies of
one record point to the same S1 and `context()` counts them as that S1's other records (`e_n`, `e_ge*`, `e_sum`). So
the model learns and is validated on a cue test does not have (every test record appears once).
`harness.build` duplicates the same way. This is our lead suspect for validation 0.988 vs leaderboard 0.967.

**Change (commit 2a2c1fb on `v4-analysis`, added here only if it wins).** `DISTRACTORS=weight`: every distractor once
with row weight = average copies (about 1.8), in the LightGBM loss, in the validation precision
(`harness.score(..., wt=)`), in the threshold choice and in the refit. Default `dup` = current behaviour.

**Status.** Being tested now: current vs weighted stage 2, both scored on the same no-copies validation weighted to
39%. Note: that validation is harder than the current one (no copies to lean on), so its number is **not comparable
to 0.989**; compare models on the same view only.

## Other findings (no code change)

- **Leaderboard.** v4 with KF=5, no state table (validation 0.98799, thr 0.70): **0.967**. The official validator
  passed.
- **Test distractor share is about 40% in every country**, from record counts (assuming train's 3.46 matches per S1)
  and from the mix of stage-1 probabilities. The 39% assumption holds, and test distractors are not harder.
- **France looks like US/India in the predictions**: accepted records / expected matches 95% / 94% / 93%
  (France / India / US); records per S1 3.28 / 3.25 / 3.20; empty S1 5.9% / 6.1% / 6.3%. The one difference is more
  retrieval near-ties (median top-1 minus top-2 cosine: France 0.21, India 0.29, US 0.38), which is exactly where
  the gap shortlist helps.
- **Missed matches are the largest loss** in every country: about 3.2 records per S1 predicted vs 3.46 true on
  train.
- **Stage-1 judge A trained unstably** in the KF=5 run (validation logloss jumped around iteration 300, stopped at
  263 vs judge B at 1412). Worth a look.
- **Caches after 5bc64a6** (state table removed): re-runs must delete `work/pq/norm2_*`, `work/feats2_*`,
  `gbm_A.txt`, `gbm_B.txt`, `oof_train.parquet`, `p1_test.parquet`, `gbm2*`. The 5bc64a6 commit message wrongly
  names `norm_*`.
