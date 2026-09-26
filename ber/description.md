# v9fS: Sahaj's v9 merged with v8-generalize, plus self-training for unlabelled countries

Branch `v9fS`, built on `ber-pipeline` 4d7e85b (Sahaj's v8). It reproduces his unpushed v9 and adds the parts of
`v8-generalize` that still help on top of it.

## Starting points

| Model | Branch | Copy-free validation | Leaderboard | Test pairs (per S1) |
|---|---|---:|---:|---:|
| v7w | ber-pipeline ff64094 | 0.99240 | 0.982641 | 49.8M (28.8) |
| Sahaj v8 | ber-pipeline 4d7e85b | 0.99236 | – | 17.25M (10.0) |
| Sahaj v9 | not in git; work folder copied to PADUM | 0.99254 | **0.985** | 12.76M (7.4) |
| Sahaj v9p | not in git ("peers" stage-2 features) | 0.99279 | – | 12.76M (7.4) |
| our v8 / v8J | v8-generalize 9c2760f | 0.99267 / 0.99276 | – | 16.5M (9.5) |

Rules that changed on 2026-09-26 (organiser answers):
- `candidate_pairs.tsv` now counts toward the final ranking: fewer candidates per S1 ranks higher.
- Self-training and pseudo-labelling on the test records are allowed.
- Small hand-written normalisation dictionaries are allowed.
- The public and private leaderboards score different subsets of the same test file.

## What was merged and why

1. **v9's text-aware shortlist, reproduced in code** (`match.py` `SHORTLIST=text:0.001`, `shortlist.py`).
   - v9's shortlist code was not in git. `analysis/v9/g8_shortlist_combo.py` applies v9's saved model
     (`shortlist_text.txt`) to every retrieved test pair.
   - At threshold 0.001 it keeps exactly v9's 12,760,925 test pairs: 100% identical.
   - The same study compared three shortlist models (train records of S1 folds 7-9, tau 0.005):

   | Shortlist model | True S1 kept | Per record | Per S1 |
   |---|---:|---:|---:|
   | retrieval only (v8) | 99.281% | 1.341 | 6.27 |
   | + text similarities (v9) | 99.360% | 1.167 | 5.46 |
   | + text + S1 side (v8-generalize) | 99.364% | 1.155 | 5.40 |

   Once the text features are in, the S1-side features add almost nothing, so the branch keeps v9's model.

2. **Distractor-word features** (`words.py`, from v8-generalize).
   - Every word a record adds to its S1's name is scored from label-free per-country statistics plus a
     multilingual-embedding k-NN, leave-one-country-out.
   - On top of v9's name-edit "suspicion" features, validation moves from 0.99254 to 0.99257. The two overlap.

3. **Stage 2 scored per stage-1 model** (`stage2.score_per_model`).
   - Stage 2 is trained on single-model out-of-fold probabilities, but test pairs carried the mean of the models.
   - It now runs once per model (`pm0`, `pm1`, `pm2` saved by stage 1) and the results are averaged. This only
     changes test predictions.

4. **Self-training for countries without training labels** (`SELFTRAIN=1`, `x_stage_multi.self_train`).
   - Which records: test countries with no labelled training records, found from the data. Here that is France,
     with 1.42M records.
   - Pseudo-labels from the stage-1 mean:
     - 825k records whose best pair has p >= 0.97 match that pair;
     - 430k records whose best pair has p <= 0.03 match nothing.
   - Two stage-1 models, each also trained on the pseudo-labels of one half, re-score the other half.
   - Simulation with one train country hidden (`analysis/v9/g7_selftrain_loco.py`, stage 1; pseudo-labels
     99.0-99.9% correct):

   | Hidden country | Source only | + pseudo-labels | With its own labels |
   |---|---:|---:|---:|
   | India (close to US) | 0.99339 | 0.99334 | 0.99376 |
   | US (further from India) | 0.98756 | **0.98811** | 0.99211 |

   France is further from both than they are from each other, so the gain should resemble the US row.

5. **Decision judge** (`x_judge.py`, optional). On the same rows, LightGBM, XGBoost and their average score
   0.99258, 0.99254 and 0.99259. These are within noise, so v9fS keeps LightGBM.

## v9fX: cross-encoders self-trained on France

Stage-1 self-training (v9fS) only re-weights existing features. The cross-encoders are the text models, and they
learned to read names and addresses from US / India pairs only. v9fX adapts them to the unlabelled country.

1. **Pseudo-labels** come from v9f's stage-2 test scores (no stage-1 self-training), for the 1.42M France records.
   - 828k records with q >= 0.95: the argmax pair is a match, and the record's other shortlisted pairs are hard
     negatives.
   - 450k records with q <= 0.05: none of their pairs match.
   - This gives 1.91M labelled pairs, 43% positive.
2. **Fine-tuning** (`x_ce_st.py`):
   - Each cross-encoder (normalised and raw text) continues from its trained weights for one epoch, at LR 2e-5.
   - It trains on one half of those records plus the same number of its own labelled training pairs (retrieved
     top-5 of S1 folds 0-3), so it keeps US / India.
   - It then re-scores the other half. Two halves per cross-encoder, about 10 minutes each on one A100.
3. **Stages 1-2 are re-run** with the re-scored test pairs (`CE_TEST_SUFFIX=st`). Train pairs, and therefore the
   stage models' training data, are unchanged.

What changed on France (US / India are unchanged, validation still 0.99257):

| France | v9f | v9fX |
|---|---:|---:|
| accepted at 0.70 | 871,196 | 867,161 (3.34 per S1) |
| confident match (q >= 0.95) | 58.1% | 59.0% |
| confident no-match (q <= 0.05) | 31.6% | 35.0% |
| uncertain (0.3 < q < 0.9) | 5.25% | 3.15% |
| same argmax S1 as v9f | – | 95.5% |
| accepted by only one of the two | – | 18.5k v9f-only, 14.5k v9fX-only |

v9fX is the variant most likely to move France, up or down. Compare its leaderboard score with v9f's.

## Candidate-set size

Every row of the merged model's validation filtered by the shortlist probability, as a stricter shortlist would
(`analysis/v9/g9_text_tau.py`):

| tau | F0.5 at 0.70 | Change | Test pairs | Per S1 | US / India / France per S1 |
|---:|---:|---:|---:|---:|---|
| **0.001 (v9fS)** | 0.99257 | 0 | 12.76M | 7.37 | 6.6 / 7.1 / 10.1 |
| 0.002 | 0.99257 | -0.00001 | 12.12M | 6.99 | 6.5 / 6.8 / 8.9 |
| 0.005 | 0.99255 | -0.00003 | 11.35M | 6.55 | 6.3 / 6.5 / 7.5 |
| 0.01 | 0.99252 | -0.00006 | 10.87M | 6.27 | 6.2 / 6.2 / 6.8 |
| 0.02 | 0.99246 | -0.00012 | 10.33M | 5.96 | 5.9 / 5.9 / 6.1 |

- tau 0.005 would cut candidates by 11% for -0.00003.
- It cuts France hardest, and validation cannot show France's cost.
- The file can be filtered after the fact without retraining.

## Test outputs (all pass `validate_submission.py --check-ids`)

| File | Accepted | India | US | France | Empty S1s |
|---|---:|---:|---:|---:|---:|
| v9f (v9fS without self-training) | 5,884,325 | 2,752,808 | 2,260,321 | 871,196 | 101,362 |
| v9fJ (judge average, 0.66) | 5,893,612 | 2,754,935 | 2,263,605 | 875,072 | – |
| **v9fS** | 5,889,020 | 2,752,808 | 2,260,321 | **875,891** | – |
| v9fX | 5,880,290 | 2,752,808 | 2,260,321 | 867,161 | – |

Matches per S1 are 3.38-3.40 in every country; the train truth is 3.46. France is not over-accepted: 3.38 per S1
against 3.40 for US / India.

**Expected leaderboard: about 0.9855-0.9860** (v9: 0.985).
- Word features and per-model scoring: +0.0000 to +0.0005.
- Self-training on France: +0.0000 to +0.0010.
- Submitting v9f and v9fS one after the other measures self-training directly.

## Still open

- **v9p "peers" features** (Sahaj, 0.99279 on validation): merge once the code is pushed.
- **Leaderboard comparison:** submit v9f, v9fS and v9fX. They differ only on France, so the score differences are
  the France effects of the two kinds of self-training. Then combine the ones that help.
- **`bge-reranker-v2-m3`** (Apache-2.0, 568M, multilingual) as a third cross-encoder: parked.
  - After 500k training pairs, its fold-9 argmax accuracy is 0.98173, against 0.98169 / 0.98227 for the e5
    normalised / raw cross-encoders (`analysis/v9/g10_ce_compare.py`).
  - So it is no better on US / India, and its France value could only be measured by a leaderboard run.
- **Final package:** one clean `python x_chain.py v9m` run from this branch.

## Files

- **New:**
  - `src/words.py`, `src/x_judge.py`, `src/x_ce_st.py` (v9fX).
  - `analysis/v9/g7_selftrain_loco.py`, `g8_shortlist_combo.py`, `g9_text_tau.py`, `g10_ce_compare.py`,
    `g11_report.py`.
- **Changed:**
  - `match.py`: `text_feats`, `SHORTLIST=text:tau`.
  - `shortlist.py`: fits the retrieval and text models.
  - `stage2.py`: `record_inputs`, `score_per_model`.
  - `x_stage_multi.py`: word features, `pm*` columns, `self_train`, `CE_TEST_SUFFIX`.
  - `x_nocopy.py`: per-model test scoring.
  - `x_final.py`: `THR=auto`.
  - `x_chain.py`: plans `v9m` (= v9fS) and `v9mX` (= v9fX).
  - `x_ce.py` / `x_ce3.py`: the fold-9 check is optional, so cross-encoders can train before the features exist.
  - `retrieve.py`: FAISS threads from `NT`.
  - `common.py`: fork start method for Python 3.14.
  - `requirements.txt`: `xgboost-cpu`.
