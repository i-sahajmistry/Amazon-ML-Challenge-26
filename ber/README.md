# Business Entity Resolution

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.

**Leaderboard best so far: v7w, 0.982641** (2026-09-25; branch `ber-pipeline`). A France probe puts US + India at
about 0.991 and France at about 0.933, so France is where the remaining gap is (see [Leaderboard](#leaderboard)).

**This branch (`v8-generalize`): v8**, the same architecture with every hand-written, country-specific piece replaced
by a procedure that runs identically on every country label. It adds a learned blocking shortlist, a distractor-word
model, and an optional XGBoost decision judge. Validation 0.99267 copy-free (v7w 0.99240); leaderboard pending. What
was done and why: [description.md](description.md).

## Algorithm (v8)

Key observation from the training ground truth: **every S2/S3 record belongs to at most one S1 entity**
(7.6M matched ids, none reused), ~26% of S2/S3 records match nothing, and only 5.6% of S1 entities are singletons.
So we solve it record-by-record: *for each S2/S3 record, which S1 entity (if any) is it?*

`country` is treated as an open set of labels. Nothing in the pipeline names a country. Every country-specific
dictionary is **mined from the data** under that country's label, so an unseen country gets its own in the same way.

1. **Normalise** (`common.py`, dictionaries from `mine_dicts.py`): transliterate every script to ASCII with
   `anyascii`, lowercase, strip website suffixes, expand abbreviations and derive a "core" name without legal
   words.
   - The dictionaries are mined per country from confident retrieval pairs. For an abbreviation, a token on one
     side only must be a same-initial subsequence of a longer token on the other side. For a legal form, a short
     token must be dropped often, in both directions.
   - Mined examples: US tx→texas, st→street; India mh→maharashtra, prvte→private; France r→rue, imp→impasse,
     sas/sasu/sci as legal forms.
   - Only generic English defaults are written by hand (Rd/Road, Pvt/Private, the noise the problem statement
     lists).
2. **Blocking** (`train_embed.py`, `retrieve.py`, `shortlist.py`):
   - `intfloat/multilingual-e5-small` (MIT, 118M) is fine-tuned with in-batch negatives on (record, its S1) pairs
     of S1 folds 0–3.
   - Each record takes its top-20 S1 records **with the same country label** by exact GPU cosine search.
   - A **learned shortlist** then decides how many go on. A LightGBM model on retrieval-only features (the
     record's own list, and the S1's list of records pointing at it) keeps the best candidate plus every candidate
     with probability ≥ 0.001.
   - It keeps the true S1 for 99.47% of matched records at 1.54 candidates per record (top-5: 99.02% at 5). That
     is 16.5M test pairs instead of 49.8M. Near-ties keep more candidates: France 2.2 per record, US 1.2.
3. **Pair features:**
   - From `match.py features`: embedding score, rank and gaps; rapidfuzz similarities on full, core and
     consonant-skeleton names and on addresses; numbers, token rarity and name frequency.
   - From `x_feats.py`: integer-aware house numbers, and legal-form bits including "a legal form mined for this
     country".
   - From `words.py`: **distractor-word features**. A model scores each word a record *adds* to its S1 name
     (label-free statistics plus a multilingual-embedding k-NN). The scores are leave-one-country-out, so train
     features are as uncertain as an unseen country's. For France it finds groupe, holding, developpement,
     international, … with no word list.
4. **Two cross-encoders**, fine-tuned only on pairs of S1 folds 0–3:
   - `x_ce.py`: e5-small started from the fine-tuned bi-encoder, on normalised `name | address`.
   - `x_ce3.py`: e5-small on raw transliterated text.

   Each adds its score, its rank within the record, the gap to the record's next candidate, its rank within the
   S1 and the S1's positive claims. (v7's e5-base cross-encoder is dropped: it added 0.00003.)
5. **Stage 1** (`x_stage_multi.py s1`): LightGBM on pair, cross-encoder and word features, cross-fitted: model A
   on records of S1 folds 4–6, B on 7–9. Each train pair gets the probability of the model that didn't see it;
   test keeps both.
6. **Stage 2** (`x_nocopy.py`, or `x_judge.py` to compare against XGBoost):
   - Each record's best candidate is re-scored with entity context: the other records claiming the same S1 (how
     many above 0.9 / 0.5 / 0.2, their sum and max, rank, same-source claims).
   - Distractors appear once, **weighted** (3.04) to the 39% test share.
   - At test it is scored once with model A's and once with model B's probabilities, and averaged, because it
     was trained on single-model probabilities.
7. **Decision** (`x_final.py`): accept a record if its stage-2 score clears one threshold (0.70), the same for
   every country. `candidate_pairs.tsv` lists exactly the shortlisted pairs that were scored.

Folds: `crc32(S1 id) % 10`; 0–3 train the bi-encoder and the cross-encoders, 4–9 train stages 1 and 2. Stage 2 is
validated on entity folds 8–9 after fitting on 4–7. Distractors follow their own hash fold, and those in 0–3 are left
out of stage 1/2 training and validation because the cross-encoders saw them.

## Results

Validation: entity folds 8–9, distractors at the test share of 39%, macro F0.5. Two views:
- **copies**: distractors repeated to reach 39% (how v4–v7 were validated). Reads about 0.0016 high.
- **copy-free**: each distractor once, weighted to 39%. This is the view to compare models on.

| Version | Change | Validation | Leaderboard |
|---|---|---|---|
| v1 | bi-encoder blocking + LightGBM matcher | 0.9884 (fold 9, train distractor share) | – |
| v2 | lr 0.05, threshold 0.20 | 0.9890 (same) | – |
| v3 | intermediate build | – | upload failed (portal) |
| v4 | richer pair features, cross-fitted stage 1, entity-context stage 2 | 0.98861 copies | 0.967214 (Sarvesh's run: top 5, no state map) |
| v5 | + small cross-encoder, integer / legal-form features | 0.99273 copies, 0.99108 copy-free | – |
| v5f | v5 + France word rule, France threshold 0.80 | as v5 | – |
| v6 | + e5-base cross-encoder | 0.99276 copies | – |
| v7 | + raw-text cross-encoder | 0.99304 copies | – |
| v5w | v5, stage 2 without distractor copies | 0.99186 copy-free | – |
| **v7w** | v7, stage 2 without copies, threshold 0.70 everywhere + France word rule | **0.99240 copy-free** | **0.982641** |
| **v8** | compliant rebuild (this branch): mined per-country dictionaries, learned shortlist, word model, 2 cross-encoders, per-model stage 2, one rule for all countries | **0.99267 copy-free**, 0.99332 copies | pending |
| v8J | v8 with LightGBM + XGBoost averaged as the stage-2 judge, threshold 0.68 | 0.99276 copy-free | pending |

Tried without gain: word-edit log-likelihood features (`x_llr.py`, 0.99274 vs 0.99273), re-ranking each record's top
2 (`x_top2.py`, 0.99278 either way), an expected-F0.5 decision per S1 instead of a threshold (v3: 0.9775 vs 0.9788).

Cross-encoders on their own (fold 9, share of real records whose argmax is the right S1): normalised e5-small 0.98237,
e5-base 0.98211, raw-text e5-small 0.98290; pair AUC 0.99984–0.99986. They add value through stacking. v8 (trained
on the learned shortlist, mined normalisation): normalised 0.98282, raw text 0.98304.

Decision judge on v8 (`x_judge.py`, copy-free validation): LightGBM 0.99270, XGBoost 0.99271, their average 0.99276
(best thresholds 0.66 / 0.68 / 0.68). XGBoost matches LightGBM; the average is within noise of both.

Where the remaining validation errors are (`x_errors.py` and the other `x_*.py` diagnostics, v5): 73% of missed true matches and 89.5% of
blocking misses are records with an empty address whose name is shared by several S1 entities.

## Leaderboard

| File | Score |
|---|---|
| v8, v8J | pending |
| v7w | 0.982641 |
| v7w with every France S1 left empty (probe) | 0.851155 |
| v4, Sarvesh's run | 0.967214 |
| Rank 1 (2026-09-25 ~22:45) | 0.988319 |

An empty prediction scores 1 on an S1 with no true matches and 0 otherwise. France is s = 259,452 / 1,732,544 =
14.975% of test S1s, and about e = 5.5% of them should have no match (train prior and our predictions). So:
- US + India ≈ (probe − s·e) / (1 − s) = **0.991** (validation says 0.992),
- France ≈ (best − probe) / s + e = **0.933** (±0.005 from e).

France costs about 0.009 of the overall score. At the US/India level the total would be about 0.991.

What changed from v4 (0.967) besides the models, from teammates' reviews:
- Mohanish: v4's stage 2 learned that an *averaged* stage-1 probability marks a distractor (only distractors of hash
  folds 0–3 had one in training), and at test every probability was averaged. v5 onwards train stage 2 on single-model
  out-of-fold probabilities only, so they are not affected.
- Sarvesh and Mohanish: repeated distractors inflate validation (v5: 0.99266 → 0.99108 at the same threshold) and
  training without them helps (`x_nocopy.py`: v5 0.99108 → 0.99186, v7 → 0.99240).
- Sarvesh's adaptive shortlist (`SHORTLIST=gap:0.1`, branch `ber-improvements`) gained 0.0011 on v4. v8's learned
  shortlist keeps more true S1s (99.47% vs 99.42%) with fewer candidates (1.54 vs 1.98 per record).
- v8 (this branch): no hand-written country-specific dictionaries or rules; see [description.md](description.md).

`x_compare.py` compares submission files without labels (matches per S1, empty S1s, and macro F0.5 of one file scored
against another). Scored against v5f as if it were the truth, Sarvesh's 0.967 file gets 0.968.

## Reproduce

### Setup
```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt
```
Model (MIT licence): `intfloat/multilingual-e5-small`, fetched on first use (`E5=/path` for an offline node). v8 needs
no other model (`x_ce2.py`, the e5-base cross-encoder of v6/v7, is kept but not used).

Data: place (or symlink) the provided `student_resource/` folder next to `src/`, or set `AMLC_ROOT` to the folder that
contains it. Caches, models and logs go to `$AMLC_ROOT/work`. XGBoost for `x_judge.py`: `pip install xgboost-cpu`
(Apache-2.0).

### Run v8 (from `src/`; one A100 80GB, 16 cores, ~200 GB RAM; about 4.5 h end to end)
```bash
NO_DICTS=1 python common.py          # self-checks for normalisation and the F0.5 scorer (generic defaults only)
python train_embed.py                # fine-tune the e5-small bi-encoder on S1 folds 0-3        (~10 min)
python retrieve.py train             # embed + same-country top-20 search                        (~25 min)
python retrieve.py test              #                                                          (~25 min)
python x_chain.py v8                 # everything below, in order                               (~3 h 40 min)
```
`x_chain.py v8` runs, each step logging to `work/logs/<step>.log` (restart from a step with `python x_chain.py v8 <step>`):

| Step | Script | What | Time (16 CPUs, 1 A100) |
|---|---|---|---|
| v8_mine_dicts | `mine_dicts.py` | per-country abbreviation and legal-form dictionaries from confident retrieval pairs | 3 min |
| v8_prep_norm | `prep_norm.py` | normalise every record with them (cached `work/pq/norm2_*`) | 3 min |
| v8_shortlist | `shortlist.py` | fit the learned shortlist (`work/shortlist_gbm.txt`) | 13 min |
| v8_stage1_v4 | `stage1_cv.py` | pair features on the shortlisted pairs (`SHORTLIST=learned:0.001`) + the v4 stage 1 used by the cross-encoder check | 45 min |
| v8_x_feats_* | `x_feats.py` | integer and legal-form features | 4 min |
| v8_words | `words.py` | distractor-word model and pair features | 14 min |
| v8_ce_* | `x_ce.py` | normalised-text cross-encoder: train on folds 0-3, score | 63 min |
| v8_ce_raw_* | `x_ce3.py` | raw-text cross-encoder | 60 min |
| v8_stage1 | `x_stage_multi.py s1` | cross-fitted stage 1 (keeps both models' test probabilities) | 7 min |
| v8_stage2_* | `x_nocopy.py` | copy-free stage 2: validation table, then refit + per-model test scoring | 7 min |
| v8_final | `x_final.py` | `../output_v8/{matching_results,candidate_pairs}.tsv` at 0.70 | 1 min |

Optional decision judge (LightGBM vs XGBoost vs their average on the same rows; the best one on validation):
```bash
export TAG=_v8 CE_TAGS=,_raw WORDS=1
python x_judge.py && JUDGE=best python x_judge.py test && python x_final.py ../output_v8J _v8J auto
```
`BLANK_COUNTRY=<label> python x_final.py ...` leaves one country's S1s empty (a leaderboard probe that isolates its
score). v4–v7w are reproduced from branch `ber-pipeline` (ff64094); this branch's normalisation no longer contains
their hand-written dictionaries.

Validate (from this folder):
```bash
cd student_resource && python3 utils/validate_submission.py --check-ids \
  --matching ../output_v7w/matching_results.tsv --candidate ../output_v7w/candidate_pairs.tsv --test-dir dataset/test
```

No external databases, APIs or lookup services are used at any stage. The only hand-written resources are small
normalisation dictionaries (abbreviations, legal forms, state codes, the seven French words above).
