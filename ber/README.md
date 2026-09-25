# Business Entity Resolution

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.

**Current best: v7w, leaderboard 0.982641** (2026-09-25). A France probe puts US + India at about 0.991 and France at
about 0.933, so France is where the remaining gap is (see [Leaderboard](#leaderboard)).

## Algorithm (v7w)

Key observation from the training ground truth: **every S2/S3 record belongs to at most one S1 entity**
(7.6M matched ids, none reused), ~26% of S2/S3 records match nothing, and only 5.6% of S1 entities are singletons.
So we solve it record-by-record: *for each S2/S3 record, which S1 entity (if any) is it?*

1. **Normalise** (`common.py`): transliterate every script to ASCII with `anyascii`
   (`व्हाइट बिल्डर्स` → `vhait bildrs`), lowercase, expand abbreviations (`pvt→private`, `st→street`, 2-letter
   US/India state codes → names), strip website suffixes, and derive a "core" name without legal words.
2. **Blocking** (`train_embed.py`, `retrieve.py`): `intfloat/multilingual-e5-small` (MIT, 118M) fine-tuned with
   in-batch negatives on (record, its S1) pairs of S1 folds 0–3. Each record takes its top-20 S1 records **with the same
   country label** by exact GPU cosine search; the top 5 go on. Recall of the true S1: 97.8% @1, 99.0% @5, 99.5% @20.
3. **Pair features** (`match.py features`, `x_feats.py`): embedding score, rank and gaps; rapidfuzz similarities on
   full, core and consonant-skeleton names and on addresses; numbers, token rarity, name frequency; integer-aware
   house-number features and legal-form bits.
4. **Three cross-encoders**, fine-tuned only on pairs of S1 folds 0–3:
   `x_ce.py` (e5-small from the fine-tuned bi-encoder, normalised `name | address`), `x_ce2.py`
   (`intfloat/multilingual-e5-base`, MIT, 278M, normalised text, each record's top 3) and `x_ce3.py` (e5-small on raw
   transliterated text that keeps punctuation and suffix spellings; the best single model). Each adds its score, its
   rank within the record, the gap to the record's next candidate, its rank within the S1 and the S1's positive claims.
5. **Stage 1** (`x_stage_multi.py s1`): LightGBM on pair + cross-encoder features, cross-fitted: model A on records of
   S1 folds 4–6, B on 7–9. Each train pair gets the probability of the model that did not see it; test pairs get the
   mean of A and B.
6. **Stage 2** (`x_nocopy.py`): each record's best candidate is re-scored with entity context, i.e. the other records
   claiming the same S1 (how many above 0.9 / 0.5 / 0.2, their sum and max, rank, same-source claims). Test has about
   39% distractors against 26% in train, so distractors are **weighted** (each appears once, weight 3.04); repeating
   them creates identical twins that test never has.
7. **Decision** (`x_final.py`): accept a record if its stage-2 score is ≥ 0.70. France records whose name *adds* one of
   `groupe france developpement participations distribution international holding` to the S1 name are rejected: the
   French counterparts of train's distractor words (group, holdings, <country>, development, international,
   distribution), which never appear in true pairs. `candidate_pairs.tsv` lists the top-5 candidates that were scored.

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

Tried without gain: word-edit log-likelihood features (`x_llr.py`, 0.99274 vs 0.99273), re-ranking each record's top
2 (`x_top2.py`, 0.99278 either way), an expected-F0.5 decision per S1 instead of a threshold (v3: 0.9775 vs 0.9788).

Cross-encoders on their own (fold 9, share of real records whose argmax is the right S1): normalised e5-small 0.98237,
e5-base 0.98211, raw-text e5-small 0.98290; pair AUC 0.99984–0.99986. They add value through stacking.

Where the remaining validation errors are (`x_errors.py` and the other `x_*.py` diagnostics, v5): 73% of missed true matches and 89.5% of
blocking misses are records with an empty address whose name is shared by several S1 entities.

## Leaderboard

| File | Score |
|---|---|
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
- Sarvesh's adaptive shortlist (`SHORTLIST=gap:0.1`, branch `ber-improvements`) gained 0.0011 on v4 and is not yet
  in v7w.

`x_compare.py` compares submission files without labels (matches per S1, empty S1s, and macro F0.5 of one file scored
against another). Scored against v5f as if it were the truth, Sarvesh's 0.967 file gets 0.968.

## Reproduce

### Setup
```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt
```
Models (MIT licence): `intfloat/multilingual-e5-small`, fetched on first use (`E5=/path` for an offline node), and
`intfloat/multilingual-e5-base` for `x_ce2.py`: download it and set `CE_BASE` in the `base` plan of `x_chain.py`.

Data: place (or symlink) the provided `student_resource/` folder next to `src/`, or set `AMLC_ROOT` to the folder that
contains it. Caches, models and logs go to `$AMLC_ROOT/work`.

### Run v7w (from `src/`; one A100 80GB, ~32 cores, ~200 GB RAM; about 9 h end to end)
```bash
python common.py                     # self-checks for normalisation and the F0.5 scorer
python prep_norm.py                  # normalise all records (cached)
python train_embed.py                # fine-tune the e5-small bi-encoder on S1 folds 0-3        (~10 min)
python retrieve.py train             # embed + same-country top-20 search                        (~25 min)
python retrieve.py test              #                                                          (~25 min)
python stage1_cv.py                  # pair features (work/feats2_*) + v4 stage 1               (~40 min)
python x_chain.py v5                 # x_feats, small cross-encoder, v5 stages                  (~2 h)
python x_chain.py base               # e5-base cross-encoder                                    (~2 h)
python x_chain.py rawv7              # raw-text cross-encoder, then stage 1 with all three      (~2.5 h)
TAG=_all CE_TAGS=,_base,_raw python x_nocopy.py test   # stage 2 without distractor copies  (~15 min)
python x_final.py ../output_v7w _allw 0.70 0.70        # matching_results.tsv + candidate_pairs.tsv
```
`x_nocopy.py` without `test` prints the copies vs copy-free comparison. `FR_BLANK=1 python x_final.py ...` writes the
France probe. Each `x_chain.py` step logs to `work/logs/<step>.log`.

Validate (from this folder):
```bash
cd student_resource && python3 utils/validate_submission.py --check-ids \
  --matching ../output_v7w/matching_results.tsv --candidate ../output_v7w/candidate_pairs.tsv --test-dir dataset/test
```

No external databases, APIs or lookup services are used at any stage. The only hand-written resources are small
normalisation dictionaries (abbreviations, legal forms, state codes, the seven French words above).
