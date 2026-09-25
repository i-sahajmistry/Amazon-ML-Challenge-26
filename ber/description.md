# v8: compliant, country-agnostic rebuild of v7w

Branch `v8-generalize`, built on `ber-pipeline` ff64094 (v7w, leaderboard 0.982641).

## Why

1. **Rules risk.** The problem statement says to treat `country` as an open set of labels and not to hard-code
   the pipeline to specific countries. v7w contained several hand-written, country-specific pieces:
   - `common._STATES`: US and India state codes.
   - French street words and `mg`→"mahatma gandhi" in `common._ADDR_ABBR`.
   - French legal forms in `common._LEGAL` and in `x_feats.LEGAL`.
   - In `x_final.py`, a France-only rule: an `== "France"` branch, a list of 7 French words and a France
     threshold.

   All of these are knowledge about the test countries written by hand. They are the parts a reviewer is most
   likely to question.
2. **Generalisation.** The rule that fixed France only works because someone read the French test data. The
   same situation for any other unseen country needs a procedure, not a word list.
3. **Speed.** 50M candidate pairs, scored by three cross-encoders, dominated the ~9 h run.

v8 replaces every hand-written country-specific piece with a procedure that runs the same way on every country
label and learns from the data. It also adds a learned blocking shortlist.

## What changed

| Area | v7w | v8 |
|---|---|---|
| Normalisation dictionaries | hand-written: US/India state codes, French street words, French legal forms, `mg` | **mined per country** from the data (`mine_dicts.py`); only generic English defaults stay (the noise the problem statement lists: Rd/Road, Pvt/Private) |
| Blocking → matcher | fixed top-5 per record (49.9M test pairs) | **learned shortlist** (`shortlist.py`): 1–20 candidates per record, 16.5M test pairs |
| France distractor words | hand-written list of 7 words, hard reject, France only | **word model** (`words.py`): scores every word in every country from label-free statistics plus its multilingual embedding; used as features, not rules |
| Cross-encoders | e5-small (normalised text), e5-base, e5-small (raw text) | e5-small (normalised) and e5-small (raw); e5-base dropped (v6 gained +0.00003) |
| Stage 2 at test | mean of stage-1 models A and B | scored once with A's and once with B's probabilities, then averaged (stage 2 learned on single-model probabilities) |
| Decision | 0.70, plus the France rule | one threshold for every country (0.70; 0.68 for the judge average); optional XGBoost judge (`x_judge.py`) |
| Python 3.14 | pools that rely on inherited globals fail under forkserver | `common.py` sets `fork` |

### 1. Mined dictionaries (`mine_dicts.py`)

- **Input pairs:** confident top-1 retrieval pairs, cosine ≥ 0.90 and ≥ 0.05 above the runner-up, from train
  and test. No labels are used.
- **Abbreviations:** a token that appears on one side only is an abbreviation of a longer one-sided token when
  it has the same first letter and is a subsequence of it. It must be seen ≥ 50 times and explain ≥ 50% of its
  occurrences.
- **Legal forms:** tokens of at most 4 letters that are often missing on one side, in both directions.

What it found, with no lists written by hand:

| Country | Addresses | Names / legal forms |
|---|---|---|
| US | st→street, rd→road, tx→texas, oh→ohio, il→illinois, blvd→boulevard … (253) | llc, inc, corp, lp |
| India | mh→maharashtra, ka→karnataka, gj→gujarat, prdes→pradesh, rod→road, flor→floor … (85) | ltd, llp, co, corp; prvte/pivte→private, siva→shiva |
| France | r→rue, av→avenue, all→allee, imp→impasse, bd→boulevard, q→quai, crs→cours, jen→jean … (25) | sas, sa, sasu, sci; frs→freres |

### 2. Learned shortlist (`shortlist.py`)

A LightGBM model on retrieval-only features decides which of each record's 20 retrieved candidates go on to the
matcher.
- **Record side:** score, rank, gap to the best and to the next candidate, near-ties.
- **S1 side:** how many records retrieve this S1 or rank it first, and this record's rank and gap among them.

It is trained on records of S1 folds 4–6 and evaluated on folds 7–9:

| Candidates kept | True S1 kept | Candidates per record |
|---|---:|---:|
| top 5 (v7w) | 99.02% | 5.00 |
| gap 0.1 (Sarvesh, `ber-improvements`) | 99.42% | 1.98 |
| **learned, τ = 0.001 (v8)** | **99.47%** | **1.54** |
| all 20 (ceiling) | 99.50% | 20 |

On test it keeps 2.2 candidates per record in France, 1.5 in India and 1.2 in the US. France has more
near-ties, and the model adapts to that without any country rule.

### 3. Distractor-word model (`words.py`)

- **What it scores:** for each word a record adds relative to its top-1 retrieved S1, the probability that the
  record is a distractor.
- **Label-free statistics:** add rate, how often the word appears in S1 names, insertion vs substitution
  ("holdings" vs "praivet" for "private"), and position in the name.
- **Embedding signal:** a k-NN score over the multilingual e5 embedding, so "groupe" inherits what "group"
  taught.
- **Leave-one-country-out:** each train country's words are scored by a model fitted on the other train
  countries. Test on US words, trained on India: AUC 0.80. Test on India words, trained on US: AUC 0.95. So
  stage 1 learns from features exactly as uncertain as an unseen country's.
- **France (fitted on US + India):** it recovers all 7 words of the old hand-written list (groupe, france,
  developpement, participations, distribution, holding, international). It also flags club, comite, ecole,
  amicale, … These words are common in French S1 names, yet added only ~10% of the time, which is the same
  pattern as train's hotel/steel (India) and development/international (US) distractors.
- **Pair features:** max, sum and count of p_bad over added words, max add rate, and max p_bad of dropped
  words.

### 4. Stage-2 scoring per stage-1 model

Stage 2 is trained on out-of-fold stage-1 probabilities from a single model. At test it now sees the same
kind of input: one pass with model A's probabilities and one with model B's, averaged. In v4 this fix was
worth +0.0097 on the leak-free test bed (the averaged-input bug, see the `decision-fixes` notes). Since v5,
stage 2 trains only on single-model probabilities, so here it is a consistency fix with a smaller expected
gain.

### 5. XGBoost as the decision judge (`x_judge.py`)

LightGBM and XGBoost (CPU build, Apache-2.0) are fitted on exactly the same copy-free stage-2 rows, and the
average of the two is also evaluated. The one that validates best is used for `output_v8J`.

Same copy-free validation (entity folds 8–9, distractors weighted to 39%):

| Threshold | LightGBM | XGBoost | Average |
|---:|---:|---:|---:|
| 0.60 | 0.99266 | 0.99266 | 0.99269 |
| 0.66 | 0.99270 | 0.99270 | 0.99275 |
| 0.68 | 0.99269 | 0.99271 | **0.99276** |
| 0.70 | 0.99268 | 0.99270 | 0.99273 |
| 0.80 | 0.99255 | 0.99260 | 0.99257 |
| 0.90 | 0.99210 | 0.99222 | 0.99217 |

XGBoost is as good as LightGBM (+0.00001 at the best threshold) and flatter at high thresholds. The average adds
+0.00006, which is within noise. `output_v8J` uses the average at 0.68. XGBoost needed one fix: it rejects `inf`
(the cross-encoder gap of a record with a single candidate), so `x_judge.finite` maps it to ±1e6.

## Results

### Validation

Entity folds 8–9 with distractors weighted to the 39% test share. The copy-free view is the one to compare on.
The "copies" view repeats distractors, which reads about 0.0006–0.0016 high.

| | Copy-free F0.5 | "Copies" view | Cross-encoders | Test pairs | Run time |
|---|---:|---:|---|---:|---|
| v7w (leaderboard 0.982641) | 0.99240 (thr 0.70) | 0.99304 | 3 (small, base, raw) | 49.8M | ~9 h, 32 CPUs |
| **v8** | **0.99267** (0.70), 0.99271 (0.65) | **0.99332** | 2 (small, raw) | **16.5M** | **3 h 40 min**, 16 CPUs |
| **v8J** (LightGBM + XGBoost judge) | **0.99276** (0.68) | – | 2 | 16.5M | +10 min |

Parts that can be measured on their own:
- **Shortlist and mined dictionaries**, with the v4 feature set and v4 stage 1 on the same folds: best 0.98509
  (top-5, hand-written dictionaries including the state table) → **0.98614** (learned shortlist, mined
  dictionaries), +0.0011. Removing the country lists cost nothing.
- **Cross-encoders on the shortlisted pairs** (fold 9, right entity at the record's argmax): normalised
  0.98237 → 0.98282, raw text 0.98290 → 0.98304. They were trained on 60% fewer pairs.
- **Word features in stage 1:** `w_add_sum` ranks 7th, `w_add_max` 13th and `w_addrate_max` 17th of 71
  features by gain, behind only the cross-encoders and house-number features.

France has no labels, so validation cannot measure the France-specific effects (mined French dictionaries, the
word model replacing the 7-word rule). Only the leaderboard can.

### Test outputs

| File | Accepted records | India | US | France | Empty S1s | Validator |
|---|---:|---:|---:|---:|---:|---|
| `output_v8` (LightGBM, 0.70) | 5,889,878 | 2,752,420 | 2,256,985 | 880,473 | 99,789 (5.8%) | PASS `--check-ids` |
| `output_v8J` (average, 0.68) | 5,898,480 | 2,753,878 | 2,259,259 | 885,343 | 99,598 (5.7%) | PASS `--check-ids` |

Matches per S1 are 3.39–3.40 in every country (train truth: 3.46). `candidate_pairs.tsv` shrinks from 665 MB to
235 MB.

### What to expect on the leaderboard

- **US + India:** +0.0003 over v7w on validation, and v7w's US + India leaderboard score (~0.991) tracked its
  validation (0.992).
- **France:** unknown. v7w's hand-written rule rejected records adding one of 7 French words. v8 learns the same
  words, plus about 20 more, as features. That is safer for compliance and could be better or worse for France.
- **To isolate France**, submit `output_v8` and then the same file with France left empty
  (`BLANK_COUNTRY=France python x_final.py ...`), exactly as the v7w probe did.

## How to run

```bash
# once: bi-encoder + retrieval (unchanged from v7w; reuse work/e5_ft and work/cand_*.parquet if present)
python train_embed.py && python retrieve.py train && python retrieve.py test
python x_chain.py v8                          # mine_dicts ... x_final  -> ../output_v8     (~4 h, 1 A100, 16 CPUs)
TAG=_v8 CE_TAGS=,_raw WORDS=1 python x_judge.py && JUDGE=best TAG=_v8 CE_TAGS=,_raw WORDS=1 python x_judge.py test
python x_final.py ../output_v8J _v8J auto     # submission from the best judge
```

## Files

- **New:** `mine_dicts.py`, `shortlist.py`, `words.py`, `x_judge.py`, and `analysis/v8/g{1,2,3,5}_*.py` (the
  analyses above; run them from that folder with `PYTHONPATH=../../src`).
- **Changed:**
  - `common.py`: mined dictionaries, no country lists, fork.
  - `match.py`: country-aware normalisation, `SHORTLIST=learned:τ`.
  - `x_feats.py`: mined legal bit.
  - `x_stage_multi.py`: word features, pA/pB saved.
  - `x_nocopy.py`: per-model test scoring.
  - `x_final.py`: one rule for all countries.
  - `x_chain.py`: `v8` plan.
