# Business Entity Resolution

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.

**Leaderboard best: v7w, 0.982641** (2026-09-25). A France probe puts US + India at about 0.991 and France at about
0.933 (see [Leaderboard](#leaderboard)).
**Current pipeline: v8** (2026-09-26): the same validation score as v7w (0.99236 vs 0.99240, copy-free) with scalable
blocking (FAISS HNSW search + a calibrated shortlist: 65% fewer candidate pairs than top-5), no hand-written
abbreviation / legal-form / country tables (a lexicon learned per country from the records), and one decision rule for
every country. Its leaderboard score is pending.

## Algorithm (v8)

Key observation from the training ground truth: **every S2/S3 record belongs to at most one S1 entity**
(7.6M matched ids, none reused), ~26% of S2/S3 records match nothing, and only 5.6% of S1 entities are singletons.
So we solve it record-by-record: *for each S2/S3 record, which S1 entity (if any) is it?* Country is an open set of
labels: nothing below names a country or a language.

1. **Normalise** (`common.py`, `lexicon.py`): transliterate every script to ASCII with `anyascii`
   (`व्हाइट बिल्डर्स` → `vhait bildrs`), lowercase, tokenise. Single letters separated by dots / spaces form one token
   (`L.L.C.` → `llc`, `P O Box` → `po box`) and letters glued in front of a number are split off (`N°29` → `ndeg 29`).
   Abbreviations and legal forms come from a **per-country lexicon learned without labels** from pseudo-matches (a
   record and its top retrieved S1 when the cosine is ≥ 0.9), because the same abbreviation means different things in
   different countries (`tn` is Tennessee in the US and Tamil Nadu in India; `ste` is suite / sainte; `de`, `la` are
   French words):
   - an abbreviation is a short token that is a letter subsequence of the other side's word, or the initials of a 2–3
     word span, frequent and unambiguous in that country and at least 2 letters shorter (so typos are not learned);
   - legal forms are words among the last two of that country's S1 names that matching records drop far more often than
     its other name words (median drop rate + 0.1), plus words nearly always followed by one legal word ("private"
     limited) and record-side spellings of them;
   - legal spelling families (`pvt`, `praivet` → `private`; `ltd` → `limited`) join words that are spellings of each
     other and that matching records mostly swap with each other; distinct forms that share letters stay apart.

   | Learned | US | India | France |
   |---|---|---|---|
   | address abbreviations | tx→texas, st→street, ave→avenue, de→delaware | tn→tamil nadu, up→uttar pradesh, mh→maharashtra | r→rue, st→saint, bd→boulevard, imp→impasse, q→quai |
   | legal forms | llc inc corp corporation co company lp pc | limited ltd llp private (+ ~90 spellings) | sa sarl sas sasu sci |

2. **Blocking** (`train_embed.py`, `retrieve.py`, `shortlist.py`, `match.py`): `intfloat/multilingual-e5-small` (MIT,
   118M) fine-tuned with in-batch negatives on (record, its S1) pairs of S1 folds 0–3 embeds every record. Each record
   takes its top-20 S1 records **with the same country label** from a FAISS HNSW index (M 32, efSearch 512), so a query
   costs about log(#S1) instead of #S1. A **calibrated shortlist** (a small LightGBM on retrieval-only features: cosine,
   rank, gaps to the best and next candidate, near-ties, softmax share) keeps candidates with P ≥ 0.002. Those pairs are
   exactly what the matcher scores and what `candidate_pairs.tsv` lists (see [Blocking](#blocking)).
3. **Pair features** (`match.py features`, `x_feats.py`): embedding score, rank and gaps; rapidfuzz similarities on
   full, core and consonant-skeleton names and on addresses; numbers, token rarity, name frequency; integer-aware
   house numbers; legal-form edits over the learned families; name edits between core names (typo-tolerant words
   added / dropped, how alike a swapped pair is, and per-country statistics of those words from the split's own
   records: how common in that country's S1 names, how much more often inserted into records than present in S1 names).
4. **Two cross-encoders**, fine-tuned only on the retrieved top-5 pairs of S1 folds 0–3: `x_ce.py` (e5-small from the
   fine-tuned bi-encoder, normalised `name | address`) and `x_ce3.py` (e5-small on raw transliterated text that keeps
   punctuation and suffix spellings; the best single model). Each adds its score, its rank within the record, the gap
   to the record's next candidate, its rank within the S1 and the S1's positive claims. (The e5-base cross-encoder of
   v6/v7 is dropped: +0.00003.)
5. **Stage 1** (`x_stage_multi.py s1`): LightGBM on pair + cross-encoder features, cross-fitted over S1 folds 4–9
   (4–6 and 7–9). Each train pair gets the probability of the model that did not see it; test pairs get the mean.
6. **Stage 2** (`x_nocopy.py`): each record's best candidate is re-scored with entity context, i.e. the other records
   claiming the same S1 (how many above 0.9 / 0.5 / 0.2, their sum and max, rank, same-source claims). Test has about
   39% distractors against 26% in train, so distractors are **weighted** (each appears once, weight 3.04); repeating
   them creates identical twins that test never has.
7. **Decision** (`x_final.py`): accept a record if its stage-2 score is ≥ 0.70, the same rule for every country.

Folds: `crc32(S1 id) % 10`; 0–3 train the bi-encoder and the cross-encoders, 4–9 train stages 1 and 2. Stage 2 is
validated on entity folds 8–9 after fitting on 4–7. Distractors follow their own hash fold, and those in 0–3 are left
out of stage 1/2 training and validation because the cross-encoders saw them.

## Blocking

Candidates kept per record, and the share of real records whose true S1 survives (train records of S1 folds 7–9, which
neither the bi-encoder nor the shortlist model saw; `shortlist.py`):

| Candidate set | True S1 kept | Per record | Per S1 |
|---|---|---|---|
| top-5 (v4–v7) | 98.95% (exact search: 99.02%) | 5.00 | 23.4 |
| top-20 | 99.42% | 20.0 | 93.5 |
| Sarvesh's gap 0.1 | 99.35% | 1.99 | 9.3 |
| **calibrated, P ≥ 0.002 (v8)** | **99.34%** | **1.46** | **6.8** |
| calibrated, P ≥ 0.005 | 99.28% | 1.34 | 6.3 |

On test, v8 scores 17.25M pairs instead of 49.8M: 1.24 candidates per record in the US, 1.66 in India, 3.25 in France
(French names reuse a small vocabulary, so near-ties between namesakes are common and the model keeps them). Per S1:
mean 7.2 / 9.7 / 18.0, median 7 / 8 / 11.

Search (`x_ann.py`, share of real records whose true S1 is found):

| Index | India @1 / @20 | US @1 / @20 | Cost |
|---|---|---|---|
| exact GPU matmul (v1–v7) | 97.95 / 99.54 | 97.70 / 99.46 | every record x every S1 of its country |
| **HNSW M32, efSearch 512 (v8)** | 97.91 / 99.49 | 97.62 / 99.35 | ~30 s per 100k queries on 32 CPU cores |
| HNSW, efSearch 768 / M48 / centred embeddings | – | 97.63–97.64 / 99.36–99.38 | 1.0–1.6x slower |
| IVF, 256 of 939 lists probed | 96.78 / 98.10 | – | 210 s per 100k |

On all train records HNSW finds the true S1 at @1 / @5 / @20 for 97.78 / 98.95 / 99.42% vs 97.84 / 99.03 / 99.51%
exact. Adding S1s whose normalised core name equals the record's (a hash lookup, `x_namekey.py`) recovers only 0.01
points of that, so it is not used.

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
| **v8** | HNSW + calibrated shortlist, learned lexicon, name-edit features, no e5-base CE, no country rules | **0.99236 copy-free** | pending |

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
- Sarvesh's adaptive shortlist (`SHORTLIST=gap:0.1`, branch `ber-improvements`) gained 0.0011 on v4; v8 includes it
  in `match.py` next to the calibrated shortlist, which keeps the same share of true S1s with 27% fewer candidates.

`x_compare.py` compares submission files without labels (matches per S1, empty S1s, and macro F0.5 of one file scored
against another). Scored against v5f as if it were the truth, Sarvesh's 0.967 file gets 0.968.

## Reproduce

### Setup
```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt
```
Model (MIT licence): `intfloat/multilingual-e5-small`, fetched on first use (`E5=/path` for an offline node); both
cross-encoders start from the fine-tuned copy. (`intfloat/multilingual-e5-base` is only used by the v6/v7 plans.)

Data: place (or symlink) the provided `student_resource/` folder next to `src/`, or set `AMLC_ROOT` to the folder that
contains it. Caches, models and logs go to `$AMLC_ROOT/work`.

### Run v8 (from `src/`; one A100 80GB, ~32 cores, ~200 GB RAM; about 8 h end to end)
```bash
python common.py                     # self-checks for tokenisation, normalisation and the F0.5 scorer
python train_embed.py                # fine-tune the e5-small bi-encoder on S1 folds 0-3             (~10 min)
python x_chain.py v8a                # HNSW retrieval (train, test), learned lexicon, shortlist model (~2.2 h)
SHORTLIST=model:0.002 python x_chain.py v8b
                                     # shortlisted pair features, name-edit / legal features, v4 stage 1 (the CE
                                     # sanity checks read it), both cross-encoders trained and scored, stage 1,
                                     # stage 2 without distractor copies -> work/x/test_q_v8w.parquet    (~4.5 h)
python x_final.py ../output_v8 _v8w 0.70   # matching_results.tsv + candidate_pairs.tsv
```
`x_chain.py` runs each step in order and logs it to `work/logs/<step>.log`; the plans list the exact commands and
environment. The v8 submission reused the raw-text cross-encoder trained in v7 (on the exact-search top-5); the
`v8b` plan retrains it. `BLANK=<country> python x_final.py ...` writes a leaderboard probe with that country's S1s
left empty; `python x_final.py sweep _v8w` prints each country's no-match rate and matches per S1 by threshold.

Validate (from this folder):
```bash
cd student_resource && python3 utils/validate_submission.py --check-ids \
  --matching ../output_v8/matching_results.tsv --candidate ../output_v8/candidate_pairs.tsv --test-dir dataset/test
```

No external databases, APIs or lookup services are used at any stage, and since v8 there are no hand-written word
lists either: abbreviations, legal forms and word statistics are learned from the provided records (test records
without labels, as allowed for unsupervised statistics).

Known limits of v8: HNSW finds the true S1 about 0.09 points less often than exact search at @20; France's EURL fell
just under the learned legal-form cut; the "inserted word" statistic also flags harmless US / India insertions (www,
dba, honorifics), so its effect on France has to be checked on the leaderboard.
