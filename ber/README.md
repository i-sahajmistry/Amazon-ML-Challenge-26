# Business Entity Resolution

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.

**Leaderboard best: v10_fr3_llm_dd, 0.990282** (2026-09-27; v10_fr3_llm 0.987071, v10_fr3 0.986077, v10_fr2 0.985942, v9p_frand 0.984742,
v7w 0.982641). The public and private leaderboards are both subsets of the provided test file, so every country
scored is one we see: US, India and France.
**Current pipeline: v10_fr3_llm** = v9p_frand (below) with four changes, see [After v9p_frand](#after-v9p_frand-v10_fr2):
distractors are assigned to the fold of the S1 they imitate, so every training entity keeps all its distractors
(validation 0.99285 vs 0.99277 on the same records); self-training rounds 2 and 3 add two more France vetoes; and a
LoRA-tuned LLM judge (Qwen3-Reranker-4B) is blended into the unsure records (+0.000994 on the leaderboard, five times
its validation gain, so mostly France).

**v9p_frand**: the v8 design (scalable HNSW blocking + calibrated shortlist, a lexicon learned per
country, no hand-written tables) with a shortlist that also sees text similarities (7.4 candidate pairs per S1 on test,
v8 10.0, top-5 28.7), stage 2 that also sees the records claiming the same S1 at the same house number (validation
0.99277, v8 0.99236), and, for the country without training labels (France), cross-encoders self-trained on their own
confident test decisions, used as a veto (+0.00205 on the leaderboard, i.e. +0.014 on France alone).

## Algorithm (v9p_frand)

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
   costs about log(#S1) instead of #S1. A **calibrated shortlist** (`x_shortlist2.py`, a small LightGBM on retrieval
   features — cosine, rank, gaps to the best and next candidate, near-ties, softmax share — and three rapidfuzz
   similarities per pair — name token-set, core-name ratio, address token-set — with each one's gap to the record's
   best) keeps candidates with P ≥ 0.001 (`SHORTLIST=text:0.001`). One similarity per retrieved pair, so it scales like
   the retrieval. Those pairs are exactly what the matcher scores and what `candidate_pairs.tsv` lists (see
   [Blocking](#blocking)).
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
   **Self-training for countries without training labels** (`x_ce3.py train` with `SELF=`; France here): both
   cross-encoders are continued on test pairs of that country whose v8 decision was confident: q ≥ 0.98 makes the
   record's best S1 a match and its other top-5 candidates non-matches (a record has at most one S1), q ≤ 0.02 makes
   all its top-5 non-matches. 4M pairs drawn from the lists of 806k confidently matched and 429k confidently unmatched
   records are mixed with 3M train pairs so US / India are kept, one pass at lr 2e-5. No labels are used; the organisers' Q&A allows self-training on
   the test records.
5. **Stage 1** (`x_stage_multi.py s1`, `S1K=3`): LightGBM on pair + cross-encoder features, cross-fitted over S1
   folds 4–9 in three groups (4–5, 6–7, 8–9). Each train pair gets the probability of the model that did not see it;
   test pairs get the mean of the three.
6. **Stage 2** (`x_nocopy.py`): each record's best candidate is re-scored with entity context, i.e. the other records
   claiming the same S1 (how many above 0.9 / 0.5 / 0.2, their sum and max, rank, same-source claims). Test has about
   39% distractors against 26% in train, so distractors are **weighted** (each appears once, weight 3.04); repeating
   them creates identical twins that test never has. **House-number peers** (`PEERS=1`): a distractor entity's records
   share its altered house number (an S1 at 14 Impasse des Gardénias has its decoys at 15, whatever else they change),
   so stage 2 also sees the other records claiming the same S1 with the same first address number: how many, their
   best and summed stage-1 probability, and how many distinct numbers claim the S1. Five seeds are averaged (`BAG=5`).
7. **Decision** (`x_final.py`): accept a record if its stage-2 score is ≥ 0.70, the same threshold for every country.
   For a country without training labels, the self-trained stack acts as a veto: a record is accepted only if both
   stacks accept it for the same S1 (`FROM=France:_v9spw:min`). Reading 30 random records of each kind of
   disagreement, about 80% of the records the self-trained stack newly rejects are decoys ("& Associés", "Et Fils",
   "Développement", a swapped word, a changed legal form plus a new number), while about 45% of those it newly accepts
   are decoys too: it partly learned "same address, so same entity" from its own confident decisions.

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
| calibrated, P ≥ 0.002 (v8) | 99.34% | 1.46 | 6.8 |
| calibrated, P ≥ 0.005 | 99.28% | 1.34 | 6.3 |
| calibrated + text similarities, P ≥ 0.002 | 99.38% | 1.22 | 5.7 |
| **calibrated + text similarities, P ≥ 0.001 (v9)** | **99.39%** | **1.25** | **5.9** |

On test, v9 scores 12.76M pairs (v8 17.25M, top-5 49.8M). Candidate pairs per S1, mean / median: US 6.6 / 6, India
7.1 / 7, France 10.1 / 8 (v8: 7.2 / 7, 9.7 / 8, 18.0 / 11). French names reuse a small vocabulary, so namesakes are
close in embedding space; the text similarities tell them apart and cut France's list by 44%.

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
| v8 | HNSW + calibrated shortlist, learned lexicon, name-edit features, no e5-base CE, no country rules | 0.99236 copy-free | not uploaded |
| v9 | shortlist with text similarities, stage 1 over 3 fold groups, 5-seed stage 2 | 0.99252 copy-free | – |
| v9p | + house-number peers in stage 2 | 0.99277 copy-free | 0.982690 |
| v9s | cross-encoders self-trained on France (costs US / India a little) | 0.99245 copy-free | – |
| v9p_frand | v9p; a France record needs the self-trained stack (v9s + peers) to agree | 0.99277 (US / India = v9p) | 0.984742 |
| v9s2p | self-training round 2, seeded by v9p_frand's France decisions | 0.99274 copy-free | – |
| v10 | v9p with distractors in the fold of the S1 they imitate, both CEs retrained | 0.99285 on v9p's rows (0.99262 on its own, harder set) | – |
| v10_fr2 | v10; a France record needs both self-trained stacks (v9sp, v9s2p) to agree | 0.99285 | 0.985942 |
| v9p + LLM judge | Qwen3-Reranker-4B (LoRA) blended into the unsure records | 0.99297 (cross-fitted) | – |
| v10s3p | self-training round 3 on v10, seeded by v10_fr2's France decisions | 0.99255 | – |
| v10_fr3 | v10_fr2 + the round-3 France veto | 0.99285 | 0.986077 |
| v10_fr3_llm | v10_fr3 with the LLM judge blended into v10's unsure records | 0.99282 (v10's own set, 0.99262 without) | 0.987071 |
| **v10_fr3_llm_dd** | v10_fr3_llm + France same-address fixes (`x_wordlists.py`): restore 32,439 rejected records whose added words spread like true matches, reject 11,655 accepted type-word swaps | US / India unchanged | **0.990282** |

Validation scores are at threshold 0.70 from v9 on.

Tried without gain: word-edit log-likelihood features (`x_llr.py`, 0.99274 vs 0.99273), re-ranking each record's top
2 (`x_top2.py`, 0.99278 either way), an expected-F0.5 decision per S1 instead of a threshold (v3: 0.9775 vs 0.9788),
six stage-1 fold groups instead of three (0.99234 vs 0.99241), separate thresholds per score bucket (`x_verr.py`,
+0.00001 to +0.00005), taking every France decision from the self-trained stack instead of using it as a veto
(`v9p_frs`, see step 7), a "shifted house number shared with another claimant" veto (`x_final.py NUMVETO`, see below:
validation 0.99277 → 0.99074 for the US), a learned decoy-word veto (`x_decoyveto.py`: precise, but the models already
reject nearly all such records), and, in Mohanish's branch `v9fS`, a LightGBM + XGBoost judge (0.99259 vs 0.99258) and
`bge-reranker-v2-m3` as a third cross-encoder (fold-9 argmax 0.98173 vs 0.98169 / 0.98227).

Cross-encoders on their own (fold 9, share of real records whose argmax is the right S1): normalised e5-small 0.98237,
e5-base 0.98211, raw-text e5-small 0.98290; pair AUC 0.99984–0.99986. They add value through stacking.

Where the remaining validation errors are (`x_errors.py` and the other `x_*.py` diagnostics, v5): 73% of missed true matches and 89.5% of
blocking misses are records with an empty address whose name is shared by several S1 entities.

## After v9p_frand (v10_fr2)

The S1 folds are balanced (`x_folds.py`: per fold ~220k S1s, 60% US, 5.6% singletons, 3.46 matches per S1, ~268k
distractors, 4.4% empty addresses), but distractors took a hash of their own id as fold. Stages 1 and 2 drop those
in folds 0–3 (the cross-encoders saw them), so a training / validation entity kept 0.73 of its 1.22 distractors per S1
(test: about 2.1).

- **v10** (`DFOLD=1` in `harness.truth_arrays`): a distractor takes the fold of the S1 it imitates (its top-1 retrieved
  S1). Whole entity groups stay in one fold: the cross-encoders train on folds 0–3 entities with all their distractors
  and stages 1–2 use every distractor of folds 4–9. Both cross-encoders retrained, then as v9p. 0.99285 vs 0.99277 on
  v9p's validation rows; 0.99262 on its own validation, which now holds every distractor of its entities.
- **Self-training round 2** (`x_chain.py v9s2`): pseudo-labels from v9p_frand's France decisions (a match only if both
  stacks are confident), cross-encoders restarted from the originals. 26 of 30 sampled extra rejections are decoys
  ("& Fils", "& Associés", "Développement", a swapped word). v10_fr2 = v10 with both vetoes
  (`FROM=France:_v9spw:min,France:_v9s2pw:min`).
- **Number rule, rejected**: reject a record whose first house number is within 13 of its S1's, shares no number with
  it and equals another claimant's. Test samples looked like decoys, but on validation it rejects accepted rows that
  are 99.6% true matches (true matches carry ±1–10 number noise, and their same-source duplicates share it): US 0.99277
  → 0.99074, India → 0.99134. Kept only as a probe (`x_final.py NUMVETO=`).
- **Blocking cost** (`x_annloss.py`): 972 validation records (0.063%) whose true S1 exact search ranks first are missing
  from the HNSW top 20, spread over 945 S1s (so not unreachable graph nodes). With the shortlist, blocking costs about
  0.00025 validation F0.5 against exact search.
- **LLM judge** (`x_llm.py`, `x_llmstack.py`): `Qwen/Qwen3-Reranker-4B` (Apache-2.0, 4.0B parameters; Qwen3-8B has
  8.19B, over the 8B limit), LoRA r 16 on 120k train records of folds 4–7 whose stage-1 p is unsure, raw text,
  logit(yes) − logit(no). On validation's 58.6k unsure rows it is weaker alone (AUC 0.844 vs 0.936 for stage 2) but
  complementary: a logistic blend with stage 2 gives 0.99297 vs 0.99277 (cross-fitted over the S1s).
- **Self-training round 3** (`x_chain.py v10s3`): seeded by v10_fr2, cross-encoders continued from v10's. About 23 of
  30 sampled extra rejections (6,095 records) are decoys; leaderboard +0.000135.
- **bge-reranker-v2-m3 as a third cross-encoder** (`x_chain.py v10b`, in progress): Sarvesh's branch `bge-llm` measured
  +0.00016 on v9p's validation (0.99277 → 0.99293) with the model trained on the old folds; v10b trains it on v10's
  folds (`CE_TEXT=orig`: text as given, since the model reads case, accents and scripts itself).

### v10_fr3_llm_dd: France same-address fixes (leaderboard 0.990282, +0.003211)
France's contested records sit at the S1's own house number. Each S1 gets a roughly fixed number of decoys, so a decoy
placed at the S1's address leaves the S1's shifted decoy cluster short. For every word added at the same address,
`x_wordlists.py` compares the spread of those records over nD (the S1's records at other numbers) with unedited records
(total variation distance): TV ≤ 0.10 is true-like, TV ≥ 0.20 decoy-like. On validation, true-like words are 98.7% (US)
and 90.8% (India) true, and decoy-like words 22.5% (India). No labels are used on France.
```
python x_chain.py llm2 llm2_seed      # x/test_q_v10fr3l: v10_fr3_llm's combined France q (then stop the chain)
python wstat.py test; python rule_fr.py _v10plw; python x_rule1.py   # rule1 candidates -> x/rule1_added
python x_wordsame.py; python x_families.py; python x_frfix.py; python x_wordlists.py   # -> x/restore_fr_dd, x/reject_fr_dd
FROM=France:_v9spw:min,France:_v9s2pw:min,France:_v10s3pw:min RESTORE=restore_fr_dd REJECT=reject_fr_dd \
  python x_final.py ../output_v10_fr3_llm_dd _v10plw 0.70
```
The restore candidates are rule1's additions (`x_rule1.py` -> `x_census.R1`): records that `rule_fr.py` accepts (same
house number incl. bis/ter, same compound sub-number, the S1's distinctive street words, same first + rarest core-name
word, no word with a `wstat.py` decoy statistic below 0.1) and that v10_fr3_llm rejects.

## Leaderboard

| File | Score |
|---|---|
| **v10_fr3_llm_dd** | **0.990282** |
| v10_fr3_llm | 0.987071 |
| v10_fr3 | 0.986077 |
| v10_fr2 | 0.985942 |
| v9p_frand | 0.984742 |
| v9p | 0.982690 |
| v7w | 0.982641 |
| v7w with every France S1 left empty (probe) | 0.851155 |
| v4, Sarvesh's run | 0.967214 |
| Rank 1 (2026-09-25 ~22:45) | 0.988319 |
| 5th place (2026-09-26 morning) | > 0.988 |
| Rank 1 / 5th place (2026-09-26 ~15:00) | 0.990556 / 0.98857 |

An empty prediction scores 1 on an S1 with no true matches and 0 otherwise. France is s = 259,452 / 1,732,544 =
14.975% of test S1s, and about e = 5.5% of them should have no match (train prior and our predictions). So:
- US + India ≈ (probe − s·e) / (1 − s) = **0.991** (validation says 0.992),
- France ≈ (best − probe) / s + e = **0.933** (±0.005 from e).

France costs about 0.009 of the overall score. At the US/India level the total would be about 0.991.

v9p and v9p_frand differ only in France (23.6k of its 1.42M records), so their difference is France's alone:
0.002052 / s = **+0.0137 on France** from the self-trained veto. v9p against v7w is +0.00005; if US / India gained what
validation says (+0.0003 overall), France lost about 0.0017 in v8 / v9 (the France word rule of v7w was replaced by
learned statistics). That puts France at about 0.931 in v9p and **0.945 in v9p_frand**, still the whole gap to the top.
v10_fr2 adds +0.0012: about +0.0001 from v10 on US / India (validation) and +0.0011 from France (round-2 veto and the
v10 model), so France is about **0.952**. A top-5 score (0.98857) needs France near 0.97.
The round-3 veto adds +0.000135 (v10_fr3), and the LLM judge +0.000994 (v10_fr3_llm vs v10_fr3). Its validation gain
(+0.0002 on US / India) explains about 0.00017, so about +0.0008 comes from France (+0.0055 on France alone): a
multilingual model reads French decoys that models trained on US / India text cannot.

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

### Run v9p_frand (from `src/`; one A100 80GB, ~32 cores, ~200 GB RAM; about 12 h end to end)
```bash
python common.py                     # self-checks for tokenisation, normalisation and the F0.5 scorer
python train_embed.py                # fine-tune the e5-small bi-encoder on S1 folds 0-3             (~10 min)
python x_chain.py v8a                # HNSW retrieval (train, test), learned lexicon, shortlist model (~2.2 h)
SHORTLIST=model:0.002 python x_chain.py v8b
                                     # v8: shortlisted pair features, name-edit / legal features, both
                                     # cross-encoders trained and scored, stages -> work/x/test_q_v8w.parquet,
                                     # whose confident France decisions seed the self-training     (~4.5 h)
python x_shortlist2.py               # shortlist model with text similarities -> work/shortlist_text.txt
# features() reuses an existing work/feats2_*.parquet: move v8's pair caches aside first
mkdir -p ../work/archive_v8/x && mv ../work/feats2_*.parquet ../work/oof_train.parquet ../work/p1_test.parquet \
  ../work/archive_v8/ && mv ../work/x/ce_{train,test}.npy ../work/x/ce_raw_{train,test}.npy \
  ../work/x/extra_{train,test}.parquet ../work/archive_v8/x/
python x_chain.py v9b                # text shortlist (P >= 0.001), features, CE scoring, stage 1 over 3 groups,
                                     # 5-seed stage 2 -> work/x/test_q_v9w.parquet                    (~1 h)
python x_chain.py v9p                # + house-number peers -> work/x/test_q_v9pw.parquet
python x_chain.py v9s                # both CEs continued on confident France decisions of v8, re-scored,
                                     # stages with and without peers -> work/x/test_q_v9spw.parquet   (~1.5 h)
FROM=France:_v9spw:min python x_final.py ../output_v9p_frand _v9pw 0.70   # matching_results + candidate_pairs
```
Then v10_fr2 and the LLM judge (GPU 2 of the node with `GPU_IDX=1` in our `run.sh`):
```bash
python x_chain.py v9s2               # self-training round 2 -> work/x/test_q_v9s2pw.parquet
python x_chain.py v10                # DFOLD=1: CEs retrained, re-scored, stages -> work/x/test_q_v10pw.parquet
FROM=France:_v9spw:min,France:_v9s2pw:min python x_final.py ../output_v10_fr2 _v10pw 0.70
python x_llm.py train && python x_llm.py val && python x_llm.py test   # Qwen3-Reranker-4B LoRA (LLM=/path)
python x_llmstack.py _v9p _v10p      # blend into v10's unsure records -> work/x/test_q_v10plw.parquet
FROM=France:_v9spw:min,France:_v9s2pw:min python x_final.py ../output_v10_fr2_llm _v10plw 0.70
python x_chain.py v10s3              # self-training round 3, seeded by v10_fr2 -> work/x/test_q_v10s3pw.parquet
FROM=France:_v9spw:min,France:_v9s2pw:min,France:_v10s3pw:min python x_final.py ../output_v10_fr3_llm _v10plw 0.70
```
`x_chain.py` runs each step in order and logs it to `work/logs/<step>.log`; the plans list the exact commands and
environment (`python x_chain.py <plan> <first_step>` resumes). `python x_final.py ../output_v9p _v9pw 0.70` writes
v9p. `BLANK=<country>` writes a leaderboard probe with that country's S1s left empty, `THR_C=France:0.8` sets one
country's threshold, and `python x_final.py sweep _v9pw` prints each country's no-match rate and matches per S1 by
threshold. `x_frdiag.py <tag>` prints each country's score histogram and sample S1 groups from the unsure band.

Validate (from this folder):
```bash
cd student_resource && python3 utils/validate_submission.py --check-ids \
  --matching ../output_v9p_frand/matching_results.tsv --candidate ../output_v9p_frand/candidate_pairs.tsv --test-dir dataset/test
```

No external databases, APIs or lookup services are used at any stage, and since v8 there are no hand-written word
lists either: abbreviations, legal forms and word statistics are learned from the provided records (test records
without labels, as allowed for unsupervised statistics), and the self-training uses only the pipeline's own confident
decisions on the test records.

Known limits: HNSW finds the true S1 about 0.09 points less often than exact search at @20; France's EURL fell just
under the learned legal-form cut; the "inserted word" statistic flags decoy words (participations, holding, associés)
and harmless ones alike (www, dba, Indian honorifics), so the models lean on it little; the self-trained
cross-encoders take the address as proof of identity, so their new acceptances are not used.
