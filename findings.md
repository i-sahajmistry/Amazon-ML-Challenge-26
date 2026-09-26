# Findings log (Sahaj's pipeline, branch v11 / ber-pipeline)

Newest first. Each entry: what was measured, the number, what it means. LB = public leaderboard; "val" = US/India
validation (entity folds 8-9, distractors weighted to the test share of 39%, copy-free).

## 2026-09-27 01:30 — the LLM judge as an independent check of suf7 / trej; hedge file; what v12b (GitHub) offers
`x_frllm.py` reads the LLM margins computed at ~23:00 for every unsure row (0.01 < q < 0.99), before the hand review
existed, so they are an independent label-free signal. Calibration on US / India validation band rows: P(no | true)
0.072, P(no | not true) 0.599.

| set | rows with a margin | LLM "no" rate | median margin | stage q median |
|---|---|---|---|---|
| **reject_fr_type** (accepted, same-address type-word swap) | 3,621 of 11,277 | **0.600** (= val decoys) | −1.00 | 1.00 |
| restore_fr_suf7 | 28,141 of 32,316 | 0.389 | +0.81 | 0.22 |
| restore_fr_r1 (my R1) | 4,621 of 5,065 | 0.087 | +2.12 | 0.84 |
| rule1 (LB: ~57.5% true) | 43,908 of 54,774 | 0.526 | −0.31 | 0.29 |
| rule1, family TYPE | 15,109 | **0.769** | −2.31 | |
| rule1, family NONE | 1,155 | 0.075 | +2.88 | |
| rule1, SUF7 adding associes / france | 4,297 / 3,153 | **0.007 / 0.006** | +3.50 / +5.62 | |
| rule1, SUF7 adding fils / et / cie / services | 9,383 / 5,129 / 40 / 50 | 0.18 / 0.23 / 0.13 / 0.24 | ≥ +0.8 | |
| rule1, SUF7 adding groupe / developpement | 5,069 / 4,999 | **0.826 / 0.982** | −1.69 / −3.62 | |

- **The judge independently confirms the type-word rejects:** the same "no" rate as known decoys. It also confirms
  the suffix restores for fils / associés / et / france / cie / services and the no-word records.
- **Disputed: groupe / développement (11,453 of suf7's 32,316).** The judge says decoy. But it learned
  "group / holdings added = decoy" from US distractors, and most France groupe records *are* shifted decoys (stay rate
  0.19), so its "no" may be a word prior rather than evidence at the same address. The census (spread like true
  matches) and the LB arithmetic say true. If groupe / dev at SAME were decoys and TYPE ~0% true (as both the judge
  and the census say), rule1 would be ~38% true, not the observed 57.5%. **The LB number favours the hand review.**
- The France LoRA (`_fr`) is useless as a check: it rejects 93% of suf7 and accepts 66% of trej. It learned the
  vetoes' decisions (confirmation bias, as the council predicted).
- **Hedge file `submissions/v10_fr3_llm_suf7ngtrej`** (md5 7a4b5947, PASS; `x_suf7ng.py` → `x/restore_fr_suf7ng`):
  suf7trej minus the 11,453 groupe / développement restores. France 848,451; US / India unchanged.
  - Hand review right (groupe/dev ~97% true): suf7trej ≈ +0.0031, suf7ngtrej ≈ +0.0025.
  - Judge right (groupe/dev decoys): suf7 would be ~62% true, below the ~70% break-even. suf7trej ≈ +0.001,
    suf7ngtrej ≈ +0.0025.
  - Upload suf7trej first (the LB-backed version). If it comes in well below ~+0.003, the gap tells groupe/dev's true
    share; then use suf7ngtrej.
- **Mohanish's v12b (origin/v12b, ac8c6f4)**: v10_fr3_llm + v9fX as a fourth France veto (2,731 records) + `x_llmhi.py`
  (the judge on France rows accepted at stage q ≥ 0.99, which the blend never scores; `LLMVETO=0` rejects 7,205 where it
  says no: 0.92% of France vs 0.12% of US / India true). France 838,865 → 829,412; md5 e3d9e4a7; LB pending (his
  estimate 0.9873-0.9879).
  - Useful: the same judge that confirms trej. The rows it vetoes at q ≥ 0.99 are the high-confidence part of the same
    same-address decoy population.
  - Risk: it will also veto accepted groupe / développement suffix records at SAME, which the hand review says are
    true. Before combining v12b with suf7trej, drop the LLMVETO rows that are in the suf7 family.
  - Its scores (`x/llmhi_test_*.parquet`) are not on padum. Rerunning x_llmhi here takes ~785k rows at ~150 pairs/s
    ≈ 90 min per GPU.

## 2026-09-27 02:05 — hand-written vs data-driven word lists for the France fixes (`x_frfix.py` vs `x_wordlists.py`)
**Hand-written lists** (`x_frfix.py`; picked by hand from the SAME / D ratio table in `x_wordsame.py`):
- suffix-7 (restore when these are the only added words, 'et' allowed with them): **fils, associes, cie, services, groupe,
  developpement, france**.
- type words (reject accepted SAME records adding one): every word with SAME / D ratio 0.3-1.2 and ≥ 150 SAME records,
  **except** compagnie, service and the legal forms (eurl, sarl, sas, sa, sasu, sci, snc, ei).

**Data-driven rule** (`x_wordlists.py [TV_TRUE=0.1] [TV_DEC=0.2] [MIN=300]`):
- For each word added at SAME, compute TV = the total variation distance between the nD census of those records
  (0..4+) and that of unedited SAME records.
- true-like: TV ≤ 0.1 and SAME / D ≥ 0.1 (not a pure shifted-decoy word). decoy-like: TV ≥ 0.2. Both need ≥ 300
  records.

Validation of the rule on labels (true rate of SAME records adding the word):

| | true-like words | decoy-like words |
|---|---|---|
| US val | 38 words, 126,574 rows, **98.7% true** | none (US same-address decoys are too rare) |
| India val | 131 words, 186,633 rows, **90.8%** (India's base rate at SAME) | 15 words, 22,008 rows, **22.5%** (TV > 0.3: 0%) |

France lists from the rule:
- **true-like (44):** all 7 hand words plus known true-noise templates. DBA / alias words: dba, aka, formerly, nee,
  fka, known, doing, business, trading, labs, sys, one, co. Initials / letters: a, b, d, f, k, t, ac, cb, cc, cs, ec,
  lc, mc, pc, sc. Also com, eurl, 5arl, de, du, la, saint, et.
- **decoy-like (41):** the type words. It matches the hand list except `service` (TV 0.235: data rejects it, hand
  skipped it). Five rare words (culturel, medico, sportif, atelier, auto) are under MIN, so the hand list has them and
  the data list doesn't. compagnie (TV 0.116) and eurl (0.031) are left out by both.
- Pure shifted-decoy words at SAME (participations, distribution, holding, international, snc) have TV 0.15-0.19 and
  ratio < 0.05, so the rule leaves them out of both lists; the best file already rejects them.

Comparison (both on v10_fr3_llm with the three vetoes; France only):

| file | restore | reject | France accepted | md5 | expected LB at 70% / 95% right |
|---|---|---|---|---|---|
| `v10_fr3_llm_suf7trej` (hand) | 32,316 | 11,277 | 859,904 | 75b39c01 | +0.00089 / +0.00311 |
| **`v10_fr3_llm_dd`** (data-driven) | 32,439 (contains every hand one; +123, mostly eurl combos) | 11,655 (11,221 shared; +418 'service') | 859,649 | 75f68b10 | +0.00092 / +0.00317 |

The two files differ in 613 S1 rows (dd vs hand: +179 / −434 pairs), within ±0.0001 of each other on the LB. The
data-driven one needs no hand-written words and is validated on US / India labels, so it is the better submission;
the hand one stays as the reference.

## 2026-09-27 01:40 — France hand review → two same-address patterns → three probe files
Hand review of 180 France groups (random / unsure / empty; `review/france_review.md` has notes for every group,
`review/frgroups_France_*.txt` the dumps, `review/anat_*_val.txt` the labelled US / India view). Scripts: `x_frgroups.py`,
`x_anatomy.py`, `x_wordsame.py`, `x_census.py`, `x_families.py`, `x_frfix.py`.
- **Where France's decisions sit** (test rows per S1 by position relative to the S1's first house number):

  | | SAME | D (decoy cluster) | OTHER | NONE |
  |---|---|---|---|---|
  | US | 2.73 (97% accepted) | 1.96 (10%) | 0.46 (40%) | 0.43 (85%) |
  | India | 2.74 (83%) | 1.57 (31%) | 0.68 (33%) | 0.67 (63%) |
  | France | **3.32 (89%)** | 1.48 (2%) | 0.32 (4%) | 0.38 (67%) |

  France's contested records are at the S1's own address; France true matches carry little house-number noise.
- **Word estimator** (decoys sit at SAME at a constant small share alpha of their D count): validated on labelled US
  (corr 0.977, "partners" implied 0.972 vs actual 0.970) and India (0.997). On France it said the same-address records
  are ~97% true, but rule1 (all 54,774 at SAME) was ~58% true on the LB. **So France has a same-address decoy type.**
- **Census** (nD = the S1's records at other numbers): labelled US / India edited-SAME records are 39% / 31% decoys
  when nD = 0 and 1-2% / 6-16% otherwise. Decoys placed at the S1's address leave the shifted cluster short.
- **The two France families** (spread over nD against unedited records, total variation distance; best-file
  acceptance at SAME):

  | family at SAME | words | TV to unedited | accepted |
  |---|---|---|---|
  | suffix-7 | fils, associes, cie, services, groupe, developpement, france (dropped type word or not) | 0.02-0.08 | cie / services 97-99%; fils 23%, associes 6-9%, groupe 1-17%, developpement 2-13%, france 20-33% |
  | type word | club, comite, amicale, ecole, centre, societe, federation, pharmacie, union … | 0.25-0.33 | 0-72%, arbitrary by word (comite 72%, pharmacie 0.1%) |
  | legal (eurl) | | 0.03 | 89% |

  Suffix-7 records are spread exactly like true matches and like cie / services, which the model accepts. Type-word
  swaps at SAME are spread like decoys: the France "same-building" decoy.
- **Check against the LB:** rule1 = 31,052 suffix-7 + 1,264 no-word + 20,877 type + 1,581 other. "Suffix-7 true,
  the rest decoys" predicts 59.0% true; the LB implies 57.5% (my delta model reproduces rule1: −0.00120 at 58% vs
  −0.00125 observed). The fit gives suffix-7 ≈ 97% true and the restored type swaps ≈ 0% true.
- **Probe files** (all on v10_fr3_llm with the same three vetoes; US / India unchanged; validator PASS; `submissions/`):

  | file | md5 | change | break-even | expected LB at 95% right |
  |---|---|---|---|---|
  | `v10_fr3_llm_suf7` | d90b81dc | + 32,316 (rule1 ∩ suffix-7 / no word), France 3.36 per S1 | ~70% true | +0.0016 |
  | `v10_fr3_llm_trej` | 2b7bef61 | − 11,277 accepted same-address type-word records (not compagnie / service / legal) | ~30% decoys | +0.0015 |
  | **`v10_fr3_llm_suf7trej`** | 75b39c01 | both; France 859,904 (3.31 per S1) | | **+0.0031 → ~0.990** |

  - Code: `x_frfix.py` writes `x/restore_fr_suf7` and `x/reject_fr_type`.
  - `x_final.py` gained `REJECT=` (q = 0 for the listed pairs), next to `RESTORE=`.
  - Build: `FROM=France:_v9spw:min,France:_v9s2pw:min,France:_v10s3pw:min RESTORE=restore_fr_suf7 REJECT=reject_fr_type x_final.py OUT _v10plw 0.70`.
- The other ALIAS-vetoed records (Drextavo, Drexveobelo, Drexarc Labs) and the no-address TWIN records are not
  handled yet.

## 2026-09-27 00:25 — two more files: v10b with full LLM coverage, and the France-LoRA probe
- **v10b + LLM, full coverage** (`x_llm.py extend` scored the missing 17,145 val / 94,686 test rows; stack
  `OUT=_v10bpl2 x_llmstack.py _v10bp _v10bp`): 0.99293 at 0.70 (0.99291 at 0.65) — the same as the partial stack
  (0.99294). Coefficients [logit q 0.885, llm 0.251].
  - **`submissions/v10b_fr3_llm2`** (md5 6e1dca1a, validator PASS): + the three France vetoes, thr 0.70. Accepted US
    2,255,859 / India 2,751,539 / France 833,490. Against v10_fr3_llm it adds 6,602, removes 13,875, and changes 19,835 S1
    rows. It supersedes `v10b_fr3_llm` (partial coverage); upload llm2, not both.
- **France LoRA** (`x/llm_lora_fr`, the LLM continued on France pseudo-labels): stacked with `LLM_TAG=_fr x_llmstack.py
  _v10p _v10p` → `x/test_q_v10pl_frw`. Validation (US / India) 0.99281 at 0.70, the same as the original LoRA. Before
  the vetoes it accepts 849,547 France records vs 863,509 with the original LLM (−14k): it learned the vetoes' rejections.
  - **`submissions/v10_fr3_llmfr`** (md5 eb3784dd, PASS): v10_fr3_llm with France re-blended by the France LoRA, then the
    same three vetoes (`FROM=France:_v10pl_frw,France:_v9spw:min,France:_v9s2pw:min,France:_v10s3pw:min x_final.py OUT
    _v10plw 0.70`). US / India are identical to the best file, so the LB delta / 0.15 is France's.
  - France 838,865 → 836,561: it adds 1,055 pairs, removes 3,359, and changes 4,345 S1 rows (1.7% of France S1s). The
    expected LB change is ~±0.0001, close to noise. Low priority: the council expected pseudo-label continuation to repeat
    the model's blind spot.
- Upload order from my side: `v10b_fr3_llm2` (all countries, val +0.00011 over v10 + LLM), then the sibling's
  `rule1_q10ng`. `v10_fr3_llmfr` only if there is a spare upload.
- Hold jobs: 1065360 ended at ~00:17. 1066547 now runs, also on scai04 (so run.sh still works); 1067108 is queued.
- Sibling §10: US / India validation's ceiling is ~0.998. Almost all the remaining error is no-address records whose
  full name equals their S1's and another S1's (7,847 twins; they cap val at 0.99850). v10 picks the right namesake 49%
  of the time. A namesake ranker (`x_amb2.py`) is running there.

## 2026-09-27 00:15 — council: how to bring France into training
Verdict: stop self-training rounds and chasing France recall; the remaining France failure is same-address decoys.
- Build labelled French-looking data: Frenchify train folds 0-3 (a ~30-line dictionary for the decoy words and legal
  forms, labels unchanged), add same-address copies of the distractors (shift removed), and continue only bge (the
  multilingual CE) for ~1h. Frenchify a held-out fold as a French dev set.
- Use the new CE as a France veto vote, and as a filter on rule1's 54.8k rejects: restore only rows above its
  precision-0.75 cutoff on the French dev fold.
- CPU in parallel: France blocking-recall check (records outside the shortlist sharing postcode + house number, vs US);
  a per-S1 expected-F0.5 decision rule (ship only if US/India val improves).
- Already done and rejected (not re-proposed): keep-rate stage features, the down-shift restore, the count-prior match.
- My caveats: "fils/associés/cie are decoys" is a hypothesis. The FILL slice of rule1 has never been uploaded on its own,
  and US true matches also add suffixes at the same address. The same-address copies must not teach "any suffix at the
  same address = decoy", so they should cover only the decoy-word edits, not filler additions.

## 2026-09-27 00:00 — would re-sizing the fold groups help? Probably not (evidence from the logs)
Current allocation (S1 folds, crc32 % 10):

| folds | used by | limited by |
|---|---|---|
| 0-3 (40%) | e5 embedder, the three CEs (small, raw, bge) | compute: the CEs train on a 12M-pair cap (bge 3M), one pass, ~4k pairs/s |
| 4-6 | shortlist model | tiny model |
| 4-9 (60%) | stage 1, cross-fitted, S1K=3 → each model sees 4 folds | data size doesn't matter: saturated (below) |
| 4-7 fit / 8-9 val; test refit on 4-9 | stage 2 | already refit on everything for test |
| 4-7 | LLM LoRA (120k pairs) | compute: 35 pairs/s |
| 8-9 | LLM stack (logistic, 59k rows), threshold | enough rows for 2 coefficients |

- Stage 1 is saturated. Same v9 setup: S1K=3 (4 folds per model) 0.99241, S1K=6 (5 folds per model) 0.99234. More
  stack data did not help, so taking folds from the CE block to give to the stages would not help either.
- The CE and LLM blocks are capped by GPU time, not by the number of folds. Their fold pool is already larger than the
  sample they train on. Giving them more folds only adds entity diversity, and only pays if CE_N / LLM_N and the
  training time grow with it.
- The one change that would use every fold is cross-fitted CEs (a CE on 0-4 scoring 5-9, one on 5-9 scoring 0-4, test =
  mean). Stages 1 / 2 could then train on all 10 folds. Cost: 2× CE training for each of the three CEs plus re-scoring,
  about 6-8 GPU hours, with an expected gain < +0.0001 (stages saturated). Not worth it before the deadline.
- None of it touches France (no labels in any fold), where most of the LB gap sits.

## 2026-09-26 23:45 — v10b + LLM file; rule1 lost on the LB, so R1 is superseded
- **v10b + LLM** (`x_llmstack.py _v10p _v10bp`, the LLM's v10p scores reused for the same pairs): 0.99294 at 0.70
  (0.99291 at 0.65) vs v10 + LLM 0.99282, **+0.00012**. Only 41,507 of v10b's 58,652 unsure validation rows have an LLM
  score. `x_llm.py extend` (QT=_v10p QB=_v10bp, GPU1) is scoring the other 17k val and ~101k test rows. The stack is then
  redone on full coverage.
- bge shrinks France's unsure band (0.01-0.99): 223k → 172k rows. US 210k → 193k; India 130k → 118k.
- **File `submissions/v10b_fr3_llm`** (md5 d3a0aea8, validator PASS): v10b + LLM + the same three France vetoes as
  v10_fr3_llm, thr 0.70. Against v10_fr3_llm it adds 6,470 pairs, removes 12,938, and changes 18,809 S1 rows.
  - Net per country: US −1,057, India −812, France −4,599. France now accepts 834,266 (3.22 per S1).
  - Upload it as its own probe (a model change, all countries). US / India val says +0.00012; France is unknown. It
    lowers France recall, the opposite of what rule1 tried.
- **rule1 scored 0.985819 on the LB (−0.001252)** (sibling §7). About 58% of its 54.8k additions are true, so France
  *does* have same-address decoys; the US / India labels did not carry over. Sibling §9's recommended next probe is
  `rule1_q10ng` (23,336 records, stage q ≥ 0.1, no groupe / développement / france word). It is the only subset that
  gains under both hypotheses.
- **My R1 is superseded:** 3,332 of its 5,065 pairs are in rule1_fill and 3,419 in rule1_q10ng. It is a small slice of
  the q10ng bet with 1.6k extra pairs that failed the stricter rule, so **don't upload R1**. Upload q10ng instead.
  v10b_fr3_llm keeps none of R1 (vetoes still applied).
- Of rule1's 54.8k additions, v10b_fr3_llm itself accepts 826; bge does not reopen the same-address population.

## 2026-09-26 23:40 — v10b (bge-reranker-v2-m3 as a third cross-encoder) and llm2 France-LoRA validation
- **v10b** = v10 CEs + bge on v10's folds, plus v10n's signed-shift features (x_feats was recomputed before v10b_s1),
  S1K=3, BAG=5, PEERS:

  | thr | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 |
  |---|---|---|---|---|---|---|
  | v10b | 0.99276 | **0.99283** | **0.99283** | **0.99283** | 0.99279 | 0.99272 |
  | v10 (same rows) | – | 0.99259 | 0.99263 | 0.99262 | 0.99258 | 0.99248 |

  +0.0002 over v10 (of which the signed-shift part is +0.00004, so bge ≈ +0.00016 — the same as Sarvesh measured on v9p).
  That is below the +0.0003 bar alone, but it is the same size as the LLM blend (v10 + LLM 0.99282), and bge is a
  multilingual model, so its France effect may be larger than validation shows (as the LLM's was: +0.0002 val, +0.001 LB).
  bge ranks #5-6 by stage-1 gain (`ce_b10_s1_rank`, `ce_b10_rank`). Test q → `x/test_q_v10bpw`.
- **llm2 France LoRA on validation** (US / India rows): unsure band 59,650, AUC 0.8347 (the original LoRA 0.844);
  + LLM 0.99281 at 0.70 (+0.00019), so continuing on France pseudo-labels did not hurt US / India. France test
  re-scoring (`llm2_fr_test`, ~223k rows at ~145 pairs/s) runs until ~00:10.
- Next: `x_llmstack.py _v10p _v10bp` (the LLM on top of v10b; launched 23:42), then a v10b file with the same three
  France vetoes as v10_fr3_llm.

## 2026-09-26 23:25 — my R1 vs the structure session's rule1 (sibling folder `AmazonMLChallenge-structure`)
Both are France-only recall probes on top of v10_fr3_llm (0.987071); neither removes any pair.

| file | pairs added | md5 | what it restores |
|---|---|---|---|
| `submissions/v10_fr3_llm_r1` (here) | 5,065 | 03623aac | vetoed records, same number + street, added words stay ≥ 0.7 |
| `../AmazonMLChallenge-structure/submissions/rule1` | 54,774 | 9418efbd | every rejected record passing the strict same-address rule |

- 3,484 of R1's 5,065 (69%) are inside rule1; 1,581 are R1-only (they fail one of rule1's stricter checks, e.g.
  exact first core-name word or the compound sub-number — initials like "TZR Pharmacie → TP" are the likely case).
- rule1 is backed by labels: in US / India validation the same rule's population is 99.93% / 99.18% true, and the
  model rejects only 0.065% / 0.81% of it, vs 6.7% of it in France. rule1 puts France at 3.44 matches per S1 and on
  the train shape of matches-per-S1 (US = India exactly), with ~1k likely false singletons as the known cost.
- Upload order: **rule1 first** (the bigger, label-backed bet; expected ≈ +0.002). If rule1 gains, R1 adds almost
  nothing (1.6k pairs on top). If rule1 loses, R1 tells whether the smaller vetoed subset is still good.
- Note: the sibling's section times (00:05-00:35 "27 Sep") run ahead of the clock; its rule1 file is dated 22:40.
- llm2 chain died silently after saving `x/llm_lora_fr` (23:07; no process left, no "done" line). Relaunched from
  `llm2_fr_val` at 23:25 on GPU0.

## 2026-09-26 23:20 — France veto attribution and recall-restore sets (`x_frrecall.py`)
- Of 866,775 France records the LLM-blended v10 stage accepts, the three self-training vetoes remove 27,910
  (round 1 v9sp 15,998; round 2 v9s2p 20,826; round 3 v10s3p 25,886, of which 5,973 only by round 3).
- 15,280 vetoed records keep the S1's first house number; 11,643 of those are also on the same street.
- Label-free stay rate (share of records adding word w that keep the S1's house number), France: international,
  holding, distribution, snc, participations 0.03-0.04 (pure decoy words); développement, france, groupe 0.20;
  dba / formerly / aka ~1.0 (true-match noise). Agrees with the structure session.
- **R1** = vetoed, same number, same street, every added word stay ≥ 0.7: **5,065 records over 4,870 S1s**. Samples are
  "type word dropped + & Fils / Et Fils / & Associés" at the same address, and initials (TZR Pharmacie → TP). By the
  generator's logic (distractors move to a shifted number) these look like true-match noise that the vetoes reject;
  the structure session found the same "type word dropped + suffix" pattern is true-match noise in the US.
  My earlier hand-labelling of "& Fils" as distractors was probably wrong.
- R2 (rejected, no added word, small downward shift) is dominated by same-name businesses on other streets even after a
  street check — dropped.
- France's shortfall vs US / India (3.23 vs ~3.40 accepted per S1) is ~44k records, close to the structure session's
  50k same-address rejections. If those are true matches, restoring them is the biggest remaining lever.
- Probe file: `output_v10_fr3_llm_r1` = v10_fr3_llm + R1 restored (France only). LB delta / 0.15 = France effect.

## 2026-09-26 23:00 — the sign has little room left on test; France is short on recall
- Accepted records (current best, v10_fr3_llm) with a small house-number shift, up vs down. True-match noise is
  symmetric, so the surplus of upward shifts ≈ accepted distractors:

  | country | up | down | surplus (≈ accepted distractors) |
  |---|---|---|---|
  | US | 17,415 | 15,691 | +1,724 (0.08% of accepted) |
  | India | 47,124 | 49,842 | none |
  | France | 1,104 | 2,212 | none: ~1.1k up-shifted true matches are *rejected* |
- v10n (signed shift features): validation 0.99266 vs 0.99263 (v10, own set, thr 0.65): +0.00003.
- Conclusion: the sign is a real generator fingerprint but worth ~+0.0002 at most; the models already reject nearly
  all shifted distractors. The remaining France loss looks like recall: France accepts 3.23 records per S1 vs 3.40
  (US / India) and 3.46 (train truth); the structure session finds France accepts 51-62% of ambiguous cells that the
  US accepts at 94-100%, and the self-training vetoes removed ~13k France records at the S1's own number.
- Council (5 advisors + peer review) verdict: switch to France recall. Uploads, each one change vs 0.987071, keep if
  ≥ +0.0003: (1) restore vetoed France records at the S1's own number whose added words are filler-like (label-free
  stay rate ≥ 0.7); (2) restore rejected France records with no added word and a small downward shift; (3) depends on
  the 50k same-address check; (4) v10n only if ≥ +0.0003 val; (5) final stack by ~21:00. Drop bge unless ≥ +0.0003.
  P(0.991) ≈ 15%.
- Running: `x_frrecall.py` (veto attribution + R1 / R2 restore sets).

## 2026-09-26 22:45 — direction audit and ceiling check (`x_asym.py`)
Share of non-zero differences (record minus S1) that go UP, true matches vs distractors, train top-1 pairs:

| field | US true | US distractor | India true | India distractor |
|---|---|---|---|---|
| first house number, \|d\|<=13 | 0.498 | **0.994** | 0.481 | **0.923** |
| second house number, \|d\|<=13 | 0.380 | 0.745 | 0.481 | 0.841 |
| core-name tokens | 0.382 | **0.909** | 0.673 | **0.944** |
| core-name characters | 0.433 | 0.850 | 0.514 | 0.815 |
| name tokens | 0.407 | 0.886 | 0.418 | 0.779 |
| zip code, \|d\|<=13 | 0.830 | 0.998 | – | – |
| address tokens / count of numbers | ~0.3-0.55 | ~0.35-0.62 | | (no signal) |

- Distractors grow the name (a word added) and shift numbers up; true-match noise goes both ways.
  Name growth is already covered by `ed_add` / `ed_del`; the numbers and zip had no signed feature.
- **Ceiling on validation (v10):** of the weighted false positives, only 5.5% have a small upward shift (0.9% downward);
  of the false negatives 2.3% / 2.1%. The US/India model already rejects most shifted decoys through other features,
  so the sign can fix only a few percent of the remaining validation error. Any big gain has to come from France,
  where the models never learned the French decoys but the sign transfers without language.

## 2026-09-26 22:20 — the SIGN of the house-number shift separates distractors (`x_numsign.py`)
Train records whose first house number is within 13 of their top-1 S1's but different:

| | record number > S1 number |
|---|---|
| true matches, US | 49.8% (noise is symmetric) |
| true matches, India | 48.1% |
| **distractors, US** | **99.4%** |
| **distractors, India** | **92.3%** |

- A small *upward* shift is almost always a distractor; a *downward* shift is almost always a true match.
  In the US, ~1.1M of 1.6M train distractors have a small upward shift.
- None of our ~70 features had the sign: `int_a0_mindiff` is log1p(min |diff|). The failed NUMVETO rule (below) used
  |diff| too, which is why it rejected true matches.
- France distractors in samples also shift up (14→15, 161→170, 284→295, 311→318).
- Action: `x_feats.py` now adds `int_first_shift` and `int_near_shift` (signed, clipped ±100); plan `v10n` rebuilds the
  stages on v10 (validation verdict tonight). Not yet on the leaderboard.

## 2026-09-26 21:50 — leaderboard
| file | LB | what changed |
|---|---|---|
| **v10_fr3_llm** | **0.987071** | + LLM judge (Qwen3-Reranker-4B LoRA) blended into unsure records: **+0.000994** (val +0.0002, so most of it is France) |
| v10_fr3 | 0.986077 | + round-3 France self-training veto: +0.000135 |
| v10_fr2 | 0.985942 | v10 (distractors in their imitated S1's fold) + round-2 France veto: +0.0012 |
| v9p_frand | 0.984742 | round-1 France veto: +0.00205 (France +0.0137) |
| v9p | 0.982690 | house-number peers, text-aware shortlist |
| v7w | 0.982641 | |
Top 5 were all above 0.990 on 2026-09-26 night.

## Decoy / data structure
- Test size by country (S1 / S2 / S3): France 259,452 / 703,378 / 731,615; US 663,106 / 1,871,330 / 1,945,701; India 809,986 / 2,312,565 / 2,405,000. France = 14.975% of S1s (its weight in the macro F0.5) and 14.4% of the records to match.
- S1 folds (crc32 % 10) are balanced. Distractors used to take a hash of their own id as fold, so validation entities
  kept only 0.73 of their 1.22 distractors; v10 (`DFOLD=1`) puts each distractor in its imitated S1's fold
  (val 0.99285 vs 0.99277 on the same rows).
- Train distractors add country-specific words that true matches never add (US: holdings, group, partners, place words
  like southside/midtown; India: enterprises, industries, ventures, exports, overseas). France (samples): holding,
  groupe, participations, développement, & fils, & associés, cie, legal-form swaps, one generic word swapped.
- France accepts 3.25 records per S1 vs 3.40 for US / India; 6.4% of France S1s are predicted empty vs ~5.7%
  elsewhere, so France may now be short on recall, not only precision.

## Tried and rejected
- NUMVETO (reject a record at a shifted house number shared by another claimant): on validation it rejects accepted
  rows that are 99.6% true matches (US 0.99277 → 0.99074). My test hand-checks that called them decoys were biased.
  Lesson: check any rule on labelled validation before trusting eyeballed test samples. (The sign finding above
  explains the failure: the rule ignored direction.)
- Learned decoy-word veto: precise on train (99.9%) but the models already reject nearly all such records.
- LightGBM + XGBoost judge, bge-reranker alone (Mohanish): ~0. bge as a third cross-encoder: +0.00016 val (Sarvesh);
  being rebuilt on v10's folds (plan v10b).
- Qwen2.5-7B zero-shot judge (Sarvesh): AUC 0.59 on unsure pairs, no gain. Our LoRA-tuned Qwen3-Reranker-4B: AUC 0.844,
  +0.0002 val, +0.001 LB.

## Running (2026-09-27 01:45)
- **Uploading: `submissions/v10_fr3_llm_dd`** (matching md5 75f68b10, candidate 1a8b4f5c, validator PASS; the user
  submits it). Expected ≈ 0.9902 if the hand review / census are right, ≈ 0.9881 if groupe / développement are decoys.
  A result near 0.988 means the next upload should be a groupe/dev-free hedge (`suf7ngtrej`, 7a4b5947, or a dd version).
- Also ready: `v10_fr3_llm_suf7ngtrej` (7a4b5947), `v10b_fr3_llm2` (6e1dca1a), `v10_fr3_llmfr` (eb3784dd).
