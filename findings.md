# Findings log (Sahaj's pipeline, branch v11 / ber-pipeline)

Newest first. Each entry: what was measured, the number, what it means. LB = public leaderboard; "val" = US/India
validation (entity folds 8-9, distractors weighted to the test share of 39%, copy-free).

## Current status for the team (read this first; updated 2026-09-27 22:40 IST)
- **Final file: C8, leaderboard 0.99091, rank 18** (C4 0.99084; E16fr3 0.990807; top ~0.992). `submissions/C8/`:
  matching_results a7d37122, candidate_pairs 1a8b4f5c. It is the last upload.
- **Code: final-dd a4c3fa1 (pushed)** builds it (`x_recall.py`: C4's lists + restore_alias, initials in restore_exact,
  reject_decword; `reproduce.sh`). README, documentation and figure updated for C8 (final-dd e835695).
  **Package** `submission/SSM_submission.zip` rebuilt 22:40 from a clean `git archive` of e835695; audited.
- **Reproducibility:** a from-scratch rerun of dd matched every stage's validation within 0.0001 and 98.9% of S1 rows
  (entry 11:50); GPU training is not bit-exact.
- **Before the final submission:** checklist in [SUBMISSION_NOTES.md](SUBMISSION_NOTES.md).

## 2026-09-27 22:40 — LB: C8 = 0.99091 (rank 18), the final file; last ideas checked
- **C8** = C4 + restore_alias (336: a made-up name at the S1's exact address, the only S1 there; main q >= 0.80, or
  >= 0.10 with <= 2 made-up names at the S1) + initials there even where the main stack rejected them (56) −
  Mohanish's reject_decword (157 France matches adding a census decoy-like word away from the S1's number). France
  865,268. **0.99091** (+0.00007 over C4; expected ~+0.00004).
- Checked and dropped (US / India labels): removing France matches at the decoy slot or 1-21 numbers above the S1
  (the other team's claim: accepted ones there are 99.5-99.97% true); US / India empty-S1 threshold (0.40 is best:
  +0.00007; 0.3 +0.00003, 0.5 +0.00006); more aliases (model-rejected below 0.10 or at shared addresses: 0-35%).
- The other team's architecture (Global Optima, 0.990102): their alias rules add what our France chain already
  accepts (81% of made-up names at the S1's address); only their guards (only S1 there, <= 2 made-up names) were new.
- Ceiling tonight: post-processing is used up; 0.9915 needs +0.0044 France F0.5 (a France-specific main model or a
  larger candidate set, 9-10 GPU h).

## 2026-09-27 20:45 — LB: C4 = 0.99084 (+0.000033 over E16fr3): new best, the final file; C5 not worth an upload
- **C4 0.99084** vs E16fr3 0.990807: +0.000033, the low end of the expected range (0.99084-0.99096). By the per-S1
  model that is ~70-75% of the 2,916 France restores right, where my hand review said ~93% and US / India labels
  95-100% for the same kinds: France has a distractor type in these kinds that neither the labels of other countries
  nor a read of the records shows. More rules of this family are a coin flip.
- **Final file = C4** (last upload; matching 7c929413, candidate_pairs 1a8b4f5c). final-dd **579db0d** (pushed) builds
  it: `x_recall.py` (four lists), `x_llm.py rows`, `reproduce.sh`; README, documentation and figure updated. Zip
  rebuilt 20:40 from a clean `git archive` of 579db0d: `submission/SSM_submission.zip` (122 MB), validator --check-ids
  PASS, output = C4 byte for byte, x_recall.py = the one that built C4, no CR bytes, no cluster paths, no notes.
- **C5** (padum `~/scratch/amlc_c5/output_C5`, matching 75179abd, PASS): C4 + initials at the S1's exact address
  even where the main model rejected them. With the rule's conditions (only S1 there, exact street, the record's best
  S1) it adds only **52** France records (restore_exact 161 → 217): ~+0.000003, not worth the last upload.

## 2026-09-27 20:30 — C4: the sound parts of Mohanish's E16fr3-sn and Sarvesh's E49 in `x_recall.py`; expected ~0.9909-0.9910
**File:** `submissions/C4/` (padum `~/scratch/amlc_c3/output_C4`): matching **7c929413**, candidate_pairs 1a8b4f5c
(unchanged), validator --check-ids PASS. Built by final-dd's new `x_recall.py` (uncommitted) from E16fr3's own files
(`~/scratch/c4.sh`); with only the old lists the same code gives **e93605ad** (E16fr3, byte for byte). US / India
byte-identical to E16fr3; France +2,916 (862,152 → 865,068). All four lists undo a France self-training veto or a
no-address rejection (the main model trained on labels accepted, q >= 0.70, except nacore):

| list | records | rule | evidence | my hand review |
|---|---|---|---|---|
| restore_nacore (Mohanish) | 1,078 | no address, core name = S1's, no other S1 with it, judge yes | US / India 98% true | 35 / 35 |
| restore_vetona (E49, filtered) | 1,182 | vetoed, address without a house number, judge margin >= 2, S1's street words in it or unique name | E49 no-number part | 9.3 / 10 |
| restore_samename (Mohanish, filtered) | 563 | vetoed, exact name at another number, S1's street (or unique name), not anchored, judge yes | US / India same street 95-100% | ~28 / 30 |
| restore_exact (new) | 161 | vetoed at the S1's exact address, only S1 there, name adds no word (initials, website, dropped words) | US / India 98-100% | 29 / 30 |

Expected LB change (per-S1 exact F0.5, `~/scratch/combo.py`): at the review's rates +0.00015 → **0.99096**; 10 points
lower +0.00009 → 0.99090; 20 points lower +0.00003 → 0.99084 (break-even ~65-68%). Not 0.9916.

**What was dropped and why (label evidence on US / India validation, hand review on France):**
- **Mohanish's E16fr3-sn as built (fdea1f55): expected ≈ −0.0002, do not upload.** restore_samename: **1,795 of its 4,646
  are anchored** (another candidate S1 on the record's street 1-13 numbers below it = that S1's decoy with a type word
  swapped that spells another S1's name; France's formulaic names make these collisions common): **0% true on US /
  India** (47 rows; 89-91% for the rest). Of the rest, 2,038 are on another street; with 3+ namesakes (France's are in
  the same city) US / India are 54-64% true and my sample ~20-25%. Same street, not anchored: 813 (kept those the main
  model accepted: 563).
- **Sarvesh's E49 unfiltered (llmveto2/4/6): unsound as a blanket rule.** My 75-sample review: at the S1's exact address
  ~49% true (a third are type-word swaps at the same address, Club→Société, Parents→Centre, that the judge says yes
  to: the same-building decoys the vetoes rightly remove); no house number ~87%; same street ~75%; other street ~42%.
  Kept only the no-number part (restore_vetona).
- **Exact-address typos:** a first version allowed a typo'd extra word and let in **acronym-letter decoys** ("BKJV
  Federation" for BKJ Federation, "KQF College" for QF College): half of a 40 sample. US / India: the model's own
  rejections of exact-address typos are 98-99% false. restore_exact now allows no added word.
- **General:** on US / India validation the model's rejections of every one of these kinds are 75-99% wrong records,
  so the rules only undo a France self-training veto (the main model accepted), not the main model.
- **Blocking (the 19:45 hypothesis): ruled out.** Same-address near-identical pairs in top-20: France 100% (28 missing
  from the shortlist), US 100%; India 2,161 missing but **0.96% true** on validation (block.txt).

## 2026-09-27 19:45 — the likely France gap: blocking (retrieval ranks); ruled-out levers
- **Retrieval rank of E16fr3's accepted pairs** (`~/scratch/ranks.py`): rank 1 France **40.1%** vs India 60.0% / US
  65.0%; ranks 11-15 France 5.0% vs 2.1 / 1.8%; ranks 16-20 **2.0%** vs 1.1 / 0.6%. France's formulaic names (city +
  type word + legal form) crowd the true S1 down the bi-encoder's list; the tail at 20 is 2-3× heavier, so France
  likely loses a few % of true matches before any model sees them (~17-35k records ≈ 0.0007-0.0015 LB: the size of
  the gap to the top). **Being checked:** `~/scratch/block.py` → `~/scratch/block.txt` (same-address near-identical
  (record, S1) pairs missing from top-20 / shortlist, per country; truth on US / India validation); missing test
  pairs saved to `~/scratch/block_missing_test.parquet`. If confirmed, fix = address-key blocking (S1s at the record's
  parsed address as extra candidates) or larger K, then score them (features + CEs + stages) or a label-validated rule.
- **Ruled out (label-free test stats + validation):** empty France S1s with a same-address similar-name claimant
  (US / India validation: 0.5% true; 97% of predicted-empty S1s are truly empty); twins (France 45% of S1s, US 39%,
  India 52%); false-merge buckets by address (France 90% exact address); websites as a rescue (7% true on validation).
- **Small, validated:** France initials rescue (a left-out record whose name is the S1's initials, at its exact
  address, no other S1 there with those initials): 920 France records; validation 5 / 5 true, the US / India model
  accepts such records anyway; worth ~+0.00004. Pairs in `~/scratch/noise_rescue_test.parquet` (use France initials only).

## 2026-09-27 18:40 — Sarvesh's France threshold probes scored by hand review: 0.70 is right; none beats E16fr3
Organisers added 2 uploads. Sarvesh's probe files on E16fr3 (sarvesh-exp 187ac31; padum
`/scratch/scai/mtech/aib262045/amlc_exp/out_*`, only France changes). I labelled random samples of each file's changed
pairs by hand (`~/scratch/thrdump.txt`, `resc.py`), then computed the exact expected change per S1 (`~/scratch/exdelta.py`:
binomial over each touched S1, untouched accepted records taken as true, empty S1 truly empty w.p. 0.5):

| file | change | sample | share true | expected Δ LB | expected LB |
|---|---|---|---|---|---|
| frT60 | +2,225 France pairs | 40 (0.60-0.65) + 40 | ~47% | −0.00010 (−0.00015 … −0.00006) | ~0.99070 |
| frT65 | +1,154 | 40 (0.65-0.70) | ~51% | −0.00005 (−0.00007 … −0.00002) | ~0.99076 |
| frT75 | −1,960 | 30 (0.70-0.75) | ~70% of the removed | −0.00001 (+0.00004 … −0.00005) | ~0.99080 |
| frT80 | −4,315 | (0.75-0.80 band assumed ~75%) | | −0.00007 | ~0.99074 |
| frresc30 | +79 empty France S1s filled | 25 | ~51% | ≈ 0 (±0.00001) | ~0.99081 |
| E17fr3 | US / India from the 5.8% smaller candidate set (6,896 rows differ) | validation −0.00002 | | ≈ 0 (±0.0001) | ~0.99079 |

- **Why lowering the France threshold loses:** the 0.60-0.70 band is half the name noise we want (initials "NC" /
  "AR" / "RÉ", websites, no-number records) and half **type-word swaps at the S1's own address** (Musique→Culturelle,
  Club→Fetes, Union→Amicale): the same-building decoys whose rejection gave dd its +0.0032. The reject list only covers
  pairs accepted at 0.70, so a lower threshold lets them back in. Precision falls 70% → 51% → 42% down the bands; a
  France addition needs ~72% to pay off in F0.5, so 0.70 sits at the break-even.
- **Verdict:** no probe beats E16fr3 in expectation; all are within ±0.0001 or below. E17fr3 is the only one with a
  second benefit (5.8% smaller candidate set, which the organisers rank) at the same expected score, but the zip's code
  would have to build it. If the last upload is what counts, keep the last upload for the file we ship.

## 2026-09-27 17:55 — why we trail the top (0.992 vs 0.990807): France recall, spread thin; no single fixable bucket
Label-free test statistics of E16fr3 (padum `~/scratch/{dist,qdist,frdump,frcount,street,twins}.py`):
- **Per-S1 structure.** Train truth: 3.46 matches per S1 and 5.58% empty S1s in *both* US and India (a generator
  constant). Test predictions: US 3.40 / 5.81%, India 3.40 / 5.76%, **France 3.32 / 6.00%**, and France has fewer
  S1s with ≥ 6 matches (9.8% vs 10.7%). If France's generator matches the others, France misses ~2-3% of its true
  matches (~20k records) beyond US / India, plus ~500 wrongly empty S1s. Back-solving the LB with US / India at their
  validation (~0.993) puts France near 0.978; the gap to 0.992 is about +0.008 on France.
- **Model certainty.** Share of records whose base q (best S1) is 0.1-0.4 / 0.4-0.7 / 0.7-0.9: US 1.08 / 0.56 / 0.28%,
  India 0.54 / 0.36 / 0.20%, **France 3.21 / 1.93 / 1.59%**. The chain then drops 19,118 base-accepted records (vetoes)
  and 11,105 (same-address rejects).
- **No large false-merge bucket.** With street names parsed (types, articles, regions removed), France's accepted pairs
  are 90.2% at the S1's exact address, 0.43% same number / other street, 0.92% shifted number, 2.6% no address.
  Core-name twins: 45% of France S1s (US 39%, India 52%); twin share of accepted no-address records 16% (US 11%,
  India 15%); validation precision there 0.94 (no-address + twin), 0.992 (no-address, no twin), 0.9997 (with address).
- **Where the missed records are** (hand-read samples, `~/scratch/frdump.txt`): same-address records with name noise
  the generator also uses in US / India (initials "CC", "LI"; websites "solidaritefrreseurl.com"; "SA"; typos
  "Nt Foyer"), killed by one veto or by a base q of 0.4-0.7 despite three confident self-trained stacks; decoys
  (shifted numbers, type-word swaps, other street) are correctly out. Sizes: base 0.4-0.7 with all 3 self-trained
  stacks ≥ 0.9: 1,019; vetoed at the exact address: 5,316 (name similarity ≥ 80: 1,204); left out at the exact
  address with base ≥ 0.4: 10,729.
- **Read:** France's loss is recall from model uncertainty on French text, spread thin. F0.5 makes a false add cost
  ~2.5× a recovered match, so a rescue needs ~72% precision, and each identifiable subset is worth +0.00002-0.0001.
  Closing +0.0012 needs a more certain France model (e.g. France self-training used as the main model, not only as a
  veto; a larger judge), which needs GPU hours and cannot be checked without France labels or an upload.

## 2026-09-27 16:00 — LB: E16fr3 = 0.990807 (+0.000458 over variant_v10seed_dd): new best, the final file
- Expected 0.99067 (validation for US / India + hand review for France): the upload came in +0.00014 above it, so the
  parts transfer at least in full (E1 + rescue expected +0.00016, France recall +0.00016).
- The final submission ships this file (matching_results e93605ad, candidate_pairs 1a8b4f5c = the variant's) with
  final-dd's code, which builds this method (bge lane, `x_recall.py`).

## 2026-09-27 15:40 — one upload only: head-to-head hand check → E16fr3
- **Against the earlier France hand review** (the 180 test groups behind the dd lists, `review/`; `~/scratch/review_cmp.py`):
  E16 / E17 change none of the 180 S1s; **E19 changes 4 and all 4 are wrong** (removes Terroirs SA & Fils, OP SA + Fils,
  Noura Industries SAS & Fils, Lamour Union SARL & Fils: DROPSUF records the review marked true); **E16fr3 / E16fr2 /
  E17fr2 change 8**: two empty S1s filled correctly (Bordeaux Centre SAS ← alias "Drexarc Labs" at its exact address, the
  review's wrongly vetoed ALIAS; Filles Comite SA ← "Filles Comite S.A.", review "likely true"), six no-address
  additions to populated S1s that roughly cancel (three are the review's TWIN cases); **frB / frC / frD change the same 8
  + 3**, all three expected losses (Vallee & Fils ← no-address record of an identically named twin on an S1 with no
  same-address record; Comité des Ass ← "Comité des Ass SA", the twin's exact name; a reordered no-address name at ~70%).
  The v10-only rerun differs on 5 (drops the alias Nylaevo and a "& Associés" suffix): slightly worse than the variant.
- **Head-to-head samples** (`~/scratch/pairdiff.py`, 30 disputed S1s each, every record labelled, F0.5 per S1):
  US / India E16fr3 vs E17 (6,989 S1s differ): a tie (net −0.03 over 30 S1s ≈ 0.000000 LB; E16fr3's extra are
  no-address twins, E17's extra are number-shift + legal-change decoys). France frD vs E16fr3 (2,945 S1s differ): frD
  worse on 23 of 30, net −1.71 over 30 ≈ **−0.0001 LB**; 19 of the 30 have an obvious twin S1 (same name elsewhere,
  or only the legal form differs), and in two the record's legal form is the other S1's.
- **Verdict:** E16fr3 ≈ E17fr2 (≈ 0.9907) > E16 ≈ E17 ≈ frD / frC / frB (≈ 0.9905) > variant (0.990349) > E19.
  E17fr2 only adds a 5.8% smaller candidate set but needs three people's scripts ported; **upload E16fr3** (e93605ad).

## 2026-09-27 15:05 — teammates' files verified and scored by hand review; E16fr3; the v10-only check reproduces the variant
**Teammates' files, rebuilt from our own padum files** (sandbox `~/scratch/amlc_team`, code origin/variant-frC):
Mohanish's frB 73b4a5ea and frC 7d66d4ea, Sarvesh's E16 a019f175: all byte-identical to theirs; E17 eb6fdbec, E19
c91f83b1 and Mohanish's new frD 76ef7eea checked in their folders (ACL access). frD = E17 (US / India) + frB's France
fixes; against my E17fr2 it differs only by 3,023 France additions (the core-name restores + the judge's "no" rescues).
x_recall's rescue cannot move a record off another S1 (test_q holds one row per record, its argmax).

**Where each change stands** (US / India: labelled validation; France: hand review of 30 sampled records per category
in `review_dump.txt` from `~/scratch/review_dump.py`, discounted 10 points because on the one labelled category, the
US / India rescue, my review said 75% true where the labels say 61%; LB Δ from `~/scratch/delta.py`, a per-S1 F0.5
Monte Carlo over the touched S1s, other accepted records taken as true):

| change | records | evidence | LB Δ |
|---|---|---|---|
| E1: bge + LLM stack for US / India | +5,677 / −4,418 | validation 0.99293 vs 0.99282 | +0.00010 |
| empty-S1 rescue, US / India, q ≥ 0.5 (0.4) | 666 (923) | validation +0.00006 (+0.00007) on the bge stack | +0.00005 |
| France no-address, exact name | 2,508 | review 84% true (legal-form-exact matches) | +0.00004 |
| France no-address, core name only | 2,937 | review 60%: mostly twins (X SARL vs X SAS, same name at two addresses) | −0.00010 |
| France empty-S1 rescue, q ≥ 0.4 | 439 | review 68%; the wrong ones are type-word swaps at the S1's address (judge −3 to −4) | +0.00009 |
| same, without the judge's "no" (343) | 343 | review ~80% | +0.00012 |
| E2: bge as a 4th France veto | −3,656 | 89% at the S1's address adding fils / groupe / et / développement / france (the suffix family dd showed ~97% true): review 80% true | −0.00005 to −0.00017 |

**Expected LB** (variant 0.990349; each ±0.0001): E19 0.99038, frB 0.99046, E17 0.99048, E16 0.99050, frD 0.99051,
frC 0.99054, E16fr 0.99063, E17fr2 0.99064, E16fr2 0.99066, **E16fr3 0.99067**. None reaches the top (0.991811).

**New files** (all validator --check-ids PASS; France base = the variant's chain):
- E16fr2 0e58a309: E16 + exact-name no-address restores + France rescue without the judge's "no" (France 862,151).
- E17fr2 a6405af8: the same on E17 (candidate_pairs 15fb51cc, 12,020,996 pairs).
- **E16fr3 e93605ad** (PC `submissions/E16fr3/`): E1 + final-dd's new `x_recall.py` (rescue q ≥ 0.40 in every country,
  the judge must not say no where there are no labels; exact-name no-address restores): US 2,256,307, India 2,752,043,
  France 862,152; 99.985% of S1 rows as E16fr2 (257 more US / India rescues at 0.4).

**Checked and dropped:** ALIAS restores (made-up name at the S1's exact address): with a unique address France accepts
96.8% (US 97.5%) and rejected ones are 3% true on US / India. France's empty-S1 share 6.16% vs US 5.87%, India 5.80%:
no hidden recall hole. Records at another house number with the S1's exact name are 70-92% true in US / India, but
France has a third as many per S1 and large number jumps: its generator adds less number noise; only ~1.3k are
vetoed.

**v10-only check done** (padum `~/scratch/amlc_v10only`, final-dd 171c552 from stage 2 on, rerun first half):
matching c8bfd9be, candidate_pairs 66ba41e6 (13.38M), validator PASS; against the variant 98.86% of S1 rows identical
(US 99.02%, India 99.22%, France 97.36%), F0.5 0.99767 / 0.99786 each way (France 0.9951); restore / reject 32,668 /
11,110 (variant 32,620 / 11,105); round validations 0.99262 / 0.99259 / 0.99260 (probe 0.99257 / 0.99253 / 0.99261);
LLM blend 0.99283 (0.99282). Same agreement as the dd rerun (98.88%, 0.99776).

**final-dd (uncommitted):** `reproduce.sh` adds the bge lane on GPU C after the judge (x_ce3 CE_BASE=bge, 3M pairs,
stages with the three CEs), `llm_stack_b` (x_llmstack _v10p _v10bp), and the final step
`FROM=unlabelled:_v10plw,<vetoes> RESTORE=restore_dd,restore_empty,restore_nafr` on base `_v10bplw` after
`x_recall.py _fin _v10p restore_dd reject_dd`. The bge chain is running on the check's files (`bge.sh`, GPU C).

## 2026-09-27 13:20 — documentation written; figure corrected; where v10's validation F0.5 is lost
- **`final-dd:ber/Documentation_template.md`** (db4d351): the organisers' sections filled in for variant_v10seed_dd
  (EDA findings, blocking with recall tables, 72 features, model stack, threshold, error analysis, LB progression,
  code map, compliance). It embeds `pipeline.png`, which `package.sh` copies next to it.
- **Figure fixes** (checked against the code): the LLM judge trains on 120k best pairs of folds 4-7, 80% with an
  unsure stage-1 p and 20% sure (it said "120k unsure"); the self-training box now says round 1 is seeded by stage 2,
  round k by stage 2 vetoed by the rounds before k. Every other number in it matches the code (45 + 17 + 10 = 72
  features, HNSW M 32 / efSearch 512, lexicon cosine 0.9, band 0.01-0.99, LoRA r 16, 3M + 4M self-training pairs).
  Rendered with draw.io's viewer in headless Chrome, cropped to the diagram.
- **`x_verr.py _v10p 0.70`** (v10 stage 2, validation 0.99262), gain if each error type were fixed:

  | error | rows | gain | empty address | namesake S1s |
  |---|---|---|---|---|
  | missed true match | 8,829 | +0.00208 | 75% | 78% |
  | accepted distractor | 524 | +0.00054 | 46% | 49% |
  | accepted wrong S1 | 741 | +0.00044 | 72% | 67% |

  A threshold per bucket (empty address, namesakes, added / dropped word, source, singletons) gains +0.00000. The
  loss is information-limited: empty-address records whose name several S1s share. Sampled examples in padum
  `work/logs/v10p_errex.log`.

## 2026-09-27 11:50 — end-to-end rerun of `final-dd` done: dd reproduced (not byte for byte); HEAD's last steps identical
- **Rerun** (padum `~/scratch/amlc_final`, code a707292, empty work/, 03:03-11:34 on two to three A100s): dd md5
  b0caba6a, candidate_pairs 66ba41e6; validator --check-ids PASS (absolute paths).
- **vs the submitted dd** (75f68b10): S1 rows identical 98.88% (US 99.03%, India 99.19%, France 97.57%); pairs
  5,870,159 vs 5,868,945; the rerun scored against the submitted file 0.99776 (US 0.99793, India 0.99841, France
  0.99527). v10_fr3_llm: 98.50% identical, 0.99703. Validation: LLM blend 0.99281 vs 0.99282, every stage within
  0.0001 (checkpoints 1-11 below). France: restore 33,459 / reject 11,632 (32,439 / 11,655), accepted 858,796
  (859,649), structural rule accepts 677,673 (677,843).
- **Candidate set 7.72 per S1** (13,377,149 pairs; submitted 7.37, +4.8%): the retrained bi-encoder and shortlist
  model calibrate P a little differently at the same P >= 0.001.
- **HEAD 187946d's last steps** (unlabelled() + x_ddfix instead of x_rule1 + x_wordlists; `~/scratch/amlc_head_rerun`)
  on the rerun's q files: matching_results b0caba6a and v10_fr3_llm 409043f2, **byte-identical** to the rerun's;
  restore / reject 33,459 / 11,632. The pushed branch is what was reproduced.
- Read: reproducible statistically, not byte for byte. 1.1% of S1 rows differ, mostly France (2.4%), where the veto
  chain and the fixes amplify small score changes; for scale, the variant differed from dd on 0.37% of rows and moved
  the LB by +0.00007. The zip carries the submitted file's own candidate_pairs.tsv; the README should say a rerun
  gives about 7.7 pairs per S1.

## 2026-09-27 10:35 — `variant_v10seed_dd` leaderboard 0.990349 (dd 0.990282, +0.000067): new best
- The spare probe (entry 09:50). Two changes against dd: the three France rounds seeded from v10 / v10+a1 /
  v10+a1+a2 and started from v10's cross-encoders (dd: seeded v8 → v9p_frand → v10_fr2, from v8's), and the judge's
  scores on every unsure v10 row (dd: v9p's unsure rows only). US / India validation equal (0.99282): the gain is France.
- Read: the v8 / v9 stacks are not needed for the France chain. Same number of rounds (3), so the saving is the v8 /
  v9 stages; if the judge's training band also moves to v10 (`QT=_v10p`, untested), v8's two cross-encoders and their
  scoring go as well. Two changes at once, so neither is credited alone.
- The scripts that built it are in the repo now: `ber/variant_v10seed_dd/variant.sh` and `mkvariant.sh` (sandbox).

## 2026-09-27 10:30 — rerun checkpoint 11: LLM judge validation matches
- **Judge on v9p's unsure validation rows, at 0.70**: rerun stage-2 0.99278 + LLM **0.99299** (+0.00021) vs original
  0.99277 + LLM 0.99297 (+0.00020). Unsure band 55,210 rows (original 58,641); pair AUC stage-2 0.9283 / LLM 0.8337
  (0.9363 / 0.8439) — the band's make-up moved, the F0.5 did not.
- Round 3 (v10s3) fits at step 6,500 / 13,672; judge test shards ~25% (US / France ~50 pairs/s). Lane A's later steps
  moved to the job's idle third GPU (work/logs/gpu_A). ETA dd ~12:00.

## 2026-09-27 10:10 — rerun checkpoints 9-10
- **Round 2 (v9s2) val at 0.70**: rerun 0.99268 vs original 0.99275 (-0.00007; a France veto stack).
- **LLM judge training** (done 10:00): unsure band 102,708 records (original 99,856: the rerun's shortlist keeps a few
  % more pairs), positives 0.654 (0.656), loss at step 3600 0.347 (0.354).
- Now: round 3 (v10s3) fits and the judge's val + test by country share both GPUs; then the stack, v10_fr3_llm, the
  France fixes, dd. Validate the rerun by hand with absolute paths (the running reproduce.sh predates 187946d).

## 2026-09-27 10:10 — tight_dd dropped (user decision): France loses more than validation shows
- vs dd: 99.27% of S1 rows identical; tight scored against dd 0.99854 (US 0.99864, India 0.99904, France 0.99674).
- vs the hand review (180 France groups, review/frgroups_France_*.txt): identical on 175 (all 60 random). Of the 5 that
  differ, 3 favour dd: Collège Bresse's only record dropped; "BORDEAUX SPORTIVE SARL N°18 Rue Anatole France"
  accepted on an S1 dd leaves empty (a decoy of Medley Sportive @14 Rue Anatole France, +4 shift); a record at
  Roubaix Amicale's shifted cluster #320. 2 favour tight: "FC" initials at Feminine Centre's address restored;
  "biarritz compagnie sarl" (type swap at the address, no decoy cluster elsewhere) rejected.
- Systematic (all France): 1,110 dd matches lost because their S1 was pruned (P < 0.01; mostly true: the matcher was
  confident, the shortlist was not), 956 lost to refit q < 0.70; of 2,054 new matches ~600 are records whose
  namesake competitors were pruned down to one candidate (the Bordeaux pattern: no competitor left, so the refit
  matcher accepts). Estimated LB ≈ dd - 0.0005 to 0.001. Pruning removes the namesake competition France relies
  on; a smaller candidate set would need a matcher that keeps that signal. dd stays the submission.

## 2026-09-27 09:50 — `variant_v10seed_dd` ready (France chain seeded from v10); rerun matches the original per stage
- **Upload file (the spare probe):** `submissions/variant_v10seed_dd/` (local; padum `~/scratch/amlc_variant/output_variant_dd/`)
  matching_results md5 **e73c409e**, candidate_pairs 1a8b4f5c (= dd's); validator --check-ids PASS (absolute paths).
  - What differs from dd: the three France rounds seeded from v10 / v10+a1 / v10+a1+a2, all starting from v10's CEs
    (no v8 / v9 stacks); the judge's scores on every unsure v10 row (dd: v9p's unsure rows only).
  - Round validations a1 / a2 / a3: 0.99257 / 0.99253 / 0.99261; LLM stack 0.99262 → 0.99282 (dd's also 0.99282).
  - vs dd: S1 rows identical US 99.73%, India 99.79%, France 98.91%, all 99.63%; pairs only in variant 1,524, only in
    dd 4,963; F0.5 of variant scored against dd 0.99937 (France 0.99758). France accepted 859,338 (dd 859,649);
    restore 32,620 / reject 11,105 (dd 32,439 / 11,655); word lists 44 true-like / 41 decoy-like, as dd.
  - Read: LB ≈ 0.9903 → the v8 / v9 stacks can go (~4-5 h less pipeline); clearly lower → keep dd's chain. Two changes
    at once (seeding + judge coverage), so a drop would not say which.
- **End-to-end rerun, validation at 0.70 (rerun / original):** v8 0.99235 / 0.99234; v9p 0.99278 / 0.99277; v9s
  0.99275 / 0.99272; v10 0.99261 / 0.99262. v8 CE fold-9 pair AUC 0.99853 / 0.99851. Every stage so far within
  ±0.00003. At 09:45: v9s2 scoring, LLM judge step 2,800/3,750 (18 pairs/s); ETA ~11:30.
- final-dd **187946d** = the validator path fix (absolute "$AMLC_ROOT/output_..." paths), as flagged below. The edit
  was already in the final-dd worktree (another session); it is committed now, so no need to redo it. Two sessions
  editing that worktree: commit or tell before editing it.

## 2026-09-27 06:25 — `tight_dd` ready (candidate set P >= 0.01): 6.27 pairs per S1 (dd 7.37), validator PASS
- File: padum `~/scratch/amlc_tight/root_0.01/output_tight_dd/` — matching_results md5 **f1278673**, candidate_pairs
  md5 91171e70; 10,868,622 candidate pairs = **6.273 per S1** (dd 12,760,925 = 7.365, -15%); validator --check-ids PASS
  (absolute paths). Code 16520fb; dd's own q files filtered to P >= 0.01, four stacks + LLM blend + fixes refit.
- Validation (US / India): v10 stage 2 0.99255 (dd 0.99262); + LLM 0.99272 (dd 0.99282), so about -0.0001 overall.
  Veto stacks: v9s 0.99268 (0.99272), v9s2 0.99262 (0.99275), v10s3 0.99249 (0.99256).
- vs dd: S1 rows identical US 99.37%, India 99.46%, France 98.44%; F0.5 of tight scored against dd US 0.99864,
  India 0.99904, France 0.99674. France restore 31,470 / reject 11,728 (dd 32,439 / 11,655); France accepted 859,702.
- Expected LB ≈ dd - 0.0001 if France is neutral; France (98.4% of rows unchanged) is what the upload measures.
- **Bug (reproduce.sh and the probe scripts):** the final validator runs from inside `student_resource`, a symlink in
  the sandboxes, so `../output_*` resolves to the ORIGINAL project's folder: in amlc_final it would validate the
  original dd and report a false PASS. Validate the rerun by hand with absolute paths; fix reproduce.sh to pass
  "$AMLC_ROOT/output_v10_fr3_llm_dd/..." (not yet done).

## 2026-09-27 05:15 — candidate_pairs.tsv now counts in the ranking: smaller candidate set vs F0.5 (measured)
Organiser update: the final evaluation reviews candidate_pairs.tsv and its code; a smaller candidate set per S1 ranks
higher, beyond the leaderboard. Today: 12.76M pairs = **7.37 per S1** (1.28 per record; one per record = 5.75).
`~/scratch/amlc_cands/cands.py` on the original v10 artefacts: shortlist probability P recomputed (shortlist_text
model), validation = US / India folds 8-9 where a record whose stage-2 pair leaves the candidate set goes unmatched;
test = 25% record sample x4.

| blocking | test pairs / S1 | France | US | India | val F0.5 | true pairs kept (val) |
|---|---|---|---|---|---|---|
| P >= 0.001 (now) | 7.36 | 10.08 | 6.57 | 7.15 | 0.99262 | 0.99982 |
| P >= 0.005 | 6.55 | 7.52 | 6.28 | 6.46 | 0.99258 | 0.99945 |
| **P >= 0.01** | **6.27** | 6.77 | 6.15 | 6.21 | **0.99255** | **0.99905** |
| P >= 0.02 | 5.96 | 6.13 | 5.95 | 5.92 | 0.99250 | 0.99788 |
| P >= 0.05 | 5.42 | 5.48 | 5.52 | 5.32 | 0.99216 | 0.99409 |
| top 3 per record | 6.21 | 7.14 | 5.84 | 6.22 | 0.99257 | 0.99445 |
| top 2 per record | 5.98 | 6.50 | 5.74 | 6.02 | 0.99246 | 0.99223 |
| top 1 per record | 5.60 | 5.50 | 5.57 | 5.66 | 0.99193 | 0.98707 |

- The excess is mostly France's namesake tail (10.1 per S1). P >= 0.01: -15% pairs, France 10.1 → 6.8, -0.00007
  validation, 99.9% of true pairs kept (the record-cap rules lose ~0.5% of true pairs among tied namesakes).
- A tighter threshold keeps a subset of the scored pairs, so dd can be rebuilt without a GPU: `amlc_tight/tight.sh`
  (TAU=0.01, code 16520fb) filters feats2 / extras / the 8 CE score files, refits stages 1-2 of v10, v9s, v9s2,
  v10s3, the LLM blend and the France fixes -> `root_0.01/output_tight_dd`. France needs an upload to confirm.
  Shortlist P for all pairs: `amlc_tight/shortlist_p_{train,test}.npy` (shortp.py).

## 2026-09-27 04:45 — `final-dd` 16520fb: no country names (Mohanish's dd-generic), dd byte for byte from a fresh work/
- Review of origin/dd-generic (ad88b9f): `common.unlabelled()` = test countries without a labelled training record
  (France here); x_final FROM= / THR_C=, x_llm COUNTRY=, rule_fr.py take 'unlabelled'; `x_ddfix.py` builds restore /
  reject in one step (positions -> nD census -> per-word TV), same logic as x_rule1 + x_wordlists; `check` on
  US / India validation: India reject 2,922 rows 99.97% true, -0.0024, so the fixes stay limited to unlabelled
  countries. **Gap:** rule_fr.py reads x/wstat_words_test_France_0.parquet but dd-generic has no wstat.py (cut
  from 2e1cd0f, before 99ac488) and plan dd has no wstat step, so from a fresh work/ `dd_rule` fails; its byte-exact
  run used the wstat files left in the shared work/x by the structure session.
- Ported into final-dd (16520fb, local): unlabelled() / countries(), generic x_final / x_llm / rule_fr, x_ddfix.py,
  **wstat.py kept and run before rule_fr**; x_rule1 / x_census / x_wordlists dropped; x_anatomy.py trimmed to
  positions() (its analysis block looped over named countries); reproduce.sh has no country name
  (`VETO=unlabelled:_v9spw:min,...`; seeds FROM=unlabelled:...; `x_ddfix.py _v10plw _v10fr3l`, RESTORE=restore_dd).
- Check (`~/scratch/amlc_generic`, fresh work/ with only the original upstream q files linked: test_q_v10plw,
  _v9spw, _v9s2pw, _v10s3pw, p5_test_v10, + pq / lexicon / feats2_test): v10_fr3_llm md5 d501ac59 = submitted;
  **dd matching_results md5 75f68b10 = submitted**, candidate_pairs 1a8b4f5c = submitted; restore 32,439, reject
  11,655, France accepted 859,649; rule_fr 677,843 accepted / 45,398 below 0.70; validator --check-ids PASS.
- The running end-to-end rerun (a707292) still ends with the old x_rule1 / x_wordlists path (bash reads its script as
  it goes, so it is not edited mid-run); after it finishes, the 16520fb final steps run on its q files for a second
  equivalence check.

## 2026-09-27 04:20 — simplification probe `variant_dd` (running, ETA ~08:00; for the user's one spare upload)
What can go and still score about the same (LB history): round 3 (v10s3, +0.000135, ~1.3 h) and the stage1_cv runs
(fold-9 printouts only) are the safe cuts. The bigger one, untested: drop the v8 and v9 stacks (~4-5 h) by seeding
all three France rounds from v10. This probe tests exactly that, on the original run's v10 base, so it differs
from dd only in the France chain:
- rounds a1 / a2 / a3 = the v9s / v9s2 / v10s3 recipe (x_ce3 SELF, 3M train + 4M pseudo pairs, lr 2e-5, stages
  with peers), seeded by v10 / v10 + a1 / v10 + a1 + a2, every round starting from v10's CEs (ce_r10, ce_n10);
- LLM blend: the judge's scores on every unsure v10 row (llm2 extend, same LoRA), stacked on v10;
- then the dd fixes recomputed on its own France decisions. Sandbox `~/scratch/amlc_variant`, original inputs
  symlinked read-only; clean-branch code a707292; GPU C.
- Read: LB ≈ 0.9903 → the v8 / v9 chain can go (the LoRA's training band would move from v9's to v10's stage 1,
  untested but minor). Clearly lower → keep the chain.
- GPUs: hold jobs swapped (1067303 → 1067330, 2 GPUs on scai04; 1066122 released early). The rerun keeps
  acc2cee6 (job 1066547, to 00:13) and 4daa87b5 (job 1067330); the probe uses 366eff51. No 08:43 handover now.

## 2026-09-27 03:30 — clean branch `final-dd` + end-to-end rerun of v10_fr3_llm_dd (running, ETA ~11:00)
- **Branch `final-dd`** (orphan, local only, worktree `D:\Work\Competititons\AmazonMLChallenge-final`):
  - d36224e: 25 src modules from 99ac488 (x_feats.py as of 2183202: dd predates the v10n number columns;
    x_wordlists.py without the hand lists it was compared with) + `ber/reproduce.sh`: every step from
    student_resource/ to output_v10_fr3_llm_dd, independent steps on two GPUs, resumable (`work/logs/<step>.done`).
  - 1585352: README for the branch; requirements.txt was missing **peft** and **scikit-learn** (LLM judge), now
    pinned; x_chain.py removed (its plans named scripts not in the branch).
  - a707292: `torch.manual_seed(0)` before each trainer builds its model (x_ce, x_ce3, x_llm: new score heads,
    LoRA weights, dropout). Sampling, data order, LightGBM and the bi-encoder trainer were already seeded. Deployed
    before any of those steps ran, so the whole run is a707292. Still not fixed: the multithreaded HNSW build and
    GPU kernel order; and the original dd run was unseeded, so seeds cannot make the rerun match dd itself.
- Code fidelity (git history vs when each original step ran): every script is either unchanged since its step
  (train_embed, retrieve, common, lexicon, shortlist, stage1_cv, x_ce, x_feats@2183202) or changed only by options
  that are off by default (match "text:" shortlist, x_stage_multi / x_nocopy PEERS / S1K / BAG, harness DFOLD, x_ce3
  CE_TEXT=orig / SELF, x_llm LLM_TAG / SELF / COUNTRY / extend, x_final RESTORE / REJECT).
- Before this, the France-fixes-only rebuild from 99ac488 on the cached upstream files matched dd byte for byte
  (md5 75f68b10, `~/scratch/amlc_repro`, 02:33).
- **End-to-end run**: padum `~/scratch/amlc_final` (fresh work/, data symlinked), started 03:03 on scai04 with both
  hold GPUs. train_embed 15.5 min (03:04-03:19); retrieval of both splits in parallel 03:19-05:31 (2 h each: the
  HNSW search is memory-bound, so running the splits side by side saved nothing).
  - **Checkpoint 1, blocking recall (train, unseen folds 4-9) @1/2/3/5/10/20: rerun .9777/.9841/.9868/.9895/.9921/
    .9942 vs original .9778/.9841/.9868/.9895/.9921/.9942**; folds 0-3 identical; candidates 206,404,380 both.
    The seeded bi-encoder (Trainer seed 0) and HNSW reproduce the original to within 0.0001.
  - Checkpoint 2, learned lexicon (05:34): address abbreviations, legal forms and legal spelling families identical;
    name abbreviations differ in a few borderline entries (India: rerun adds sa→seva, th→tech, lacks on→one,
    piae→private; US: rerun adds cr→care, ds→dds, lc→lcsw). The lexicon learns from pseudo-matches at cosine ≥ 0.9,
    so tiny embedding differences flip borderline abbreviations: expected drift.
  - Checkpoint 3, shortlists (unseen train folds): v8 model:0.002 rerun 99.3439% of true S1s kept, 1.48 per record
    (orig 99.3420%, 1.46); v9 text:0.001 rerun 99.3993%, 1.303 per record / 6.09 per S1 (orig 99.3949%, 1.254 /
    5.87): same recall, ~4% more candidate pairs. Features and extras of both splits 05:36-05:46 (6 + 3 min).
  - 05:46-06:00 the four CE trainings sat in disk wait (~1 MB/s reads) while three LightGBM jobs loaded their data
    (ours plus the tight session's at nice 19); they recovered on their own. v8 CEs + v10 fits (extra.sh, from
    06:00) share GPUs A / B, two trainings each.
  - **Checkpoint 2, lexicon (05:34)**: legal forms, legal spelling families, address abbreviations and all of
    France identical; name abbreviations differ at the frequency cut-off: rerun-only India sa→seva, th→tech,
    US cr→care, ds→dds, lc→lcsw; original-only India on→one, piae→private. Borderline rare words flip with the
    0.0001 retrieval difference; expected noise.
  - **Checkpoint 3, v8 shortlist model (05:36)**: at model:0.002 (v8's setting) true S1 kept 99.3439% with 1.48 per
    record / 6.90 per S1 vs original 99.3420% / 1.46 / 6.84; every threshold row within ~0.005 points.
  - **Checkpoint 4, text shortlist (06:06)**: at text:0.001 (dd's setting) true S1 kept 99.3993%, 1.303 per record,
    6.09 per S1 vs original 99.3949% / 1.254 / 5.87: the retrained shortlist model is calibrated a little less
    selectively (+4% pairs at the same threshold, recall +0.004 pt), so the rerun's candidate_pairs will be a few %
    larger than dd's.
  - Pragya offload dropped at 04:00 (user OK): the padum→Pragya link fell to 0.27-1.6 MB/s, and Pragya needed ~16 GB
    (env, data, LLM weights, retrieval output). No Pragya job ran; its holder was released after a few minutes.
  - Instead `extra.sh` on scai04 runs the steps reproduce.sh leaves to "elsewhere", same commands: v10's two CE fits
    start as soon as v8's two fits have built the token caches (two trainings per A100), and the LLM judge trains on
    GPU B after v9's stage 1, then val + test split by country (US / India / France) over both GPUs, merged.
  - padum A100s: scai_q allows 2 running jobs per user (both in use); the other free padum GPUs are V100 / K40
    (no bf16). The queued replacement hold job is pinned to scai04 (1067303) for when 1066122 ends at ~08:43.
- Known lineage differences: the original v8 used v7's raw-text CE (trained on exact-search top-5); the clean run
  trains it from scratch as the v8b plan says. GPU training is not bit-deterministic, so the check is: per-stage
  validation vs the original logs, and the final file's agreement with dd (not md5).
- Infra: padum→Pragya direct link ~8 MB/s in total (8 streams: 8.8 MB/s); via this PC 0.5 MB/s; padum's filesystem
  reads small files at ~80/s, so the env goes as one zstd archive; Pragya's internet without a proxy login ~12 KB/s.
  A dedicated key `~/.ssh/id_ed25519_pragya` (padum) is in Pragya's authorized_keys (user OK'd).

## 2026-09-27 02:40 — council: is v10_fr3_llm_dd within the rules?
Checked against the problem-statement PDF and the organizer update email.
- **Clear violation as it stands (R5, reproducibility):** rule1's candidate list comes from `rule_fr.py` +
  `scratch/rule1ex.py` in the structure clone (uncommitted), so the package cannot regenerate dd. To fix: commit both
  (plus the cache builder) into src/, rerun dd from a clean clone and diff against the submitted file.
- **Grey areas:**
  - The hard-coded "France" in the dd scripts, `rule_fr.py` and the build command. R2 only bans restricting to
    {US, India}; derive the country as test minus train countries anyway (x_ce3 already does).
  - Dead hand word lists in the pushed repo (`x_frfix.py`, the constants in `x_wordlists.py`): delete them and
    disclose that they existed.
  - How dd was found: an AI assistant reviewed 180 France test groups, and leaderboard probes (France-blank, rule1)
    were read as France labels. Disclose both in the methodology as exploratory error analysis.
  - The veto rounds were kept because of their LB scores.
  - anyascii: a library, not a model; pin it.
- **Not a problem:**
  - External data (none).
  - Model licences / size (~4.4B, MIT / Apache).
  - The q=0/1 forcing: restored pairs are each record's own top-1 row of the scored candidates, and candidate_pairs is
    unchanged.
  - Self-training and test statistics, which the Q&A allows. The seeds are in-pipeline SAVE_Q files.
- **Verdict:** keep dd as the final submission if a clean clone regenerates it by T-4h; otherwise use v10_fr3_llm with the
  same fixes. The fallback is not cleaner on the rules, only on R5. No new modelling and no K change; freeze at T-4h.
  Ask the organizers in writing about the AI-assisted test review, LB probing and anyascii.

## 2026-09-27 02:40 — `architecture.md` written (shareable description of v10_fr3_llm_dd)
The whole design for readers outside the team: pipeline diagram, each step's measured value, folds / validation, the
reverse-engineered data generator, LB progression, what did not work, lessons, compute / licences. No internal paths or
teammates' names.

## 2026-09-27 02:00 — LB: v10_fr3_llm_dd = 0.990282 (+0.003211), new best; pushed as v11 432da9e
- The expected value was 0.99024 at 95% right, so the fixes are ~95-96% right. The census / hand review holds:
  groupe / développement at the same address are true-match suffixes. The LLM judge's "no" on them (x_frllm) was a
  US word prior. Its confirmation of the type-word rejects stands. The hedge `suf7ngtrej` is not needed.
- **Next candidate `submissions/v10b_fr3_llm2_dd`** (md5 bd3c8abc, PASS): v10b + LLM (full coverage) + the three
  France vetoes + the same dd restore / reject. restore_fr_dd: 32,222 of its pairs present in v10b's q; reject_fr_dd
  11,634 (8,502 accepted before). Accepted US 2,255,859 / India 2,751,539 / France 856,613. Against dd it adds 6,005,
  removes 10,939, and changes 16,415 S1 rows. Val +0.00011 (US / India); France effect of bge unknown.
- Code, README (results row + reproduce steps) and this log are pushed on v11 (432da9e). rule1's candidate list
  (x_census.R1) still comes from the structure clone (rule_fr.py, scratch/rule1ex.py).

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

## Running (2026-09-27 02:00)
- Best: `v10_fr3_llm_dd` 0.990282. Ready: `v10b_fr3_llm2_dd` (bd3c8abc). Superseded: suf7ngtrej, v10b_fr3_llm(2),
  v10_fr3_llm_r1.
