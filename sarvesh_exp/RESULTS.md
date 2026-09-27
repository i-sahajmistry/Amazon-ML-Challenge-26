# Experiments on top of the best file (Sarvesh, 27 Sep)

## TL;DR for the final build (latest first) — TRY / AVOID, with why
**Doc review:** `sarvesh_exp/DOC_REVIEW.md` — 10 flags on `ber/Documentation_template.md` / `pipeline.png` (1 render bug, 4 medium).
**Leaderboard:** the team's final file **E16fr3 = 0.990807** (6th; md5 `e93605ad`, final-dd d7db51d) = E16 (bge stack +
empty-S1 rescue) + Mohanish's x_recall.py (rescue at q ≥ 0.40 in every country incl. France with an LLM gate; France
no-address exact-unique-name restores).

TRY:
1. **E17-style smaller candidate set** (our `out_E17`, md5 `eb6fdbec`): same validation as E16 (−0.00002, noise), 5.8%
   fewer pairs (12.02M). Only if a smaller candidate file helps the ranking; the team judged E17fr2 ≈ E16fr3 on score.

SUPERSEDED: **E30 / E30b (France rescue)** — E16fr3's `restore_empty` already fills 344 empty France S1s (q ≥ 0.40,
LLM gate), which is the better-guarded version of the same idea.

AVOID — **E19 / E31 (bge stack as a 4th France veto)**: the team's hand review found E19 wrong on all 4 reviewed changes;
the ~3,650 France records it removes sit at the S1's own address and mostly add *& Fils / Groupe / Développement*,
which the census marks as true-match suffixes. (Our E23 looked at the 3 existing vetoes, not this one.)
AVOID (validated on US / India, all ≤ 0 or noise): stack averaging (E7), per-country / per-crowding thresholds (E8,
E12), lone-match drop (E13b), sibling rule (E15), stage-2 tuning bigger trees / lower lr (E18), expected-F0.5 per S1
(E20, worse than the simple rescue), second-model veto on US / India (E25), exact / fuzzy name rules for name-only
records (E26 / E28: their misses are ambiguous, not fixable), house-number rules for US / India (E23: accepted test
pairs already look like true matches). Invented-name records → match by address (E33: −0.00007; those records are
already found 97.8% of the time, the extra address matches are only 67% right). Do not chase a "France recall gap" (E21 / E22: France just has smaller clusters). Per-record-type thresholds (E34: −0.00004, every
type already calibrated). Name-ambiguity / invented-name features in stage 2 (E36: + LLM 0.99289 = control 0.99289).
Richer judge blend with record-type features, LR or LightGBM (E38: −0.00000 … −0.00004; the 2-input LR is best).
"France over-accepts decoys" (E39: France rejects 2.18 claimants per S1 vs US 2.13 / India 2.16 — no excess).

**France evidence (label-free, E37 / E39 / E40 on test):** US / India test acceptance matches train truth kind-for-kind
(e.g. other-address same-name 0.969 vs 0.970 per S1; number-shift name-edit 0.238 vs 0.238), so their models are
calibrated. France is much stricter on number-mismatched records: *same name, record number LOWER than the S1's*
accepted 35% in France vs 85% US / 89% India (validation: 73% / 81% true, accepted 89% / 93% at 99.95% precision);
*different name, lower number* 4% vs 51% / 41%; *same name, number up > 13* 21% vs 50% / 87%. Caveat: France has fewer
such rows per S1 (0.18 vs 0.41), so part of this may be real. Decoys shift UP 1–13 and come in groups at one
number. **RESULT (E41 / E42): France is RIGHT to reject them — AVOID.** On US / India validation the REJECTED records of
these buckets are 0.2–2% true (down same-name alone 0.65%, with peers 2.4%; up > 13 same-name 0.2%); the France ones
are namesake businesses on other streets (`Bordeaux Amicale EURL` 46 Rue de Solférino vs 6 R. Sullivan). Files built
for the test are renamed `DO_NOT_UPLOAD_E16fr3_F1/F2` (+7,244 / +15,469 France pairs).

**E42 — the France veto chain on labelled data (`exp31.py`):** min(main, a1, a2, a3) on US / India validation scores
**0.99262 → 0.99248 (−0.00014)**: it removes ~1,600 true matches for ~400 false. The vetoes helped France on the LB
(v9p → v10_fr3), so they correct something France-specific; they are not label-validated (see DOC_REVIEW flag 5).

## FOR THE FINAL BUILD — read this first (kept current)
Written for whoever builds the final file (team or their assistant). Baseline = `variant_v10seed_dd`, LB 0.990349,
md5 e73c409e. Each item: what, evidence, how to apply, decision rule. Details and all numbers further down.

1. **APPLY — US / India from the bge stack (E1).** Evidence: labelled US / India validation at 0.70, bge + LLM stack
   **0.99294** vs the variant's main + LLM stack **0.99282** (+0.00012; Sahaj's `x_llmstack__v10bp__v10bp.log`); better
   in both countries. France untouched. Expected LB ≈ +0.0001.
   How: base q = the bge + LLM stack instead of the main + LLM stack for countries with labels; France keeps the
   variant's chain. With the variant's files (as run here, generic x_final of final-dd HEAD):
   `FROM=unlabelled:_v10plw,unlabelled:_a1pw:min,unlabelled:_a2pw:min,unlabelled:_a3pw:min RESTORE=restore_fr_dd REJECT=reject_fr_dd python x_final.py OUT _v10bplw 0.70`
   → md5 ab07d90e, validator --check-ids PASS, candidate_pairs unchanged (1a8b4f5c). In reproduce.sh this needs the
   v10b steps (bge-reranker-v2-m3, Apache-2.0, 568M: fit + score + stages + llm_stack on v10b). Compliance: fine.
2. **DO NOT APPLY — averaging the main and bge stacks (E7).** Validation 0.99279 < bge alone 0.99283.
3. **DO NOT CHANGE the 0.70 threshold for US / India.** Validation flat 0.60–0.70 (bge + LLM best at 0.70).
4. **OPTIONAL, LEADERBOARD-ONLY — France changes (no labels, cannot be validated):** E2 (bge as a 4th France veto,
   −3,656 France matches), E3 (France base from bge + LLM round 2, −2,500), E4a / E4b (France threshold 0.65 / 0.75,
   +1,446 / −1,960). Use only if a spare upload is available; past France vetoes helped, threshold moves are a guess.
5. **COMPLIANCE ✅ — the generic France fixes reproduce the best file exactly (E5).** Mohanish's data-only
   `rule_fr.py` + `x_ddfix.py` (final-dd HEAD, `unlabelled`, no country names, no hand lists), run on the variant's own
   scores, give restore / reject lists **pair-for-pair identical** to the variant's (32,620 / 32,620 and 11,105 /
   11,105, zero differences), and the file is **byte-identical** to E1 (md5 ab07d90e); E0 is byte-identical to the
   0.990349 file. So the final code can use the generic steps (`x_ddfix` + `unlabelled`) with no change in output.
6. **DO NOT APPLY — per-country thresholds for US / India (E8).** Best cut on validation gains +0.00001 only (US best
   0.625 0.99223 = 0.70 0.99223; India best 0.675 0.99375 vs 0.70 0.99374). Keep 0.70 for every country.
7. **OPTIONAL, LEADERBOARD-ONLY — E9: France base from the bge + LLM stack** (same 3 vetoes and fixes): France 857,002
   (−2,336), md5 ae0ecb86, PASS.
8. **CORRECTED — E11 does NOT explain the US / India validation-vs-leaderboard gap (E12).** E11 suggested test has 3×
   more sparse S1s (1–4 claimants) where the model is weak. That was an **artifact**: E11 counted validation distractors
   with their ×3 weight. Counting the same way on both sides (strong claimants, q ≥ 0.5, unweighted), test's mix
   matches validation: US+India test-like 0.99283 = validation 0.99283 (US 0.99225 vs 0.99223, India 0.99371 vs
   0.99374). The gap's cause is unknown (and the ~0.9915 LB estimate for US / India is itself soft).
   **DO NOT APPLY — threshold per crowding level:** cross-fitted gain +0.00005 / +0.00010 (mean +0.00008), and the
   best thresholds flip between folds for 3 of 7 groups: noise-level.
9. **APPLY (small, validated) — rescue empty S1s (E13a / E15).** For countries with labels, an S1 with no accepted
   record takes its best claimant if that record's q ≥ 0.5. Cross-fitted on validation (fit one entity fold, score
   the other): **+0.00008 and +0.00010, mean +0.00009, both folds pick t = 0.5** (every t in 0.4–0.65 positive).
   Stacks with E1: together ≈ +0.0002 on US / India validation. (E15 re-check of E13a with the same code: +0.00009.)
   **Ready-made file = E16** (E1 + this rescue, `exp9.py`): 666 S1s rescued (US 369, India 297, one record each),
   US / India / France accepted 2,256,168 / 2,751,925 / 859,338, matching md5 **`a019f175`**, candidate_pairs
   `1a8b4f5c` (unchanged), validator `--check-ids` **PASS**. France byte-identical to the variant.
   **This is the best validated file we have.** Caveat: E1 (+0.00012) and the rescue (+0.00009) were each measured
   alone; the rescue was measured on the stage-2 q (before the LLM blend), and the combination was not scored
   jointly (no validation q with the LLM blend is saved), so "+0.0002" is an estimate, possibly less if they overlap.
   US + India are 85% of S1s → expected LB ≈ +0.00015–0.0002 if validation transfers.
10. **DO NOT APPLY — drop a lone medium-confidence match (E13b):** negative for every t (−0.00003 … −0.00064).
11. **WHERE US / INDIA LOSE (E14, validation 0.99283, loss 0.00717):** missed some true matches 0.00470 (66%), S1 left
    empty although it has matches 0.00152 (21%), distractor merged 0.00045, wrong-S1 record merged 0.00041,
    singleton given a record 0.00009. **~87% of the remaining loss is recall**, so recall rules are the lever (E15
    running: accept "siblings" just under 0.70 when the S1 already has a very confident match).
12. **DO NOT APPLY — "sibling" rule (E15):** accepting records with t ≤ q < 0.70 when their S1 already has an
    accepted record with q ≥ 0.9–0.99 is negative for every setting (−0.00002 … −0.00118); combined with the rescue it
    is worse than the rescue alone (+0.00001). The missed matches are genuinely ambiguous.
13. **OPTIONAL (blocking) — smaller candidate set for US / India costs nothing measurable (E10).** Shortlist cut-off
    for US / India raised 0.001 → 0.005 (France keeps all its pairs), whole chain retrained (stage 1 → stage 2 → LLM
    blend on the bge stack). Test candidate_pairs **12,760,925 → 12,020,996 (−739,929, −5.8%)**; train −7.2%.
    US / India validation at 0.70, same code both sides: control **0.99289** vs smaller set **0.99287** (+LLM;
    stage 2 alone 0.99274 vs 0.99277). The difference (−0.00002) is below rerun noise (this control gives 0.99289 vs
    Sahaj's 0.99294 for the same setup). **Use it only if the smaller candidate file matters for ranking**; it
    needs the US / India test q from the smaller run plus the rescue re-applied. **Ready-made file = E17**
    (`exp10.py`): E16 on the smaller candidate set. candidate_pairs **12,020,996** (md5 `15fb51cc`) vs 12,760,925;
    720 S1s rescued (US 375, India 345); accepted US / India / France 2,256,386 / 2,751,391 / 859,338; matching md5
    **`eb6fdbec`**; validator `--check-ids` **PASS**; France byte-identical to the variant. vs E16: 99.6% of S1 rows
    identical (the retrain moves ~3.5k pairs each way, F0.5 0.9992 between them) — same validation.
    **Choose E17 over E16 if the smaller candidate file counts; otherwise they are equivalent.**
14. **Stage-2 tuning (E18, `s2tune.pbs`)** on the E10 control's stage 1 (control + LLM 0.70 = 0.99289):
    - **DO NOT APPLY — `big`** (127 leaves, min_data_in_leaf 200, BAG 5): stage 2 0.70 0.99276 vs 0.99274, + LLM
      0.70 **0.99289 = control 0.99289**; logloss worse (0.00817–0.00819 vs 0.00812–0.00815). No gain.
    - DO NOT APPLY — `lr` (learning rate 0.025, BAG 10): stage 2 0.99276 vs 0.99274 (item 19).
15. **LEADERBOARD BET — E19 = E17 + E2** (`exp11.pbs`, `amlc_exp/out_E19/`): US / India as E17, France as E2 (bge +
    LLM stack as a 4th France min-veto). Checked: US / India pair-for-pair identical to E17, France identical to E2
    (855,682, −3,656 vs the variant). matching md5 **`c91f83b1`**, candidate_pairs `15fb51cc` (12.02M), validator
    `--check-ids` **PASS**. Final ranking = private score of the best public submission, so a worse upload costs
    only the slot. Top 3 is 0.99117 (+0.00082 over us); validated changes give ≈ +0.0002, so a France change is the
    only lever big enough to matter.
16. **DO NOT APPLY — expected-F0.5 decision per S1 (E20, `exp12.py`):** for each S1, accept the top k claimants that
    maximise the expected F0.5 (Monte Carlo on q, calibration sigmoid(a·logit q + b) cross-fitted). Cross-fitted
    **+0.00004 vs 0.70 but −0.00005 vs 0.70 + rescue**; the simple rescue (item 9) is better. Does not work.
17. **E21 / E22 — France "shape" check (`exp13.py`, `exp14.py`): NO France recall gap found; do not chase it.**
    - Train truth is identical for US and India (empty S1s 5.58% / 5.59%, matches per S1 3.459 / 3.465, 4+ 48%), and
      US / India predictions look the same on validation and test (empty 5.75–5.8%, 3.40 per S1): test behaves like
      validation.
    - France predicted: empty 6.16%, **3.31 per S1** (vs 3.40). It looked like ~23k missing France matches, BUT France
      has fewer records per S1 in test (1,434,993 / 259,452 = **5.53** vs US 5.76, India 5.82), and the share of
      source-2/3 records we accept is the same or higher in France (**0.599** vs US 0.591, India 0.583). With the same
      distractor share in every country, France's true clusters are simply smaller (~3.32 per S1), so France's
      matches-per-S1 gap is explained by the data, not by lost recall.
    - France rescue: only 312 S1s at t = 0.5 (14.7k of 16k empty France S1s have a best claimant below q 0.1).
      Lowering the France threshold 0.70 → 0.50 adds only 0.02 per S1.
    - France chain (accepted records): base 863,509 → after the 3 min-vetoes 837,823 (−25.7k) → + restore 870,443 →
      − reject **859,338**. 31,331 records with base q ≥ 0.70 are vetoed, 12,586 of them with base q ≥ 0.99, mostly in
      S1s that keep 2–5+ matches. Whether those vetoes are right can only be judged on the leaderboard.
18. **E23 — Occam decoy check (`exp15.py`): test accepted pairs look exactly like train true matches; no hidden decoy
    leak.** Share of pairs whose first house numbers disagree: train true matches US 11.70% / India 7.25%; **test
    accepted US 12.09% / India 7.35%** (+0.4 / +0.1 pt); train distractors 95% / 57%. Same for same-name,
    same-address, same-name + number-conflict (US 6.63% vs 6.35%). On validation the number conflict does not separate
    wrong accepted pairs (9.3%) from right ones (10.0%). So a number-conflict rule cannot help US / India.
    France: accepted 3.98% conflict; the 31,331 vetoed records **24.7%** (same name + conflict 15.4% vs 2.9%): the
    vetoes mostly remove decoy-looking pairs, consistent with them helping on the leaderboard (supports E2 / E19).
19. **DO NOT APPLY — stage-2 `lr` (0.025, BAG 10):** stage 2 0.70 0.99276 vs control 0.99274 (noise); stopped before
    the LLM blend.
20. **E24 — where US / India validation loses true matches (`exp16.py`, 1,535,184 true pairs, recall 0.9801):**
    not retrieved by FAISS top-20 **0.58%**, cut by the shortlist 0.03%, another S1 won (stage 1/2) **0.84%**
    (0.56% + 0.28%), correct S1 but q < 0.5 0.38%, correct S1 with 0.5 ≤ q < 0.70 0.17%. 97.5% of missed pairs are in
    S1s that have other accepted records. **In the examples almost every missed record has an EMPTY address (name
    only)**, e.g. S1 `cure pub @ 1094 ohio river road` vs record `cure pub` with no address → not retrieved.
21. **DO NOT APPLY — second-model veto on US / India (E25, `exp17.py`):** requiring the main stack to agree
    (France-style min) is flat to negative: cross-fitted −0.000013; min(bge, main) ≥ 0.70 −0.000053.
22. **E26 — name-only records (`exp18.py`): they are 83% of US / India misses, but mostly unrecoverable.**
    Records with an empty address are 4.4% of true pairs with recall **0.62** (with address: 0.9965); they make 1.66%
    of the 1.99% missed pairs. AVOID the exact-name rule (empty-address record → the only S1 with the same core name):
    99.1% precise, but 22,203 of its 22,542 pairs are already accepted, validation −0.000009 (test file E27a adds 2,867,
    not recommended). The missed ones share their name with several S1s (26%) or have typos (41%).
23. **AVOID — fuzzy name channel for missed name-only records (E28, `exp20.py`):** char-trigram TF-IDF on names, the
    top S1 is right only 22% of the time (names shared by several S1s); best cross-fitted gain +0.000009 (88 pairs).
24. **No data leak (E29, `exp21.py`):** ids are random, rows are shuffled (neighbouring rows never share an S1).


## Ready-made files on padum (read access granted to aib262144 via setfacl, incl. files written later)
| What | Path (under `/scratch/scai/mtech/aib262045/`) |
|---|---|
| Submission-ready outputs (candidate_pairs.tsv + matching_results.tsv) | `amlc_exp/out_E17/` (best validated), `amlc_exp/out_E19/` (E17 + E2), `amlc_exp/out_E16/`, `out_E1` … `out_E9` |
| E17 inputs: US / India bge + LLM q on the smaller candidate set, rescue applied | `amlc_e10_0.005/work/x/test_q_v10bplw.parquet` (before rescue), `test_q_resc17.parquet` (after), `p5_test_v10b.parquet` (= the 12.02M candidates) |
| Smaller-set sandbox (feats2, CE arrays, extras cut at shortlist P ≥ 0.005) | `amlc_e10_0.005/work/` |
| Control rerun (all pairs) and stage-2 tuning runs | `amlc_e10_0.001/`, `amlc_s2_lr/`, `amlc_s2_big/` |
| Job scripts + logs exactly as run | `exp/` (`exp10.pbs` = E17, `exp11.pbs` = E19, `e10_run.pbs`, `s2tune.pbs`) |
Every TSV pair we generated (`amlc_exp/out_<name>/{matching_results,candidate_pairs}.tsv`, all readable by aib262144,
all pass `validate_submission.py --check-ids`; not in git: 94 MB + 170–179 MB each):

| out dir | what | matching md5 | candidate md5 | use? |
|---|---|---|---|---|
| out_E17 | E1 + rescue on the smaller candidate set | `eb6fdbec` | `15fb51cc` (12.02M) | ✅ **best validated** |
| out_E19 | E17 + E2 (4th France veto) | `c91f83b1` | `15fb51cc` | ❓ LB bet |
| out_E16 | E1 + rescue, full candidate set | `a019f175` | `1a8b4f5c` (12.76M) | ✅ (= E17 without the smaller set) |
| out_E1 / out_E5 | bge stack for US / India (E5 = generic France fixes, identical) | `ab07d90e` | `1a8b4f5c` | ✅ superseded by E16 / E17 |
| out_E0 | the 0.990349 file rebuilt (sanity) | `e73c409e` | `1a8b4f5c` | reference |
| out_E1b | bge + LLM round 2 | `34173047` | `1a8b4f5c` | ≈ E1 |
| out_E2 / E3 / E4a / E4b / E9 | France-only variants on E1 | `7a35ceb9` / `f8466461` / `a95424e3` / `010c7d23` / `ae0ecb86` | `1a8b4f5c` | ❓ LB only |
| out_E7 | mean of stacks | `138c24f2` | `1a8b4f5c` | ❌ do not use |

To rebuild E17 in place: `AMLC_ROOT=/scratch/scai/mtech/aib262045/amlc_e10_0.005` + the `x_final.py` line in `exp10.pbs`
(write OUT to your own dir; the FROM / RESTORE / REJECT files are symlinks to your variant files).

Not part of the submission package (this folder is outside `ber/`). Scripts here are exactly what ran on padum.

## Baseline
**`variant_v10seed_dd`, leaderboard 0.990349** (Sahaj; `~aib262144/amlc_variant/output_variant_dd/`, matching_results md5
`e73c409e`, candidate_pairs `1a8b4f5c`). US / India validation 0.99282.
Accepted matches: US 2,255,158, India 2,751,010, France 859,338.

All experiments are **CPU assembly** (`x_final.py` of `final-dd` HEAD, generic: `unlabelled:` = countries without
training labels) from Sahaj's finished q files, read-only. Every file keeps candidate_pairs `1a8b4f5c` (same 12.76M
pairs) and passes the validator with `--check-ids`. Outputs: padum `/scratch/scai/mtech/aib262045/amlc_exp/out_<name>/`.

**How to judge:** US / India have labels, so a US / India change has a validation score. France has none, so a France
change can only be judged on the leaderboard; for those we report how many France matches change.

## Results

| # | What changes (vs baseline) | US / India / France accepted | matching md5 | Evidence | Verdict |
|---|---|---|---|---|---|
| E0 | nothing: the variant rebuilt from its files with the generic x_final | 2,255,158 / 2,751,010 / 859,338 | `e73c409e` | byte-identical to the 0.990349 file | ✅ setup correct; generic assembly reproduces the best file exactly |
| E1 | US / India from the bge stack (v10b + LLM, `_v10bplw`); France unchanged | 2,255,799 / 2,751,628 / 859,338 | `ab07d90e` | US/India validation **0.99294 vs 0.99282** (bge + LLM stack vs the variant's LLM stack, Sahaj's llm_stack logs) | ✅ **+0.00012 validation**, ≈ +0.0001 LB |
| E1b | as E1 with the bge + LLM round-2 stack (`_v10bpl2w`) | 2,255,859 / 2,751,539 / 859,338 | `34173047` | ≈ E1 | ≈ E1 |
| E2 | E1 + the bge stack as a 4th France veto | … / … / 855,682 (−3,656) | `7a35ceb9` | none (France) | ❓ LB only; vetoes have helped before |
| E3 | E1, France base from bge + LLM round 2 | … / … / 856,838 (−2,500) | `f8466461` | none (France) | ❓ LB only |
| E4a | E1, France threshold 0.65 (instead of 0.70) | … / … / 860,784 (+1,446) | `a95424e3` | none (France); US/India validation is flat 0.60–0.70 | ❓ LB only |
| E4b | E1, France threshold 0.75 | … / … / 857,378 (−1,960) | `010c7d23` | none (France) | ❓ LB only |
| E5 | E1 with the generic France fixes (`rule_fr` + `x_ddfix`, `unlabelled`) instead of the variant's lists | 2,255,799 / 2,751,628 / 859,338 | `ab07d90e` (= E1) | lists pair-for-pair identical to the variant's (restore 32,620, reject 11,105) | ✅ **compliance: generic code gives the identical file** |
| E8 | per-country thresholds (US / India validation, bge stack) | — | — | +0.00001 at best (US 0.625, India 0.675) | ❌ **does not work** |
| E9 | France base from the bge + LLM stack, same vetoes + fixes | … / … / 857,002 (−2,336) | `ae0ecb86` | none (France) | ❓ LB only |
| E16 | E1 + rescue empty S1s at q ≥ 0.5 (US / India) | 2,256,168 / 2,751,925 / 859,338 | `a019f175` | validation E1 +0.00012, rescue +0.00009 cross-fitted; 666 S1s rescued | ✅ **best validated file**, PASS |
| E10 | US / India shortlist cut-off 0.005 (−5.8% test pairs), full retrain | (see E17) | — | validation +LLM 0.99287 vs control 0.99289 | ✅ no measurable cost; optional, for a smaller candidate file |
| E17 | E16 on the E10 smaller candidate set (US / India P ≥ 0.005, retrained) | 2,256,386 / 2,751,391 / 859,338 | `eb6fdbec` | candidates 12.02M (−5.8%, `15fb51cc`); validation as E16 | ✅ **best validated file with a smaller candidate set**, PASS |
| E19 | E17 + E2 (bge as a 4th France veto) | 2,256,386 / 2,751,391 / 855,682 | `c91f83b1` | US / India = E17; France = E2 (no labels) | ❓ LB bet; PASS |
| E7 | US / India from the **mean** of the bge and main stacks | 2,255,059 / 2,751,092 / 859,338 | | validation 0.99279 < bge alone 0.99283 | ❌ **does not work**: averaging is worse than bge alone (file `138c24f2` built, PASS, not to be used) |

## Per-country comparison against the variant (0.990349): what each file changes
From the team's `compare.py` (pairs = accepted record → S1 links).

| File | US rows identical | India rows identical | France rows identical | France pairs added / removed | Kind of change |
|---|---|---|---|---|---|
| E1 (bge US/India) | 99.24% | 99.42% | **100%** | 0 / 0 | US/India only (validated +0.00012) |
| E2 (+ bge 4th France veto) | = E1 | = E1 | 98.61% | **+0 / −3,656** | France precision only |
| E3 (France base bge + LLM2) | = E1 | = E1 | 97.87% | +1,584 / −4,084 | France mixed |
| E4a (France thr 0.65) | = E1 | = E1 | 99.45% | **+1,446 / −0** | France recall only |
| E4b (France thr 0.75) | = E1 | = E1 | 99.25% | **+0 / −1,960** | France precision only |
| E5 (generic fixes) | = E1 | = E1 | 100% | 0 / 0 | identical to E1 |

F0.5 break-even for a France removal: it helps if the removed pairs are more than ~30% wrong (precision-weighted metric);
past France vetoes removed mostly decoys. Additions help only if they are more than ~70% right.

## E11 (SUPERSEDED by E12 — artifact of weighted counting): validation vs test, by how many records claim each S1 (`exp5.py`)
Claimants = records whose best S1 it is (label-free, exists on test). Validation counts distractors with their weight.

| claimants | US val / test share | US val F0.5 | India val / test share | India val F0.5 |
|---|---|---|---|---|
| 1 | 0.0003 / 0.0090 | 0.97222 | 0.0028 / 0.0102 | 0.97032 |
| 2 | 0.0010 / 0.0324 | 0.97725 | 0.0102 / 0.0329 | 0.98932 |
| 3 | 0.0160 / 0.0794 | 0.93624 | 0.0307 / 0.0784 | 0.97526 |
| 4 | 0.0503 / 0.1557 | 0.97794 | 0.0564 / 0.1494 | 0.98808 |
| 5 | 0.1597 / 0.2186 | 0.99111 | 0.1528 / 0.2073 | 0.99293 |
| 6 | 0.2674 / 0.2106 | 0.99428 | 0.2451 / 0.2058 | 0.99530 |
| 7 | 0.2108 / 0.1523 | 0.99472 | 0.1934 / 0.1546 | 0.99570 |
| 8+ | 0.2944 / 0.1420 | 0.99473 | 0.3070 / 0.1609 | 0.99509 |
| **all** | validation **0.99223** → test-like **0.98581** | | validation **0.99374** → test-like **0.99171** | |

Best threshold per group (US+India validation): 2 claimants 0.5 (+0.00132), 3: 0.5 (+0.00008), 4: 0.5 (+0.00157),
5: 0.6 (+0.00012), 6–7: 0.7 (0), 8+: 0.8 (+0.00010). In-sample; E12 cross-fits it.

## US / India validation (labels; entity folds 8-9, distractors weighted, copy-free), `exp3.py`

| Stack (before the LLM blend) | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | US @0.70 | India @0.70 |
|---|---|---|---|---|---|---|---|
| main v10 (`_v10pw`) | 0.99259 | 0.99263 | 0.99262 | 0.99258 | 0.99248 | 0.99204 | 0.99349 |
| **bge v10b (`_v10bpw`)** | 0.99283 | 0.99283 | **0.99283** | 0.99279 | 0.99272 | 0.99223 | 0.99374 |
| mean of both | 0.99279 | 0.99280 | 0.99279 | 0.99273 | 0.99264 | 0.99218 | 0.99370 |

- ✅ bge is better than the main stack by **+0.00021**, in both the US and India → E1.
- ❌ averaging the two stacks does not help (below bge alone) → E7 dropped.
- Threshold: 0.70 stays (bge flat 0.60–0.70, lower above).
- With the LLM blend (Sahaj's `x_llmstack__v10bp__v10bp.log`): bge + LLM **0.99294** at 0.70 (0.60 0.99289, 0.65 0.99291,
  0.75 0.99290, 0.80 0.99282); the variant's main + LLM stack 0.99282. So E1's US/India gain is +0.00012.

## E10: smaller candidate set for the countries with labels (`cand_p.py`, `e10_prep.py`, `e10_run.pbs`)
The organisers rank a smaller candidate_pairs.tsv higher, and it must be exactly what the first model scored, so a
smaller set means re-scoring. France keeps its list (Sahaj's `tight_dd` lost France matches); US / India shortlist
cut-off raised to P ≥ 0.005; stage 1 (bge stack, S1K=3) → stage 2 (BAG=5, peers) → LLM blend rerun on the smaller
set. Control = the same run at P ≥ 0.001 (all pairs), so both sides share the code and the randomness.

| | train pairs | test pairs | stage 2, 0.70 | + LLM, 0.65 | + LLM, 0.70 | + LLM, 0.75 |
|---|---|---|---|---|---|---|
| control (P ≥ 0.001) | 12,929,263 | 12,760,925 | 0.99274 | 0.99287 | **0.99289** | 0.99285 |
| smaller (P ≥ 0.005) | 12,003,540 (−7.2%) | 12,020,996 (−5.8%) | 0.99277 | 0.99288 | **0.99287** | 0.99282 |

Verdict: −739,929 test pairs for −0.00002 validation at 0.70, i.e. no measurable cost. Runtime ≈ 35 min per run on 1 GPU.

## Earlier: candidate-set size (`cand_trim.py`, on the v9 shortlist = the same 12.76M pairs)
Raising the shortlist cut-off from P ≥ 0.001: 0.003 → −8% pairs, cuts 782 of 5.88M accepted matches; 0.005 → −11%,
1,421; France 10.1 → 8.2 / 7.5 pairs per S1. Needs a full rerun from stage 1 (candidate_pairs must be what the model
scored). Sahaj's `tight_dd` (6.27 per S1) lost France matches, so a tighter shortlist is not used.
