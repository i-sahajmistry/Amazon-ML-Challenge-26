# Review of `ber/Documentation_template.md` + `ber/pipeline.png` (final-dd d7db51d, E16fr3 = LB 0.990807)

Checked against the code on `final-dd` d7db51d, the E16fr3 build logs on padum (`~aib262144/amlc_team/build4.log`,
`rescue_val.log`, `AmazonMLChallenge/work/logs/x_llmstack__v10p__v10bp.log`) and our experiments on `sarvesh-exp`
(`sarvesh_exp/RESULTS.md`, E1–E36). Sarvesh, 27 Sep.

## Flags (fix before packaging)

| # | Severity | Where | Problem | Suggested fix |
|---|---|---|---|---|
| 1 | **High (render)** | §4, component table, the line before `\| Recall fixes \|` (line 172) | A blank line inside the table: the "Recall fixes" row is cut off and renders as raw `\|` text, not a table row. | Delete the blank line so the row joins the table. |
| 2 | **Medium (wrong number)** | Appendix B, paragraph under the progression table | "From v9p to the submitted file the leaderboard rose by **+0.0077**". Stale: 0.990807 − 0.98269 = **+0.0081** (+0.0077 was v9p → variant 0.99035). | Change to +0.0081. |
| 3 | **Medium (inaccurate)** | §4, "Threshold selection method" | "an expected-F0.5 decision per S1 did worse than the plain threshold". Our E20 (`sarvesh_exp/exp12.py`), cross-fitted on US / India validation: **+0.00004 vs the plain 0.70 threshold**, −0.00005 vs threshold **+ empty-S1 rule**. | "... did worse than the threshold plus the empty-S1 rule." |
| 4 | **Medium (misleading)** | §5, "Where validation F0.5 is lost" table and "an oracle over all three error types gives about 0.9957" | The table only counts stage-2 rows (each record's own best S1). Misses where the true S1 was **never retrieved** (0.58% of true pairs) or **another S1 won** (0.84%) are not in it; they are ~70% of all missed true pairs (our E24, `exp16.py`: recall 0.9801; not retrieved 0.58%, cut by shortlist 0.03%, another S1 best 0.84%, correct S1 but q < 0.70 0.55%). So the "oracle" is not an upper bound on the score. | Footnote: "stage-2 rows only; true pairs never retrieved (0.58%) or lost to another S1 (0.84%) are not included." |
| 5 | **Medium (claim to confirm)** | §2.2 Core innovation 2 | "Both were validated on the labelled countries before use." True for the census (TV statistics on US / India). For the **self-training AND-veto**: our E25 (`exp17.py`) found a France-style min-veto on US / India flat to negative (cross-fitted −0.00001; min(bge, main) ≥ 0.70 −0.00005). | Unless the held-out-country (v8-loco) work validated the veto, say it was validated on the leaderboard (v9p → v10_fr3 rows of Appendix B). |
| 6 | Low | §1 and Table 3.2 | "keeps **99.98%** of the true pairs that retrieval finds": Table 3.1 gives 99.39 / 99.42 = **99.97%** (our E24: 0.027% of true pairs cut by the shortlist). | 99.97%. |
| 7 | Low | Table 3.2 row "P ≥ 0.001 (used)" | 7.36 pairs per S1 vs 7.37 in §1 / §3 / the figure (12,760,925 / 1,732,544 = 7.365). | 7.37 everywhere. |
| 8 | Low | §4 "Recall fixes" row | t = 0.40 is chosen and reported on the same validation (`rescue_val.log`: bge 0.4 +0.00007, 0.5 +0.00006). Our split-half cross-fit (E13a / E15) picked 0.5 in both folds; 0.4–0.65 all positive. | Optional: "t in 0.4–0.5 all positive on validation; 0.40 used". |
| 9 | FYI | Appendix B "Tried and dropped", smaller candidate set | Dropped "for no score gain", but the problem statement ranks a smaller candidate set higher. E17 (our `out_E17`, 12.02M pairs, −5.8%) has the same validation. | Team's call; keep the sentence accurate about the trade-off. |
| 10 | FYI (figure) | pipeline.png | Every arrow runs through "Transliterate", but the bge cross-encoder reads the original text (the Cross-encoders box does say "on the text as given"). | Optional. |

## Verified correct

- Structure: every section / field of the organisers' `student_resource/Documentation_template.md`.
- Package: `package.sh` copies `pipeline.png` next to the doc; `src/` has exactly the 22 modules listed and all exist;
  `requirements.txt` includes scikit-learn (the judge blend's LogisticRegression).
- Final file: E16fr3 matching md5 `e93605ad`, candidate_pairs `1a8b4f5c` (12,760,925 pairs), validator PASS
  (`build4.log`).
- Counts: restore_empty 1,267 (US 508 + India 415 = **923**, France **344**), restore_nafr **2,508** (France); restore
  32,620 / reject 11,105.
- Validation: bge + judge **0.99294** in the exact shipped configuration (`x_llmstack _v10p _v10bp`, log above);
  0.99282 two-CE stack; + empty-S1 rule +0.00007 → **0.9930**.
- Arithmetic: France share 259,452 / 1,732,544 = 14.975%; 7.37 pairs per S1, 1.28 per S2/S3 record (9.97M records);
  within-country pairs 6.72 × 10¹² (US 2.53e12 + India 3.82e12 + France 0.37e12), reduction 99.99981%; train pairs
  7,638,365 (US 4,578,522 + India 3,059,843); 26% unmatched train records ≈ 1.22 per S1; implied France F0.5
  (0.990807 − 0.85025 × 0.9930) / 0.14975 = **0.978**; census fixes 0.99028 − 0.98707 = +0.0032.
- Code: stage 1 LightGBM 255 leaves, lr 0.05 (`stage1_cv.py`); 72 / 77 stage-1 features (45 + 17 + 5 per CE);
  judge blend = logistic regression on [logit q, LLM margin] for 0.01 < q < 0.99; veto = min of the main + 3
  self-trained stacks with the same S1; no hard-coded word lists found in `common`, `match`, `lexicon`, `x_feats`,
  `rule_fr`, `wstat`, `x_ddfix`, `x_recall` (country / legal words appear only in comments and self-tests).
- Figure: all boxes and numbers match the code (62 / 72 / 77 features, P ≥ 0.001, top-20, 5 seeds, LLM band, min of
  4 q, recall-fix thresholds and LLM gates, 0.70 decision).
