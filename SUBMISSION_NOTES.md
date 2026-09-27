# Final submission notes

What to do and remember before the final submission. Add items as they come up; tick them when done.
Deadline 27 Sep 23:59 IST (confirm on the portal that the zip has the same deadline).

## What the organisers ask for (problem statement + the candidate_pairs update email)
- One zip, `<team>_submission.zip` (team **SSM**: Sahaj Mistry, Sarvesh Nikas, Mohanish Baviskar):
  - `output/matching_results.tsv`: **the same file as the final leaderboard upload**.
  - `output/candidate_pairs.tsv`: the exact pairs the matcher scored; every matched ID must be in it.
  - `code/business_entity_resolution/`: `src/`, `README.md` with exact end-to-end run instructions, `requirements.txt`
    pinned. Anyone must be able to regenerate both output files from the data using only this folder.
  - `Documentation_template.md` filled in: methodology, blocking, model architecture + features, anything else.
- Models MIT / Apache-2.0 and ≤ 8B parameters. No external databases, APIs or lookups. Country is an open set:
  nothing hard-coded to {US, India}.
- A smaller candidate set per S1 ranks higher; they review candidate_pairs.tsv and the code that makes it.

## Never in the zip (none of these are on `final-dd`; still check the zip's file list)
- `architecture.md` and `review/` (hand inspection of test records): pushed to v11 for the team only.
- `findings.md`, this file, `submissions/`, `eda/`, `compute`, the organiser PDFs / video, `work/` caches, model
  weights, `student_resource/`.

## To do
- [ ] **Final file** (user, 27 Sep 12:50): `variant_v10seed_dd` (LB 0.990349, md5 e73c409e) stays in the package
      for now; it will be **replaced by the padum second-half check's output files** once that run finishes. Then the
      final leaderboard upload must be that output too (the zip's matching_results.tsv byte-identical to it), and the
      candidate set is the rerun's (~7.72 pairs per S1, not 7.37). Which upload counts for the private leaderboard:
      decided later.
- [x] **`final-dd` builds the variant's pipeline** (171c552): `reproduce.sh` is v10 only, no v8 / v9 stacks; the
      judge trains on v10's unsure records (the submitted file's judge used an older stage 1: README says so).
- [ ] **Second-half check** (padum `~/scratch/amlc_v10only`, started 12:22, ETA ~18:00): the new `reproduce.sh` from
      stage 2 on, on the rerun's files; compare with variant_v10seed_dd (validator, rows identical, F0.5, validation).
      Then put its numbers in final-dd's README and rebuild the package.
- [x] **Reproducibility rerun** (done 11:34, findings 11:50): validator PASS, 98.9% of S1 rows as the submitted
      dd, validation equal; HEAD's last steps give the same file byte for byte. A rerun gives 7.72 candidate
      pairs per S1 (submitted 7.37): ship the submitted files and say this in the README.
- [x] **Cluster paths**: `x_llm.py` defaults to `Qwen/Qwen3-Reranker-4B`; `package.sh` has none; package grep clean.
- [x] **`package.sh`** (6832d5f): builds `$AMLC_ROOT/submission/<TEAM>_submission/` + zip, `reproduce.sh` included,
      validator `--check-ids` by absolute path. On Windows run it from a `git -c core.autocrlf=false archive` export
      (the worktree's line endings can be CRLF).
- [ ] **Write `Documentation_template.md`**: the one in `ber/` on v11 still describes v1 (25 Sep, exact search +
      LightGBM, 0.9890) and `final-dd` has none. Cover blocking with numbers (7.37 candidate pairs per S1, the
      recall the shortlist keeps on validation, the reduction ratio), the matcher stack, the LLM judge, self-training
      for the country without labels, the same-address fixes, validation, licences, compute. Use architecture.md as
      source material; the file itself stays out. Figure: `ber/pipeline.drawio` (source), `ber/pipeline.png`
      (export): put the PNG in the zip next to the .md, or export the document to PDF. The figure shows the France
      rounds starting from the main cross-encoders (variant_v10seed_dd); if dd is the final file, say that its first
      two rounds started from the older v8 cross-encoders.
- [x] **Package grep**: no `/scratch`, `/home`, `aib262144`; country names only in docstrings and `common.py`'s self-test.
- [x] **Package built** 12:26: `submission/SSM_submission/` + `SSM_submission.zip` (122 MB, 32 files, no CRLF),
      validator `--check-ids` PASS, md5 e73c409e / 1a8b4f5c. Its `Documentation_template.md` is still the blank
      template: rebuild the package after writing it (`bash package.sh SSM <tsv dir>`).
- [x] **Size**: the zip is 122 MB; the portal has no upload size limit (user, 27 Sep).

## Remember
- Validator trap: run from a symlinked `student_resource`, `../output_*` resolves to the original project and can
  PASS the wrong file. Always pass absolute paths.
- GPU training is not bit-exact: the README says so; the evidence is the per-stage validation agreement and
  compare.py, not md5.
- `tight_dd` (6.27 pairs per S1) was dropped because France loses (findings 10:10). A smaller candidate set needs a
  matcher that keeps the namesake competition.
- Keep padum's work dirs (`~/scratch/AmazonMLChallenge/work`, `amlc_final`, `amlc_variant`, `amlc_v10only`) until the final rankings
  are confirmed, in case the organisers ask for a rerun.
- One person uploads the zip; post the final md5s to the team.
