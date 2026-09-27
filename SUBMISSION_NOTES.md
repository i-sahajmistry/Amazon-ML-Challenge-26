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
- [x] **Final file** (27 Sep 22:36): **C8, leaderboard 0.99091, rank 18** (matching_results a7d37122, candidate_pairs
      1a8b4f5c): C4 + restore_alias + initials + reject_decword. final-dd **a4c3fa1** (pushed) builds it. Package rebuilt
      from a clean `git archive` of a4c3fa1: `submission/SSM_submission.zip` (122 MB), validator --check-ids PASS,
      output = C8 byte for byte, x_recall.py = the one that built C8, no CR bytes, no cluster paths, no notes. C8 is the
      last leaderboard upload. README, documentation and figure updated for C8 (final-dd e835695); zip rebuilt 22:40. Only step left: upload the
      zip on the portal before 23:59.
- [x] **Final file** (27 Sep 20:40): **C4, leaderboard 0.99084** (matching_results 7c929413, candidate_pairs 1a8b4f5c):
      E16fr3 + four France recall lists (`x_recall.py`: restore_nacore, restore_vetona, restore_samename,
      restore_exact). final-dd **579db0d** (pushed) builds it; README, documentation and figure updated. Package rebuilt
      from a clean `git archive` of 579db0d: `submission/SSM_submission.zip` (122 MB), validator --check-ids PASS,
      output = C4 byte for byte, no CR bytes, no cluster paths, no notes. C4 is the last leaderboard upload (user).
      Only step left: upload the zip on the portal before 23:59.
- [x] Previous final file (27 Sep 15:46): **E16fr3, leaderboard 0.990807** (matching_results e93605ad, candidate_pairs
      1a8b4f5c = the variant's, 12,760,925 pairs, 7.37 per S1). final-dd d7db51d (pushed) builds its method (bge
      lane, `x_recall.py`); README, documentation and figure updated; fefa59e applies Sarvesh's doc review
      (DOC_REVIEW.md: table fix, +0.0081, error-table scope, veto validated on the LB). Package rebuilt from fefa59e:
      `submission/SSM_submission.zip` (122 MB, 33 entries, validator --check-ids PASS, no CRLF, no cluster paths, no
      architecture.md / review/ / notes). This upload is the last one (user): it is the final leaderboard file.
- [x] **`final-dd` builds the variant's pipeline** (171c552): `reproduce.sh` is v10 only, no v8 / v9 stacks; the
      judge trains on v10's unsure records (the submitted file's judge used an older stage 1: README says so).
- [x] **Second-half check** (padum `~/scratch/amlc_v10only`, done 15:00): validator PASS, 98.86% of S1 rows as the
      variant, F0.5 0.9977, every stage's validation within 0.0001 (findings 15:05). Put these numbers in final-dd's
      README reproducibility bullet.
- [ ] **bge chain on the check's files** (padum `~/scratch/amlc_v10only/bge.sh`, GPU C, ETA ~17:20): then build the
      E16fr3 method there and compare with the shipped file (evidence for the new steps).
- [x] **Reproducibility rerun** (done 11:34, findings 11:50): validator PASS, 98.9% of S1 rows as the submitted
      dd, validation equal; HEAD's last steps give the same file byte for byte. A rerun gives 7.72 candidate
      pairs per S1 (submitted 7.37): ship the submitted files and say this in the README.
- [x] **Cluster paths**: `x_llm.py` defaults to `Qwen/Qwen3-Reranker-4B`; `package.sh` has none; package grep clean.
- [x] **`package.sh`** (6832d5f): builds `$AMLC_ROOT/submission/<TEAM>_submission/` + zip, `reproduce.sh` included,
      validator `--check-ids` by absolute path. On Windows run it from a `git -c core.autocrlf=false archive` export
      (the worktree's line endings can be CRLF).
- [x] **Write `Documentation_template.md`** (final-dd db4d351, 13:14): all the organisers' sections, team SSM, embeds
      `pipeline.png` (figure checked against the code and corrected). Numbers are the variant's: when the padum
      output replaces it, update the candidate numbers (13,377,149 pairs, 7.72 per S1, 1.34 per record, reduction
      ratio), the France restore / reject counts, the validation, the LB row and the reproducibility bullet; the
      figure's candidate_pairs box too (12.76M / 7.37 / 1.28). Then rebuild the package.
- [x] **Package grep**: no `/scratch`, `/home`, `aib262144`; country names only in docstrings and `common.py`'s self-test.
- [x] **Package rebuilt** 13:14 from final-dd db4d351: `submission/SSM_submission.zip` (122 MB, 32 files, no CRLF,
      no cluster paths), validator `--check-ids` PASS, md5 e73c409e / 1a8b4f5c, filled-in documentation + figure.
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
