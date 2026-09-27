# Final submission notes

What to do and remember before the final submission. Add items as they come up; tick them when done.
Deadline 27 Sep 23:59 IST (confirm on the portal that the zip has the same deadline).

## What the organisers ask for (problem statement + the candidate_pairs update email)
- One zip, `<team>_submission.zip` (team name as on the portal; the template says SSM):
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
- [ ] **Pick the final file**: `variant_v10seed_dd` (LB 0.990349, md5 e73c409e) or dd (LB 0.990282, md5 75f68b10).
      The zip's matching_results.tsv must be byte-identical to the final upload (compare md5). Check on the portal
      which upload the private leaderboard uses (the last one or a selected one).
- [ ] **If the variant is final, `final-dd` must build it**: its steps in `reproduce.sh` (three France rounds seeded
      from v10 instead of v9s / v9s2 / v10s3; `QB=_v10p x_llm.py extend`; `x_ddfix` instead of x_rule1 + x_wordlists;
      `unlabelled:` instead of `France:`), checked on the rerun's v10 files (~4-5 h on one GPU); README title,
      leaderboard and steps updated. As run on padum: `ber/variant_v10seed_dd/variant.sh` on v11.
- [ ] **Finish the reproducibility rerun** (padum `~/scratch/amlc_final`): validator with absolute paths, `compare.py`
      against the submitted file, then final-dd HEAD's last steps (x_ddfix) on the rerun's q files. Log in findings.md.
- [ ] **Padum-only paths in `final-dd`**: `src/x_llm.py` defaults `LLM=/scratch/scai/.../qwen3-reranker-4b` (make it
      `Qwen/Qwen3-Reranker-4B`, as the README promises); `package.sh` defaults to `$HOME/scratch/...` and padum's pip.
- [ ] **Fix `package.sh`**:
  - it copies `$AMLC_ROOT/output/*.tsv`, but `reproduce.sh` writes `output_v10_fr3_llm_dd/`, so as written it packages
    the wrong files;
  - it leaves `reproduce.sh` out of `code/business_entity_resolution/`, but the README's run command needs it;
  - copy `ber/requirements.txt` instead of grepping a pip freeze;
  - run the validator with `--check-ids` and absolute paths.
- [ ] **Write `Documentation_template.md`**: the one in `ber/` on v11 still describes v1 (25 Sep, exact search +
      LightGBM, 0.9890) and `final-dd` has none. Cover blocking with numbers (7.37 candidate pairs per S1, the
      recall the shortlist keeps on validation, the reduction ratio), the matcher stack, the LLM judge, self-training
      for the country without labels, the same-address fixes, validation, licences, compute. Use architecture.md as
      source material; the file itself stays out.
- [ ] **Grep the package before zipping**: no `/scratch`, `/home`, `aib262144`; country names only in comments and
      `common.py`'s self-test.
- [ ] **Validate the packaged outputs** from `student_resource/` with absolute paths and `--check-ids`: PASS.
- [ ] **Size**: matching_results 98 MB + candidate_pairs 187 MB unzipped; check the portal's upload limit.

## Remember
- Validator trap: run from a symlinked `student_resource`, `../output_*` resolves to the original project and can
  PASS the wrong file. Always pass absolute paths.
- GPU training is not bit-exact: the README says so; the evidence is the per-stage validation agreement and
  compare.py, not md5.
- `tight_dd` (6.27 pairs per S1) was dropped because France loses (findings 10:10). A smaller candidate set needs a
  matcher that keeps the namesake competition.
- Keep padum's work dirs (`~/scratch/AmazonMLChallenge/work`, `amlc_final`, `amlc_variant`) until the final rankings
  are confirmed, in case the organisers ask for a rerun.
- One person uploads the zip; post the final md5s to the team.
