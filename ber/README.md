# Business Entity Resolution

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.
This folder holds only the code that builds our final submission file (**leaderboard 0.990934**) and `reproduce.sh`, which
runs it end to end from the provided data.

## Pipeline

Every S2/S3 record belongs to at most one S1 entity (training ground truth: 7.6M matched ids, none reused), so each
record is resolved on its own: which S1 entity, if any, is it? Country is an open set of labels; nothing below names a
country or a language.

1. **Normalise** (`common.py`, `lexicon.py`): transliterate every script to ASCII (`anyascii`), lowercase, tokenise
   (`L.L.C.` → `llc`, `N°29` → `ndeg 29`). Abbreviations and legal forms come from a per-country lexicon learned
   without labels from pseudo-matches (a record and its top retrieved S1 at cosine ≥ 0.9).
2. **Blocking** (`train_embed.py`, `retrieve.py`, `x_shortlist2.py`, `match.py`): `intfloat/multilingual-e5-small`
   fine-tuned on (record, S1) pairs of S1 folds 0–3; each record takes its top-20 S1s of the same country from a FAISS
   HNSW index (M 32, efSearch 512); a small LightGBM on retrieval features and three rapidfuzz similarities keeps
   candidates with P ≥ 0.001 (7.37 per S1 on test). These pairs are what the matcher scores and what
   `candidate_pairs.tsv` lists.
3. **Pair features** (`match.py`, `x_feats.py`): embedding score / rank / gaps, rapidfuzz similarities on full, core and
   consonant-skeleton names and on addresses, numbers, token rarity, integer-aware house numbers, legal-form edits, and
   name edits between core names with per-country word statistics from each split's own records.
4. **Two cross-encoders** (`x_ce3.py`): e5-small from the fine-tuned bi-encoder, on normalised text and on raw
   transliterated text, fine-tuned on the retrieved top-5 pairs of S1 folds 0–3. A distractor (a record with no match)
   is put in the fold of the S1 it imitates, so every entity keeps all its distractors (`DFOLD=1`).
5. **Stage 1** (`x_stage_multi.py s1`, `S1K=3`): LightGBM on pair + cross-encoder features, cross-fitted over S1
   folds 4–9 in three groups. **Stage 2** (`x_nocopy.py`): each record's best candidate re-scored with entity context
   (the other records claiming the same S1, and those at the same house number: `PEERS=1`), distractors weighted to
   the test share (39%), five seeds.
   **A third cross-encoder for the countries with training labels:** `BAAI/bge-reranker-v2-m3` on the text as given
   (case, accents, scripts), fine-tuned on 3M pairs of folds 0–3 (`x_ce3.py` with `CE_BASE`), and stages 1–2 rerun
   with all three cross-encoders. This stack is the base for the countries with labels (US / India validation with the
   LLM judge 0.99294, two-cross-encoder stack 0.99282); the countries without labels keep the two-cross-encoder stack,
   on which their self-training rounds are built.
6. **Self-training for the countries without training labels** (test countries with no labelled training record,
   found from the data by `common.unlabelled()`: France here; `x_ce3.py train` with `SELF=`): both
   cross-encoders are continued on test pairs whose decision so far is confident (q ≥ 0.98: the best S1 is a match and
   the record's other top-5 candidates are not; q ≤ 0.02: none is), mixed with train pairs, and stages 1–2 are rerun.
   Three rounds: round 1 is seeded by stage 2's decisions, round k by stage 2 vetoed by rounds < k. Each round is a
   veto: such a record is accepted only if every stack picks the same S1, at the lowest of their scores
   (`x_final.py FROM=unlabelled:<tag>:min`). No labels are used; the organisers' Q&A
   allows self-training on the test records.
7. **LLM judge** (`x_llm.py`, `x_llmstack.py`): `Qwen/Qwen3-Reranker-4B` with a LoRA (r 16) trained on 120k train
   records of folds 4–7 whose stage-1 p is unsure; its yes − no margin is blended with the stage-2 score by a logistic
   regression fitted on validation, for unsure records (0.01 < q < 0.99) only, in both stacks.
8. **Same-address fixes for the countries without labels** (`x_ddfix.py`, with `rule_fr.py`, `wstat.py`, `x_anatomy.py`):
   decoys copy an S1's name with one edit and move to a shifted house number; each S1 gets a roughly fixed number of
   them, so a decoy placed at the S1's own address leaves its shifted cluster short. For every word added at the same
   address, the spread of those records over the S1's records at other numbers is compared with unedited records
   (total variation distance): TV ≤ 0.10 is true-like, TV ≥ 0.20 decoy-like (on US / India validation 98.7% / 90.8% and
   22.5% true). Records the structural rule accepts (same house number and sub-number, the S1's distinctive street
   words, first and rarest core-name word, no decoy word by the house-number word statistic) whose added words are all
   true-like are restored; accepted same-address records adding a decoy-like word are rejected. Where labels exist the
   models already get these right and the same fixes would hurt (`python x_ddfix.py check`: India would lose 0.0024),
   so they apply only to the countries without labels.
9. **Recall fixes** (`x_recall.py`):
   - *Empty S1s:* an S1 with no accepted record takes its best candidate if its score is ≥ 0.40. An S1 left empty
     scores 0 as soon as it has one true match (US / India validation +0.00007). For the countries without labels the
     LLM judge must also not say no: the candidates it rejects there are mostly a type word swapped at the S1's own
     address, the same-building decoy of step 8.
   - *No-address records, countries without labels:* a rejected record with no address whose normalised name equals
     its S1's and no other S1's of the country, and that the judge accepts, is restored. US / India models accept
     93–97% of such records, France's 84%: its self-training vetoes learned "no address = reject".
   - *Vetoed records of kinds that are almost always true, countries without labels:* records the main stack accepts
     and a self-trained stack vetoes, of kinds the US / India models accept and get right 95–100% of the time: at the
     S1's exact address, the only S1 there, a name that adds no word (initials, a website, dropped words). Also
     no-address records with the S1's core name (legal form differs) that no other S1 has, judge yes. France: 161 +
     1,078 restores. `x_recall.py` also builds two more such lists, `restore_vetona` (an address without a house
     number on the S1's street or with a name no other S1 has, judge margin ≥ 2) and `restore_samename` (the S1's
     exact name at another number of its street), but the final step does not use them: removing their 1,712 France
     matches raised the leaderboard by +0.00002, so France's self-training vetoes of these kinds are mostly right.
   - *Aliases and initials at the S1's exact address, countries without labels* (the only S1 there): a made-up name
     sharing nothing with the S1's name (a word of 4+ letters in no S1 name of any country: the generator's
     unrelated-DBA noise) when the main stack's q ≥ 0.80, or q ≥ 0.10 with at most 2 made-up names at the S1 (US /
     India validation 97–99.8% / ~90% true; below 0.10 they are 0% true); and the S1's initials there even where the
     main stack rejected them (US / India: 100% true). France: 336 aliases, 56 more initials.
   - *Decoy words anywhere, countries without labels* (`reject_decword`): a record adding a decoy-like word of the
     census (step 8: the same-building type-word swap, ~22% true on US / India) is rejected at any position, not only
     at the S1's house number (157 France matches). With these two, the leaderboard rose by +0.00007.
10. **Decision** (`x_final.py`): accept a record when its score is ≥ 0.70, the same threshold for every country.

Folds: `crc32(S1 id) % 10`; 0–3 train the bi-encoder and the cross-encoders, 4–9 stages 1 and 2; stage 2 is validated
on entity folds 8–9 after fitting on 4–7 (US / India validation F0.5 0.99294 with the bge stack and the LLM judge,
+0.00007 with the empty-S1 rule).

## Leaderboard

| File | Score |
|---|---|
| **Final submission** (this pipeline) | **0.990934** |
| + `restore_vetona`, `restore_samename` (vetoed records without a house number or at another number of the street) | 0.99091 |
| C4 (also without the alias, initials and decoy-word lists of step 9) | 0.99084 |
| E16fr3 (without the vetoed-record and core-name lists of step 9) | 0.990807 |
| variant_v10seed_dd (without the bge stack and the recall fixes) | 0.990349 |
| v10_fr3_llm_dd (France rounds seeded from two older stacks, since removed) | 0.990282 |
| v10_fr3_llm (without the same-address fixes) | 0.987071 |
| v10_fr3 | 0.986077 |
| v10_fr2 | 0.985942 |
| v9p_frand | 0.984742 |

## Reproduce

```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt
AMLC_ROOT=/folder/with/student_resource GPU_A=0 GPU_B=1 GPU_C=2 bash reproduce.sh
```
- Data: `student_resource/` (the provided zip, unchanged) inside `AMLC_ROOT`; caches, models and logs go to
  `$AMLC_ROOT/work`, the submission to `$AMLC_ROOT/output/` (validated with `--check-ids` at the end).
- Models (fetched on first use; on an offline node set `HF_HOME`, or `E5=` / `LLM=` / `BGE=` to local copies):
  `intfloat/multilingual-e5-small` (MIT, 118M), `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M) and
  `Qwen/Qwen3-Reranker-4B` (Apache-2.0, 4.0B).
- Hardware: three A100 80GB GPUs (fewer work: `GPU_C` defaults to `GPU_B`, `GPU_B` to `GPU_A`), ~64 cores, ~250 GB
  RAM. About 9–10 h with three GPUs; retrieval (CPU) takes 2 of them.
- Each step logs to `work/logs/<step>.log` and leaves `<step>.done`; rerunning `reproduce.sh` resumes after the last
  finished step. The steps and their environment variables are listed in `reproduce.sh` in dependency order.
- GPU training (bi-encoder, cross-encoders, LoRA) is not bit-deterministic, so a rerun matches the submitted files
  closely but not byte for byte. The submitted file was built by this script's steps on the intermediate files of our
  original run. A rerun of this script (from scratch through stage 2, then every later step): every stage's
  validation within 0.0001 of the original run, and the file before the bge stack and the recall fixes matched ours
  on 98.9% of S1 rows (F0.5 0.9977 scored against it), with 7.72 candidate pairs per S1 (submitted 7.37: the
  retrained shortlist model calibrates a little differently). The submitted file's LLM judge was trained on the unsure
  records of an earlier stage 1 that used two more cross-encoders (since removed); this script trains it on its own
  stage 1's (validation with the judge 0.99283; the submitted file's judge 0.99282).

No external databases, APIs or lookup services are used at any stage, and there are no hand-written word lists:
abbreviations, legal forms and word statistics are learned from the provided records (test records without labels,
as allowed for unsupervised statistics), and self-training uses only the pipeline's own confident test decisions.
