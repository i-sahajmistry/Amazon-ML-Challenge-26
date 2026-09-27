# Business Entity Resolution: v10_fr3_llm_dd

Match every Source 2 / Source 3 record to the Source 1 (reference) entity it belongs to, scored by macro F0.5.
This branch holds only the code that builds the submitted file **v10_fr3_llm_dd (leaderboard 0.990282)**, and
`reproduce.sh`, which runs it end to end from the provided data.

**Branch `variant-frB`:** also builds **variant_frB** (leaderboard pending) = variant_v10seed_dd (0.990349: the
France self-training rounds seeded from v10) + two recall fixes (`x_recall.py`); see [variant_frB](#variant_frb).

## Pipeline

Every S2/S3 record belongs to at most one S1 entity (training ground truth: 7.6M matched ids, none reused), so each
record is resolved on its own: which S1 entity, if any, is it? Country is an open set of labels; nothing below names a
country or a language.

1. **Normalise** (`common.py`, `lexicon.py`): transliterate every script to ASCII (`anyascii`), lowercase, tokenise
   (`L.L.C.` → `llc`, `N°29` → `ndeg 29`). Abbreviations and legal forms come from a per-country lexicon learned
   without labels from pseudo-matches (a record and its top retrieved S1 at cosine ≥ 0.9).
2. **Blocking** (`train_embed.py`, `retrieve.py`, `shortlist.py`, `x_shortlist2.py`, `match.py`): `intfloat/multilingual-e5-small`
   fine-tuned on (record, S1) pairs of S1 folds 0–3; each record takes its top-20 S1s of the same country from a FAISS
   HNSW index (M 32, efSearch 512); a small LightGBM on retrieval features and three rapidfuzz similarities keeps
   candidates with P ≥ 0.001. These pairs are what the matcher scores and what `candidate_pairs.tsv` lists.
3. **Pair features** (`match.py`, `x_feats.py`): embedding score / rank / gaps, rapidfuzz similarities on full, core and
   consonant-skeleton names and on addresses, numbers, token rarity, integer-aware house numbers, legal-form edits, and
   name edits between core names with per-country word statistics from each split's own records.
4. **Two cross-encoders** (`x_ce.py`, `x_ce3.py`): e5-small from the fine-tuned bi-encoder, on normalised text and on
   raw transliterated text, fine-tuned on the retrieved top-5 pairs of S1 folds 0–3.
5. **Stage 1** (`x_stage_multi.py s1`, `S1K=3`): LightGBM on pair + cross-encoder features, cross-fitted over S1
   folds 4–9 in three groups. **Stage 2** (`x_nocopy.py`): each record's best candidate re-scored with entity context
   (the other records claiming the same S1, and those at the same house number: `PEERS=1`), distractors weighted to
   the test share (39%), five seeds.
6. **v10** (`DFOLD=1`): a distractor takes the fold of the S1 it imitates, so every entity keeps all its distractors;
   both cross-encoders retrained, then stages 1–2 as above.
7. **Self-training for the countries without training labels** (test countries with no labelled training record,
   found from the data by `common.unlabelled()`: France here; `x_ce3.py train` with `SELF=`): both
   cross-encoders are continued on test pairs whose earlier decision was confident (q ≥ 0.98: the best S1 is a match and
   the record's other top-5 candidates are not; q ≤ 0.02: none is), mixed with train pairs. Three rounds (v9s seeded by
   v8, v9s2 by v9p_frand, v10s3 by v10_fr2), each restacked; each acts as a veto: such a record is accepted only if
   every stack accepts it for the same S1 (`x_final.py FROM=unlabelled:<tag>:min`). No labels are used; the organisers' Q&A
   allows self-training on the test records.
8. **LLM judge** (`x_llm.py`, `x_llmstack.py`): `Qwen/Qwen3-Reranker-4B` with a LoRA (r 16) trained on 120k train
   records of folds 4–7 whose stage-1 p is unsure; its yes − no margin is blended with the stage-2 score by a logistic
   regression fitted on validation, for unsure records only.
9. **Same-address fixes for the countries without labels** (`x_ddfix.py`, with `rule_fr.py`, `wstat.py`, `x_anatomy.py`):
   decoys copy an S1's name with one edit and move to a shifted house number; each S1 gets a roughly fixed number of
   them, so a decoy placed at the S1's own address leaves its shifted cluster short. For every word added at the same
   address, the spread of those records over the S1's records at other numbers is compared with unedited records
   (total variation distance): TV ≤ 0.10 is true-like, TV ≥ 0.20 decoy-like (on US / India validation 98.7% / 90.8% and
   22.5% true). Records the structural rule accepts (same house number and sub-number, the S1's distinctive street
   words, first and rarest core-name word, no decoy word by the house-number word statistic) whose added words are all
   true-like are restored; accepted same-address records adding a decoy-like word are rejected. Where labels exist the
   models already get these right and the same fixes would hurt (`python x_ddfix.py check`: India would lose 0.0024),
   so they apply only to the countries without labels.
10. **Decision** (`x_final.py`): accept a record when its score is ≥ 0.70, the same threshold for every country.
11. **Recall fixes (variant_frB, `x_recall.py`)**:
    - *Empty S1s* (all countries): an S1 with no accepted record takes its best candidate if its score is ≥ 0.40. An
      S1 left empty scores 0 as soon as it has one true match; on US / India validation the rule gains +0.00009.
    - *No-address records, countries without labels*: a rejected record with no address whose normalised name, or core
      name, equals its S1's and no other S1's of the country, and that the LLM judge accepts. On US / India validation
      such records are 96–98% (name) / 76–78% (core name) true and the models accept 95–98% / 68–70% of them; France
      accepts 84% / 50%. French names reuse a small vocabulary, so a name-only match looks weak to models trained on
      US / India. Where labels exist the rejected ones are mostly wrong (53–62% / 29–32% true), so this applies only to
      the countries without labels.

Folds: `crc32(S1 id) % 10`; 0–3 train the bi-encoder and the cross-encoders, 4–9 stages 1 and 2; stage 2 is validated
on entity folds 8–9 after fitting on 4–7 (US / India validation F0.5 0.99282 for v10_fr3_llm).

## Leaderboard

| File | Score |
|---|---|
| variant_frB (`variant-frB`: variant_v10seed_dd + `x_recall.py`) | pending |
| **variant_v10seed_dd** (France rounds seeded from v10; `variant_v10seed_dd/variant.sh`) | **0.990349** |
| v10_fr3_llm_dd (`reproduce.sh`) | 0.990282 |
| v10_fr3_llm (`output_v10_fr3_llm`, also written by `reproduce.sh`) | 0.987071 |
| v10_fr3 | 0.986077 |
| v10_fr2 | 0.985942 |
| v9p_frand | 0.984742 |

## Reproduce

```bash
conda create -n amlc python=3.12 -y && conda activate amlc
pip install -r requirements.txt
AMLC_ROOT=/folder/with/student_resource GPU_A=0 GPU_B=1 bash reproduce.sh
```
- Data: `student_resource/` (the provided zip, unchanged) inside `AMLC_ROOT`; caches, models and logs go to
  `$AMLC_ROOT/work`, the submission to `$AMLC_ROOT/output_v10_fr3_llm_dd/` (validated with `--check-ids` at the end).
- Models (fetched on first use; on an offline node set `HF_HOME`, or `E5=` / `LLM=` to local copies):
  `intfloat/multilingual-e5-small` (MIT, 118M) and `Qwen/Qwen3-Reranker-4B` (Apache-2.0, 4.0B).
- Hardware: two A100 80GB GPUs (one works: `GPU_B=$GPU_A`), ~64 cores, ~250 GB RAM. About 10 h with two GPUs.
- Each step logs to `work/logs/<step>.log` and leaves `<step>.done`; rerunning `reproduce.sh` resumes after the last
  finished step. The steps and their environment variables are listed in `reproduce.sh` in dependency order.
- GPU training (bi-encoder, cross-encoders, LoRA) is not bit-deterministic, so a rerun matches the submitted file
  closely but not byte for byte.

No external databases, APIs or lookup services are used at any stage, and there are no hand-written word lists:
abbreviations, legal forms and word statistics are learned from the provided records (test records without labels,
as allowed for unsupervised statistics), and self-training uses only the pipeline's own confident test decisions.

## variant_frB

variant_v10seed_dd is dd with the three France self-training rounds seeded from v10's own decisions and started from
v10's cross-encoders (no v8 / v9 stacks), and the LLM judge's scores on every unsure v10 row;
`variant_v10seed_dd/variant.sh` lists its steps as run on the original run's v10 files. variant_frB adds the two
recall lists on top of it:
```bash
AMLC_ROOT=/variant/root bash variant_frB/frB.sh   # needs the variant's work/x (see the script's header)
python src/x_recall.py val                        # the empty-S1 rule on US / India validation, with labels
```
| | US | India | France | md5 (matching_results) |
|---|---:|---:|---:|---|
| variant_v10seed_dd, accepted | 2,255,158 | 2,751,010 | 859,338 | e73c409e |
| + empty-S1 rule (1,415 rows) | +546 | +430 | +439 | |
| + no-address fix (5,445 rows, France; 48 already in the empty-S1 list) | | | +5,397 | |
| **variant_frB**, accepted | 2,255,704 | 2,751,440 | 865,174 | **73b4a5ea** |

candidate_pairs.tsv is unchanged (1a8b4f5c: 12,760,925 pairs, 7.37 per S1).
