# v12b: v10_fr3_llm with two more France vetoes (v9fX stack, LLM judge on high-confidence rows)

Branch `v12b`, built on `v11` 2183202 (Sahaj's v10_fr3_llm, leaderboard 0.987071). US / India decisions are unchanged;
only France, the country without training labels, changes.

## Starting point

| Model | Change | Leaderboard |
|---|---|---:|
| v9p | house-number peers | 0.982690 |
| v9p_frand | + France veto round 1 (self-trained cross-encoders) | 0.984742 |
| v10_fr2 | + distractor folds, veto round 2 | 0.985942 |
| v10_fr3 | + veto round 3 | 0.986077 |
| **v10_fr3_llm** (v11) | + LLM judge (Qwen3-Reranker-4B, LoRA) blended into rows with 0.01 < q < 0.99 | **0.987071** |
| **v12b** | + v9fX veto + LLM judge on France rows accepted at q ≥ 0.99 | pending (est. 0.9873–0.9879) |

From the France-blank probe and these steps, US / India score about 0.992 and France about 0.959. The gap to the top of
the leaderboard (0.990) is France.

## What was added and why

### 1. v9fX as a fourth France veto

- The v9fX stack is built on branch `v9fS` (`python x_chain.py v9mX`, file `x/test_q_v9fXw.parquet`).
  - Pseudo-labels: France's confident decisions of v9f (q ≥ 0.95: its best S1 is a match and its other candidates are
    not; q ≤ 0.05: no candidate is a match).
  - Both e5 cross-encoders continue from their trained weights for one pass at lr 2e-5, on those pairs mixed 1:1 with
    train pairs so US / India are kept.
  - The France records are split in two halves. Each half is re-scored by the model trained on the other half, so a
    record the seed accepted confidently can still be rejected. Round 1–3's self-training re-scores the records it
    trained on, so it keeps its own confident mistakes.
  - Stages 1–2 then run on the re-scored test pairs.
- It uses the same 12,760,925 candidate pairs as v9 / v10, so the rows match and `x_final.py FROM=France:_v9fXw:min`
  applies it like the other vetoes.
- On top of v10_fr3_llm's three vetoes it rejects **2,731** more France records. The sampled ones are descriptor swaps:

  | Record | Its S1 |
  |---|---|
  | Entre Club EURL ; 103 R Racine, Lille | Entre Comite EURL ; 103 Rue Racine, Lille |
  | SARL Peche Amicale ; 091 Rue Henri Ghesquières, Lille | Peche Club SARL ; 91 Rue Henri Ghesquières, Lille |
  | Handball Amicale SARL ; 167 Rue Mandron, Bordeaux | Handball Collège SARL ; 167 Rue Mandron, Bordeaux |
  | Publique Club ; 41 Rue de la Bénétrie, Pornic | Publique Compagnie ; 41 Rue de la Bénétrie, Pornic |

### 2. The LLM judge on the rows it never saw (`x_llmhi.py`)

- v11's judge only scores rows with 0.01 < q < 0.99. Of v10_fr3_llm's 838,865 France matches, **785,586 (94%) sit at
  q ≥ 0.99** and were never shown to it.
- For a country with training labels that is safe: on US / India validation, rows at q ≥ 0.99 are 99.99% true matches
  (1,477,692 rows, 160 not true). France's stage-2 q is not calibrated: 6% of its accepted rows sit in the unsure band
  (53,279 of 838,865), against 1.5–2.2% for US / India (measured on v9p).
- The same judge (Sahaj's LoRA, `x/llm_lora`) scores those rows: ~50 min on two A100s at ~135 pairs/s each. A row is
  rejected when the judge answers no: margin logit(yes) − logit(no) < 0. The cut is the judge's own decision, not tuned
  on the leaderboard.

| Rows at stage-2 q ≥ 0.99 | Rows | Margin < −2 | < −1 | < 0 |
|---|---:|---:|---:|---:|
| US / India validation, true matches | 40,000 (sampled) | 0.04% | 0.06% | 0.12% |
| US / India validation, not true | 160 (all) | 5.6% | 5.6% | 10.6% |
| France test, accepted by v10_fr3_llm | 785,586 | 0.48% (3,731) | 0.69% (5,408) | **0.92% (7,205)** |

- The judge says no to France rows 7.6 times as often as to US / India true matches. If French true matches looked like
  US / India ones to the judge, at most 13% of the 7,205 would be true matches.
- Reading 25 random rows per band, the decoys are mostly a swapped descriptor with the address unchanged:

  | Margin | Rows | Read as decoys | Examples (record → its S1) |
  |---|---:|---:|---|
  | below −2 | 3,731 | 25 of 25 | Organisme Lycee SARL → Organisme Union SARL; KXU Ecole SA → KXU Club SA; GGK ECOLE SARL → GGK Groupe SARL |
  | −2 to −1 | 1,677 | about 23 of 25 | Calais Comite SARL → Calais Club SARL; Développement Pont SARL → Pont Union SARL |
  | −1 to 0 | 1,797 | about 18 of 25 | Fun SARL Groupe → Fun Club SARL; Comite de Sos → Institut de Sos |

  The true matches among them are acronyms ("CP" for "CB Parents SAS", "SC" for "SBV Compagnie SARL") and reordered or
  reformatted names ("NM SAS Collectif" for "NM Collectif SAS").
- 6,722 of the 7,205 are still accepted after the v9fX veto, so the two vetoes are nearly disjoint (483 in common).

## Test output

| File | Accepted | US | India | France (per S1) | Empty S1s |
|---|---:|---:|---:|---:|---:|
| v10_fr3_llm | 5,848,161 | 2,256,856 | 2,752,440 | 838,865 (3.23) | – |
| **v12b** | **5,838,708** | 2,256,856 | 2,752,440 | **829,412 (3.20)** | 102,665 |

- `matching_results.tsv` md5 e3d9e4a7e2332b14a4eba73ab3b51fd2.
- `candidate_pairs.tsv` md5 1a8b4f5c55108bc3d331817a4c86238f: the v9 / v10 candidate set, 12,760,925 pairs, 7.37 per
  S1 (unchanged).
- `validate_submission.py --check-ids`: PASS.

## Expected leaderboard

The leaderboard history prices a France record removed by a veto:
- round 1 removed 23,559 records for +0.002052, i.e. +8.7e-8 each;
- round 3 removed 6,095 records for +0.000135, i.e. +2.2e-8 each.

v12b removes 9,453, so it should add +0.0002 to +0.0008, **about 0.9873–0.9879** (v11: 0.987071). v12b differs from v11
only on France, so its score difference is France's alone.

## Compliance

- The judge is the one v11 already uses: `Qwen/Qwen3-Reranker-4B` (Apache-2.0, 4.0B parameters), run offline, with a LoRA
  adapter fine-tuned only on the provided training data (`x_llm.py train`, S1 folds 4–7).
- `x_llmhi.py` selects the countries to check from the data: test countries without a single labelled training record.
- No labels, external data or hand-written rules are used. The vetoes use the pipeline's own confident decisions on the
  test records (self-training, allowed by the organisers) and the judge's own yes / no.

## Reproduce

```bash
# the v9fX stack: branch v9fS (python x_chain.py v9m, then v9mX) -> copy work/x/test_q_v9fXw.parquet into work/x
LLM=/path/to/qwen3-reranker-4b python x_chain.py v12b          # one GPU -> ../output_v12b
```

With two GPUs, the steps of that plan are:

```bash
FROM=France:_v9spw:min,France:_v9s2pw:min,France:_v10s3pw:min SAVE_Q=_v10fr3lw python x_final.py - _v10plw 0.70
python x_llmhi.py rows _v10plw _v10fr3lw _v10pw       # France rows accepted at q >= 0.99 + a validation sample
SHARD=0/2 CUDA_VISIBLE_DEVICES=0 python x_llmhi.py test & SHARD=1/2 CUDA_VISIBLE_DEVICES=1 python x_llmhi.py test; wait
python x_llmhi.py val && python x_llmhi.py check      # the table above and sample rejected rows
FROM=France:_v9spw:min,France:_v9s2pw:min,France:_v10s3pw:min,France:_v9fXw:min LLMVETO=0 \
  python x_final.py ../output_v12b _v10plw 0.70
```

## Files

- **New:** `src/x_llmhi.py` (row selection, the judge on those rows in GPU shards, `check`), `description.md`.
- **Changed:**
  - `src/x_final.py`: `LLMVETO=M` rejects rows whose judge margin is below M.
  - `src/x_chain.py`: plan `v12b`.
  - `README.md`: v12b section, results row and reproduce steps.
  - `requirements.txt`: `peft` and `scikit-learn`, already used by v11's LLM judge.
  - `package.sh`: both added to the packaged requirements.
