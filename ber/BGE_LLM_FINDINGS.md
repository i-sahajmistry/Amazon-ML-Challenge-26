# bge reranker + LLM judge on v9p_frand: findings and hand-over

Branch `bge-llm` = `ber-pipeline` (v9p_frand, leaderboard 0.984742) + a third cross-encoder (`BAAI/bge-reranker-v2-m3`)
+ an LLM judge (`Qwen/Qwen2.5-7B-Instruct`) + the PADUM job that builds every submission variant end to end.
Everything below was run on PADUM (IIT Delhi HPC, A100 80GB) on 2026-09-26.

## Status in one table

| File (PADUM `~/scratch/amlc_v9x/output_v9x_*`) | US / India from | France from | Validation (US/India) | Leaderboard |
|---|---|---|---|---|
| 0 = v9p_frand rebuilt from its own score files | v9p | v9p, vetoed by v9sp | 0.99277 | 0.984742 (identical file: 5,879,950 accepted, same per country) |
| **e = bge everywhere, no LLM** | bge stack | bge stack, vetoed by bge self-trained stack | **0.99293** | *to be submitted (score pending)* |
| d = bge everywhere + LLM on France | bge stack | bge+LLM stack, vetoed by bge+LLM self-trained stack | 0.99293 | not submitted |
| b = bge on France only | v9p | bge stack, vetoed | 0.99277 | not submitted |
| c = bge + LLM on France only | v9p | bge+LLM stacks, vetoed | 0.99277 | not submitted |
| a = LLM on France only (no bge) | v9p | v9p+LLM, vetoed | 0.99277 | not submitted |

Accepted matches (records linked), US / India / France:

| File | US | India | France |
|---|---|---|---|
| 0 (v9p_frand) | 2,269,893 | 2,753,242 | 856,815 |
| e (bge) | 2,267,103 | 2,752,900 | 851,596 |
| d (bge + LLM) | 2,267,103 | 2,752,900 | 851,923 |
| b | 2,269,893 | 2,753,242 | 851,596 |
| c | 2,269,893 | 2,753,242 | 851,923 |
| a | 2,269,893 | 2,753,242 | 857,577 |

Reading the leaderboard: e − 0.984742 is bge's effect alone (same rule, same threshold, same candidate file).
If e is higher, bge is worth building on (next step below); b vs e would then split it into France and US/India.

## bge-reranker-v2-m3 as a third cross-encoder

- Model: Apache-2.0, 568M parameters (XLM-RoBERTa large), multilingual, pretrained as a reranker. Offline copy on
  PADUM: `/scratch/scai/mtech/aib262045/models/bge-reranker-v2-m3`.
- Trained exactly like the other cross-encoders (`x_ce3.py`, only folds 0–3, retrieved top-5 pairs), on the text **as
  given** (`CE_TEXT=orig`: case, accents, scripts kept, since the model reads them itself): 3M pairs, 14.6% positive,
  batch 256, lr 2e-5, one pass. ~1,100 pairs/s training, ~5,700 pairs/s scoring on one A100.
- Fold-9 check (records no cross-encoder saw), right S1 as the record's top pick: **bge 0.98277**, stage-1 v4
  probability 0.98232, pair AUC 0.99809. The existing e5 cross-encoders are 0.98237 / 0.98290 (README, older
  candidate set). Alone it is no better on US/India. Mohanish's earlier 500k-pair try (branch `v9fS`,
  `analysis/v9/g10_ce_compare.py`): 0.98173, parked.
- **As an extra feature it helps the judges**: stage-2 copy-free validation at 0.70
  - main stack (v9p): 0.99277 → **0.99293** (+0.00016)
  - France self-trained stack (v9sp): 0.99272 → **0.99287** (+0.00015)
  That is the largest validation gain since house-number peers (+0.00025).
- On France it removes about 5,200 accepted matches (856,815 → 851,596), i.e. it acts mostly on precision, like the veto.

## LLM judge (Qwen2.5-7B-Instruct)

- Apache-2.0, 7.62B (Qwen3-8B is 8.19B, over the limit). Offline, not fine-tuned. `src/llm_judge.py`.
- Only unsure records are judged: best stage-1 p in [0.2, 0.98) or runner-up ≥ 0.2; their top 2 candidates (a third if
  p ≥ 0.2). Score = log P("Yes") − log P("No") at the answer position of the chat prompt (never free text, so always a
  number). Stage-2 features: `llm`, `llm_alt` (best other judged candidate), `llm_gap`; NaN where not judged.
- On v9: 104,064 train records (198,872 pairs) and, with `LLM_COUNTRIES=unseen`, 124,694 France test records
  (192,861 pairs). ~85–100 pairs/s per A100 (compute-bound, ~15k tokens/s).
- **On US/India it adds nothing**: judge AUC on v9's unsure train pairs 0.59 (US 0.5921, India 0.5664; gate check on
  2,000 random pairs 0.5939). Stage-2 validation with vs without LLM features: 0.99293 = 0.99293 (bge main),
  0.99287 = 0.99287 (bge self-trained), 0.99277 = 0.99277 (v9p), 0.99273 vs 0.99272 (v9sp).
- On France it changes ~330 of ~852k accepted matches (b vs c), so its leaderboard effect should be tiny.
- Earlier evidence (v8 pipeline, India held out as a France stand-in, below): India 0.90727 → 0.91111 with LLM
  features, AUC on held-out India pairs 0.858. It helps a country the readers never saw, but on v9 France the
  effect is too small to show.
- US/India always come from a stack without LLM features in files a/c/d: a stack trained with LLM values for
  unsure US/India records would see them missing at test (only France is judged).

## Evidence from the leave-one-country-out runs (v8 pipeline, code on branch `v8-loco-llm`: `loco_eval.py`, `pseudo.py`, `llm_judge.py`, `experiments/loco.pbs`)

India removed from every training step (embedder, shortlist, cross-encoders, judges), then scored like test.
- India never trained on: **0.90727** (crc folds 8–9) vs trained 0.99363; US 0.99192.
- Loss @0.70: missed matches 0.061, distractor merged 0.023, singleton given a record 0.006, wrong S1 0.002.
- True India records: accepted 85.8% (97.8% when trained), rejected by the threshold 10.5% (0.7%), out-ranked 1.8%,
  cut by the shortlist 0.3%, not retrieved 1.6%. Best threshold 0.60: only +0.0005.
- Bi-encoder retrieval for an unseen country: right S1 at rank 1 0.979 → 0.950, in top 20 0.995 → 0.984.
- Pseudo-labelling the **judges** (stage 1) with India's own confident pairs (98.6% / 97.4% accurate): **worse**,
  0.90017 (weight 1.0) and 0.90198 (weight 0.3) vs 0.90699. Self-training works on the readers (Sahaj's v9s), not on
  LightGBM judges.
- Caveat: France's unsure share (12.4% of records with q in 0.3–0.7) is far above India-held-out's, and the only
  proven France gain (the veto) is on precision, so India is an imperfect proxy.

## Next steps, if e beats 0.984742

1. **Self-train bge on France** (Sahaj's `x_ce3.py SELF=` recipe with `CE_BASE`/`CE_INIT` = the fine-tuned bge in
   `amlc_v9x/work/x/ce_bge`, `CE_TEXT=orig`), rebuild the self-trained stack with it, and use it as the veto or as a
   third voter. ~1.5 h GPU + stages.
2. Keep US/India from the bge stack (+0.00016 validation) in the final file.

Cheap checks from a two-reviewer review of other ideas (details in the session notes):
- **Stage-2 context is thinner in training than at test (confirmed in code):** distractors of folds 0–3 are dropped
  from stage-2 rows (the cross-encoders saw them), and W=3.04 weighting fixes the loss but not the context features
  (`e_n`, `e_ge*`, `e_sum`, `e_np_*`), which are computed on the rows present: train context has about 1/3 of test's
  decoy density. Worth measuring (context feature distributions, test vs validation) before any fix.
- France removal/rescue probes on the leaderboard, pre-screened on US/India validation: records adding a learned
  suspicious word ("& Associés", "Et Fils"; the v7w rule was dropped in v8), and veto-rejected records that match
  their S1 exactly (same house number, same address, same core name).
- Trim France's candidate list (10.1 per S1 vs US 6.6) without touching accepted pairs.
- Rejected: synthetic negatives, Fellegi–Sunter EM, a listwise LLM prompt, per-S1 expected-F rules (already lost in
  v3), LoRA on the LLM tonight.

## How to reproduce (PADUM)

Code copy on PADUM: `~/scratch/ber_v9x` (this branch). Inputs: the finished v9p_frand work folder
`/scratch/scai/mtech/aib262045/work`, read through links only; every new file goes to `~/scratch/amlc_v9x/work`.

```bash
# everything: bge (GPU 0), LLM (GPU 1), stage 1/2 per variant, files 0/b/c/a, validation; resumable (.ok markers)
qsub -l select=1:ncpus=16:ngpus=2:mem=300gb ~/scratch/ber_v9x/ber/experiments/v9x.pbs
#   PHASE=gpu  only bge + LLM (for a job with few CPUs)   PHASE=llm  only the LLM (a second job; shards are shared)
# the two files built afterwards from the same stacks
qsub ~/scratch/ber_v9x/ber/experiments/v9x_bge.pbs     # e: bge everywhere, no LLM
qsub ~/scratch/ber_v9x/ber/experiments/v9x_final.pbs   # d: bge everywhere + LLM on France
```
Needs the models offline (downloaded with the proxy on the login node):
`hf download BAAI/bge-reranker-v2-m3 --local-dir /scratch/scai/mtech/aib262045/models/bge-reranker-v2-m3` and
`Qwen/Qwen2.5-7B-Instruct` likewise. Timings on A100s: bge training 50 min, scoring 20.5M pairs ~1.2 h, stage 1
per stack ~25 min, stage 2 per stack ~20 min with 10 CPUs, LLM 393k pairs ~70 min on one GPU.

Code changes on top of `ber-pipeline`:
- `src/x_ce3.py`: `CE_TEXT=orig` (text as given) for a multilingual pretrained reranker (`CE_BASE=`).
- `src/llm_judge.py`: select / score / merge, `LLM_COUNTRIES=unseen`, fail-fast gate (`LLM_GATE_AUC`, 0.55 in the
  job: it stops a broken judge, AUC ~0.5, not a weak one).
- `src/x_stage_multi.py`: `LLM=1` adds `llm`, `llm_alt`, `llm_gap` to stage 2; `S2TAG` keeps a stage-2 variant's
  files apart.
- `experiments/v9x.pbs`, `v9x_bge.pbs`, `v9x_final.pbs`, `runner.pbs` (a job that holds its GPUs and runs scripts
  dropped into a folder).

PADUM gotchas learnt: 2 running + 2 waiting jobs per user, 2 GPUs per job; qsub copies the job script at submission
(editing it later does not change a queued job); advance reservations (`pbs_rsub`) are confirmed but unusable (the
budget hook rejects jobs in reservation queues).
