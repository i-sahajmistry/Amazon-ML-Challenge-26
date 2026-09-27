# Experiments on top of the best file (Sarvesh, 27 Sep)

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
| E5 | E1 with the generic France fixes (`rule_fr` + `x_ddfix`) instead of the variant's lists | pending | | compares generic vs variant lists | compliance check |
| E7 | US / India from the **mean** of the bge and main stacks | 2,255,059 / 2,751,092 / 859,338 | | validation 0.99279 < bge alone 0.99283 | ❌ **does not work**: averaging is worse than bge alone (file `138c24f2` built, PASS, not to be used) |

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

## Earlier: candidate-set size (`cand_trim.py`, on the v9 shortlist = the same 12.76M pairs)
Raising the shortlist cut-off from P ≥ 0.001: 0.003 → −8% pairs, cuts 782 of 5.88M accepted matches; 0.005 → −11%,
1,421; France 10.1 → 8.2 / 7.5 pairs per S1. Needs a full rerun from stage 1 (candidate_pairs must be what the model
scored). Sahaj's `tight_dd` (6.27 per S1) lost France matches, so a tighter shortlist is not used.
