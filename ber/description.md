# dd-generic: v10_fr3_llm_dd built directly from data, with no country names

Branch `dd-generic`, built on `v11` 2e1cd0f (Sahaj's v10_fr3_llm_dd, leaderboard 0.990282). The output file is unchanged
(byte for byte); only the way it is built changes.

## Why

v10_fr3_llm_dd's France fixes were found by reviewing France test records and probing the leaderboard. As pushed:
- the restore candidates were read from `rule1_added.parquet` in another folder, written by an analysis script;
- the word lists were built from `fam_France.parquet`, which the hand-review scripts write;
- the vetoes and lists named France.

The problem statement asks to treat country as an open set, and to be able to regenerate the output from the code
folder alone. This branch builds the same decisions from label-free statistics of the test records, and selects the
countries from the data.

## Changes

- `common.py`: `unlabelled()`, the test countries without a single labelled training record, found from the data
  (France here); `countries(spec)` expands the token `unlabelled` in a country list.
- `x_final.py`: `FROM=` and `THR_C=` take `unlabelled:…` as well as a country name.
- `x_llm.py`: `COUNTRY=unlabelled`.
- `x_chain.py`: plans `v9s2`, `v10s3` and `llm2` use `unlabelled` instead of France. The new plan `dd` runs the whole fix.
- `rule_fr.py` (new here, from the structure session): the structural same-address rule; its default is the
  countries without labels.
- `x_ddfix.py` (new): the restore / reject lists, directly:
  1. every record's best S1 and its decision after the three self-training vetoes (`x/test_q_v10fr3l`);
  2. its position against the S1's first house number (SAME, D = the S1's shifted decoy cluster, OTHER, NONE) and the
     words it adds to the S1's core name; nD = the S1's records at other numbers;
  3. per word added at SAME in at least 300 records: TV, the total variation distance between the nD spread of
     those records and that of unedited SAME records. true-like: TV ≤ 0.10 and SAME / D ≥ 0.1; decoy-like: TV ≥ 0.20;
  4. restore the rejected records that `rule_fr.py` accepts and that add only true-like words (or none); reject the
     accepted SAME records that add a decoy-like word.

## Checks

`python x_chain.py dd`, run from scratch:

| Output | Result |
|---|---|
| vetoed France scores (`x/test_q_v10fr3l`) | identical to v11's |
| `rule_fr.py` output (`x/test_q_rule_v10plw`) | identical to v11's; 54,774 records not accepted = rule1's additions |
| word lists | 44 true-like, 41 decoy-like (the same words as v11) |
| restore / reject | 32,439 / 11,655 |
| `matching_results.tsv` | md5 75f68b10b36326698baff1a168d3cdaa, the leaderboard file (0.990282) |
| `validate_submission.py --check-ids` | PASS |

The lists hardly depend on the thresholds, because the words' TVs fall in separated groups:
- true-match noise ≤ 0.10: groupe 0.032, développement 0.021, fils 0.008;
- pure shifted-decoy words 0.15–0.19;
- swapped type words ≥ 0.22: club 0.306, école 0.289.

TV_TRUE 0.06–0.15 moves the restore pool by ±2%. TV_DEC 0.15 or 0.20 gives the same rejects; 0.25 drops 7 borderline
words (−11% rejects).

`python x_ddfix.py check` applies the same fixes to labelled US / India validation, each country with its own lists:

| Country | Fix | Rows | True share | F0.5 |
|---|---|---:|---:|---:|
| India | restore | 638 | 4.2% | −0.0016 |
| India | reject | 2,922 | 99.97% | −0.0024 |
| US | restore | 179 | 24.0% | −0.0002 |
| US | reject | 0 | – | 0 |

Where labels exist the models already get these records right, so the fixes are applied only to the countries
without labels. There, the models' word priors come from US / India text and do not carry over.

## Still open

- `rule_fr.py` reads `x/wstat_words_test_<country>_0.parquet` (the house-number word statistic). `wstat.py` is not
  in this branch yet.
- The thresholds 0.10 / 0.20 / 300 are set by hand (see the sweep above).
- `x_wordsame.py`, `x_families.py`, `x_frfix.py` and `x_suf7ng.py` are the hand-review analysis. The `dd` plan does
  not use them.
