# variant_frB: two recall fixes on variant_v10seed_dd

Branch `variant-frB`, built on `final-dd` 187946d. It adds `src/x_recall.py`, `variant_frB/frB.sh` and Sahaj's
`variant_v10seed_dd/` scripts (from v11), which build the best leaderboard file so far (0.990349).

## Why

Where the score is still lost (US / India validation, v10 + LLM judge, F0.5 0.99282):
- 88% of the loss is missed matches, and 80–90% of those are records with no address.
- Most of them are name twins: a namesake ranker picks the right S1 38% of the time (random 24%), so they cannot be
  resolved from the data.

France (test, no labels) accepts 3.31 records per S1 against 3.40 for US / India. The gap is largest for no-address
records whose name matches a single S1:

| No-address records, name matches a single S1 | US / India validation: true / accepted | France: accepted |
|---|---|---|
| exact normalised name | 96–98% / 95–98% | **84%** |
| same core name | 76–78% / 68–70% | **50%** |

## The two fixes (`x_recall.py`)

1. **Empty S1s (all countries).** An S1 with no accepted record takes its best candidate if its score is ≥ 0.40.
   - On US / India validation, with labels, this gains +0.00009 (US +0.00013, India +0.00004).
   - France has 2–3 times more such S1s per S1.
   - Test: 1,415 records (US 546, India 430, France 439).
2. **No-address records, countries without training labels.** A rejected record is restored when all four hold:
   - it has no address;
   - its normalised name, or core name, equals its S1's;
   - no other S1 of the country has that name;
   - the LLM judge accepts it (margin > 0).

   This gives 5,445 France records (1,355 of them had been rejected by the self-training vetoes). Why they are likely
   true matches (estimated 85–92%; a restore pays above ~70%):
   - US / India labels: 96–98% / 76–78% true for these groups.
   - The judge says no to only 4–5% of France's rejected ones.
   - France's decoys at the shifted number keep the S1's exact name half as often as US decoys (0.24 vs 0.48 per S1),
     so these are not decoys that lost their address. France has more of these records because its true matches keep
     the exact name more often (65% vs 54% at the S1's own number).
   - Where labels exist the rejected ones are mostly wrong (53–62% / 29–32% true), so, like the same-address fixes,
     this applies only to the countries without labels (`common.unlabelled()`).

Checked and not used:
- **The judge's "yes" alone:** rejected no-address records it accepts are only 44% true on US / India validation.
- **Other rejected France groups:** by the judge's estimate they are mostly decoys (addressed twins ~4% true,
  shifted-number records ~0%).
- **The LLM veto of branch v12b on top of the variant:** it would mostly remove true suffix records.

## Output

- `matching_results.tsv` md5 73b4a5eae26e6139dcbd396d241093d7; `candidate_pairs.tsv` md5 1a8b4f5c (unchanged).
- Validator (`--check-ids`): PASS.
- Accepted: US 2,255,704; India 2,751,440; France 865,174 (variant 859,338).
- Expected leaderboard: about 0.9905–0.9906 (variant 0.990349).

## Files

- `src/x_recall.py` (new): builds the two restore lists; `val` reproduces the empty-S1 rule's validation gain.
- `variant_frB/frB.sh` (new): the steps on the variant's files.
- `variant_v10seed_dd/` (from v11): Sahaj's scripts for the variant.
- `README.md`: variant_frB section, pipeline step 11 and leaderboard rows. This file is new.
