# E16fr3-sn: E16fr3 + two France recall lists (same name at another address; no address, same core name)

Branch `E16fr3-sn`, built on `final-dd` fefa59e (E16fr3, leaderboard 0.990807). One new restore list,
`restore_samename`, in `x_recall.py`. It applies only to the countries without training labels (France here).
US / India are unchanged.

## The rule

A record is restored to its best S1 when all of these hold:
- it is rejected after the vetoes and the same-address fixes;
- its normalised name equals the S1's;
- it has a house number, and so does the S1, but the numbers differ;
- it is the only record of that S1 at that number;
- its number is not 1–13 above the S1's number (the decoy shift: 76–83% of the decoy-cluster records sit there);
- the LLM judge accepts it (margin > 0).

Candidates outside the judge's band (stage-2 q ≤ 0.01) have no judge score. `x_recall.py` writes them to
`x/sn_rows.parquet`, `ROWS=sn_rows x_llm.py rows` scores them, and `x_recall.py` is run again.

## Why

- The data generator moves a true match to another address in every country: US / India 0.05 such records per S1,
  France 0.034.
- **US / India validation (labels):** these records belong to their S1 97–98% of the time when no other S1 has the
  name. With 1 / 2 / 3+ namesake S1s the share is 93% / 87–89% / 68–78%. The rest are distractors; they almost never
  (≤ 0.3%) belong to another namesake S1. The models accept 47–97% of them.
- **France (test, E16fr3):** accepts 62% / 24% / 14% / 5% of them (by namesake count). France's names come from a
  small vocabulary ("Lille Club", "Nantes Pharmacie SARL"), and models trained on US / India read a common name at
  another address as a different business.
- **The judge:** France's rejected records of this kind get a "yes" 69–81% of the time. US / India's rejected ones
  (1–6% true) get 17–44%. With the judge's US / India calibration (P(yes | true) ≈ 0.9, P(yes | not true) ≈ 0.2),
  ~70–85% of France's rejected ones are true, and ~92% of those the judge accepts.
- A sample of 40 restored pairs reads as the same business: the same name up to accents, case and legal-form
  punctuation, the same city, often the same street at another number ("60 Route de Vannes" ← "58 Rte De Vannes").

## Checks

| Run | restore_samename | France accepted | matching md5 | validator |
|---|---|---|---|---|
| original files (E16fr3's) | 4,646 (7,028 candidates, judge yes 4,646 / no 2,382) | 866,789 (+4,637) | `519bc93e` | PASS |
| Sahaj's v10-only rerun (`amlc_v10only`) | 3,412 | 864,359 | `0a6f432e` | PASS |

- Without the new list, the code rebuilds E16fr3 byte for byte from its files (`e93605ad`); `restore_empty` and
  `restore_nafr` are identical to the shipped ones.
- The judge adapter is the shipped one (md5 8a25ff96). Re-scoring 1,000 already-scored France rows gives a median
  |difference| of 0.000 and 99.1% sign agreement.
- Candidates must be rejected by the France chain, which shifts between GPU reruns, so the list moves more than the
  rest of the file (3,412 vs 4,646).
- candidate_pairs.tsv is unchanged (`1a8b4f5c`).
- Expected leaderboard change: about +0.0001 if ~90% of the restores are true; break-even is ~70%.

## Files

- `src/x_recall.py`: `restore_samename` (plus `x/sn_rows.parquet` for the judge).
- `src/x_llm.py`: new `rows` mode (`ROWS=<file> [SHARD=i/n] python x_llm.py rows`).
- `reproduce.sh`: `llm_sn` (lane C) and `x_recall2` before `final`; `final` adds `restore_samename`.

## Second list: restore_nacore (no address, same core name)

A rejected record with no address is restored to its best S1 when:
- its core name (legal form dropped) equals the S1's, but the full name differs ("Helena Club SAS" for "Helena Club");
- no other S1 of the country has that core name;
- the LLM judge accepts it.

Evidence:
- **US / India validation:** such records are 98.0% (US) / 98.4% (India) true, and the models accept 97.5–98.5%. When
  another S1 shares the core name, only 44–48% are true (half belong to the other S1), so those are left out.
- **France (E16fr3):** accepts 72.5% of the records without a namesake. If France's records of this kind are as right
  as US / India's (98%), at least ~93% of the rejected ones are true. The judge says yes to 95% of them.
- A sample of 40 reads as the same business with the address dropped and the legal form changed.

Result: 1,078 restores (candidates 1,125). Combined with `restore_samename`: France 867,859 (+5,707 over E16fr3),
matching md5 `fdea1f55`, validator PASS, candidate_pairs unchanged (`1a8b4f5c`). Expected leaderboard change of this
list: about +0.00004.
On Sahaj's v10-only rerun (`amlc_v10only`, sandbox `~/scratch/v10chk`): `restore_nacore` 1,059 (original files 1,078),
`restore_samename` 3,412; France 865,410; matching md5 `8a6f769d`; validator PASS.

## Expected leaderboard change

Method as Sahaj's `delta.py`. For each S1 the lists touch:
- E16fr3's other accepted records are taken as true;
- each added record is true with probability p;
- an S1 that was empty and gets only wrong records is truly empty half the time.

The per-S1 F0.5 changes are summed over all 1,732,544 S1s (Monte Carlo, 60 draws).

| p (share of added records that are right) | restore_samename (4,646) | restore_nacore (1,078) | combined | leaderboard (E16fr3 0.990807) |
|---|---|---|---|---|
| 0.60 | −0.00005 | −0.00002 | −0.00007 | 0.99074 |
| 0.70 | +0.00004 | +0.00000 | +0.00004 | 0.99085 |
| 0.80 | +0.00013 | +0.00002 | +0.00016 | 0.99096 |
| **0.90** | **+0.00023** | **+0.00004** | **+0.00027** | **0.99108** |
| 0.95 | +0.00027 | +0.00005 | +0.00033 | 0.99113 |
| 1.00 | +0.00032 | +0.00006 | +0.00038 | 0.99119 |

- **Break-even:** p ≈ 0.67 (restore_samename), 0.70 (restore_nacore).
- **restore_nacore, p ≈ 0.93–0.98:** the same kind is 98% true in US / India, and France rejects 27.5% of it.
- **restore_samename, p ≈ 0.85–0.94:**
  - US / India true shares, weighted by France's namesake counts, give ~78% before the judge; its yes lifts that to
    ~90–94%.
  - The risk is France's generic names: 75% of the candidates have 3+ namesake S1s, where US / India are 68–78% true.
- **Split noise:** sd of the public-minus-private gain ~2.3e-5 (30% public share), so the leaderboard reads the result
  clearly.

## Third list: restore_llmveto (Sarvesh's E49, France-safe part)

France's decision is the minimum of the main stack + judge and three self-training stacks. The vetoes remove 25,686
records the main stack + judge accepted. E49 restores those whose judge margin is at least m.

- **US / India validation (Sarvesh, `exp36.py`):** the same chain vetoes 2,259 accepted rows, 83.7% of them true.
  Restoring those at margin ≥ 4 gives rows that are 95% true, +0.00019.
- **France:** the vetoes were confirmed on the leaderboard (+0.0033 over three rounds; per removed record, round 1 took
  ~85% decoys and round 3 ~45%), so France's vetoed set is decoy-rich.
- **Profile of the 2,469 France restores at margin ≥ 4:**
  - no house number, same street: 843;
  - the S1's number, suffix or neutral words: 576 (census 3–11% decoys);
  - another house number: 772 (the decoy cluster, D, 323; other numbers 449);
  - decoy-like, unedited or word-dropping same-address records: 278 (census decoy-rich).

`restore_llmveto` keeps the first two groups: 1,418 restores. That is exactly Sarvesh's `llmveto4n` (number-compatible,
1,696) minus the 278, and it overlaps neither of the other two lists. Expected: +0.00007 at 90% right.

**Our idea 1 under E41 / E42.** On US / India the *rejected* same-name records at another house number are 0.2–2%
true, and the dd analysis found France's true matches rarely change number. So France's rejections of
`restore_samename`'s candidates may be right. The only evidence the other way is the judge's yes-rate, and our cut
(margin > 0, median 1.75) is weak. It is therefore uploaded as a separate step (E16fr3cv → E16fr3sncv), so the
leaderboard measures it.

`reproduce.sh`: x_recall / x_recall2 take the France base `_v10plw` (VBASE); final adds `restore_llmveto`.
