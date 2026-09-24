# EDA — ML Challenge 2026: Business Entity Resolution

Exploratory analysis of `ML Hackathon dataset/student_resource/`. Every number below was
measured on the actual files. Scripts are in [eda/](eda/) and cache parquet copies in
`eda/cache/`. Rerun with `.venv/bin/python eda/<script>.py` from inside `eda/`.

## 1. The task in one paragraph

Source 1 (S1) is a clean, deduplicated reference list of businesses. For each S1 record,
find **every** record in Source 2 (S2) and Source 3 (S3) that is the same real business.
The score is **F0.5, macro-averaged per S1 entity**, so precision counts twice as much as
recall. A singleton (an entity with no true matches) scores 1.0 for an empty prediction and
0.0 for any prediction. Two output files are required: `matching_results.tsv` (scored) and
`candidate_pairs.tsv` (the exact candidate set fed to the matcher; the organisers use it to
audit blocking). Other rules: the final model must be MIT or Apache-2.0 licensed with at most
8B parameters, and no external lookups are allowed (no geocoding, registries or APIs).

## 2. Files and sizes

| File | Rows | US | India | France |
|---|---:|---:|---:|---:|
| train_source1 | 2,206,821 | 1,323,633 | 883,188 | – |
| train_source2 | 5,034,616 | 3,016,817 | 2,017,799 | – |
| train_source3 | 5,285,603 | 3,170,056 | 2,115,547 | – |
| train_ground_truth | 2,206,821 | | | |
| test_source1 | 1,732,544 | 663,106 | 809,986 | **259,452** |
| test_source2 | 4,887,273 | 1,871,330 | 2,312,565 | **703,378** |
| test_source3 | 5,082,316 | 1,945,701 | 2,405,000 | **731,615** |

- Every file has 4 columns: `entity_id, business_name, business_address, country`. The ID
  prefix (`S1-`/`S2-`/`S3-`) identifies the source.
- **Read with `quote_char=False`** (pyarrow) or `quoting=csv.QUOTE_NONE` (pandas). Names
  contain quote characters. With quoting disabled, every file's row count matches `wc -l`
  minus the header.
- There are no duplicate entity IDs and no nulls. The only empty field is
  `business_address` in S2/S3 (about 3%).
- Everything fits in RAM. The full train set as parquet loads in a few seconds on a 32 GB
  machine.

## 3. Ground-truth structure (train)

- 7,638,365 true pairs: 3,693,619 to S2 and 3,944,746 to S3.
- **Singletons: 5.58%** of S1 (123,247 entities), the same rate for US and India.
- Matches per S1 entity:

  | # matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | S1 entities | 123k | 119k | 375k | **531k** | 484k | 322k | 165k | 64k | 19k | 4.2k | 534 | 37 |

  The mean is 3.46 (3.67 excluding singletons). Per source, an entity has 0–5 S2 matches
  and 0–6 S3 matches, so **the same source often contains several duplicates of one
  business** (for example, 5 S2 records for one S1).
- **Clusters don't overlap:** each S2/S3 record belongs to at most one S1. This means a
  one-to-many assignment constraint can be applied: give each S2/S3 record to its single
  best S1.
- **Country always agrees** between matched records (100.0%). Blocking within a country
  is lossless.
- **Distractors:** 26% of S2 records and 25% of S3 records match nothing (2,681,854 in
  total, about 1.2 per S1). They look like ordinary records, not obvious junk.

## 4. What each source looks like

**S1 (clean reference).** Title case, no empty fields, Latin script only, no noise tokens.
The S1 address order varies (`IA, Iowa City, 1064 Newton Rd, Unit 11`). Business names
repeat heavily: **30% of S1 names are exact duplicates of another S1 name** (for example,
`Cardiology Specialists LLC` exists in many cities).

**S2 and S3 (noisy).** Both have the same families of noise, with different styles:

| Pattern | S2 | S3 | Notes |
|---|---|---|---|
| Address in ALL CAPS | US 93%, India 27% | ~3% | S2 US addresses: `607 VIRGINIA ST, TERRELL, TX` |
| Name in ALL CAPS | US 22%, India 38% | US 3%, India 14% | |
| Name in an Indic script (India) | **23%** | **13%** | Devanagari 13%/7.5%, then Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi |
| Address in an Indic script (India) | 24% | 22% | often only the state is native (`ಕರ್ನಾಟಕ`, `తెలంగాణ`) |
| Empty address | 3.4% | 3.3% | |
| `null` / `NULL` / `N/A` tokens in the address | 2.9–3.9% | 2.7–3.7% | |
| Name is a web domain | 2.6–3.9% | 2.8–3.6% | `kimb1eolvas.com`, `bfprivate.com` (acronym), `familybrighthealth.com` |
| Junk prefix | 0.9–1.3% | 0.9–1.3% | `>>`, `...`, `***`, `--` in roughly equal shares |
| Digit/letter swaps | ~1.5% | ~1.5% | `Cardi0logy`, `5afe`, `D0mbivli` |
| Double spaces | ~11% | ~11% | |
| Bracketed suffix | ~6% | ~6% | `[INC]`, `(PC)`, `[Club-EURL]` |
| PO BOX / PMB added (US) | 3.3% | 3.1% | |

Other patterns seen in the samples:

- **Name changes:** a legal suffix swapped, dropped or expanded (`Inc`↔`Incorporated`,
  `Pvt Ltd`↔`Private Limited`); words duplicated (`Family Family Bright Health`); words
  reordered (`India Trust Charitable`); typos (`Sepciialsist`); a generic word added
  (`Dr`, `Center`, `Services`); and occasionally a **completely unrelated made-up name**
  (`Iriecto`, `Evohalovio`) that links to the entity only through its address.
- **Mixed scripts:** some names mix Latin and Indic script (`Al Tech प्राइवेट लिमिटेड`).
- **Address changes:** street-type abbreviations (Rd/Road, St/Street, Pkwy); state name
  vs code (TX/Texas, MH/Maharashtra, OD/Orissa/Odisha); components reordered; components
  dropped (keeping only `Flat 203, Hyderabad`); a changed or truncated house number
  (`607`→`60`, `405`→`05`); a wrong district or state (`Jalna`, `Andhra Pradesh` for a
  Telangana address); city-name variants (`Howrah`/`Haora`).
- **Postcodes are rare:** US ZIP appears in about 10% of addresses, Indian PIN in about
  1% (0.3% in S1), French code postal in 0.5%. **Postcodes can't serve as a main
  blocking key.**
- **Landmarks:** Indian addresses carry landmark text (`Near PNB Bank`, `Opp ...`) in
  8–13% of records, including in S1.

## 5. How matched pairs differ from non-matches

Based on 200k sampled true pairs, compared with three kinds of same-country negatives:
*random*, *shared-token* (shares the longest name token with the S1 record) and
*same-name* (a different entity with an identical normalised name).
Normalisation: NFKD, strip accents, lowercase, punctuation → space. Similarity is
rapidfuzz `token_set_ratio` (0–100).

| Signal | True pairs | US | India | Random neg | Shared-token neg | Same-name neg |
|---|---|---|---|---|---|---|
| Normalised name exactly equal | 25.7% | 30.4% | 18.8% | 0% | 0.6% | 100% |
| Share ≥1 name token (excl. legal words) | 84.8% | 91.2% | 75.3% | 0.9% | 100% | 100% |
| Share ≥1 address token | 95.6% | | | 21% | 23% | 25% |
| Share ≥1 address number | 79.8% | 78.0% | 82.5% | 2.7% | 2.9% | 3.2% |
| Name in Latin script | 92.7% | 100% | 81.7% | | | |

Similarity quantiles (p10 / p25 / p50):

| | Name TSR | Address TSR |
|---|---|---|
| True pairs, all | 47 / 88 / 100 | 79 / 87 / 94 |
| True pairs, Latin name | 77 / 92 / 100 | 79 / 87 / 94 |
| True pairs, Indic name | **9 / 10 / 11** | 82 / 90 / 96 |
| Shared-token negatives | 55 / 60 / 67 | 29 / 33 / 37 |
| Same-name negatives | 100 / 100 / 100 | **30 / 33 / 37** |

**Takeaways**

1. **Name alone is not enough.** 39% of sampled S1 entities have a non-matching S2/S3
   record with the *identical* normalised name. With a name threshold of 80, 22% of
   shared-token negatives still pass.
2. **The address is the strongest signal.** True pairs have address TSR ≥ 79 in 90% of
   cases, while every negative type sits around 30–37. Name ≥ 60 **and** address ≥ 60
   keeps 82.5% of true pairs and lets through only 1.5% of shared-token negatives and
   1.1% of same-name negatives. This simple two-signal rule is a strong baseline, and a
   learned model on these features should do much better.
3. **About 7% of true pairs have an Indic-script name** (18% of Indian pairs). For these,
   name similarity is effectively zero unless the name is transliterated. The address
   (mostly still Latin) carries the match. A transliteration step, or a multilingual
   character or embedding model, is worth adding.
4. 4.4% of true pairs have an **empty** S2/S3 address, so the match depends on the name
   alone. Matching these risks precision, so require a very high name score.
5. 2.5% of true pairs share **neither** a name token **nor** an address number. These
   are the made-up names, domains and truncated numbers. They are the hard ceiling for
   token-based blocking.

## 6. Blocking-key recall vs candidate volume

Measured on 150k sampled train S1 entities (519k true pairs). Keys are matched within
country against all ~10.3M train S2+S3 records. "Candidates per S1" adds up the sizes of
the blocks each S1 record falls into, before any ranking or deduplication.

| Blocking key (within country) | Pair recall | Candidates per S1, mean | median | Largest block |
|---|---:|---:|---:|---:|
| Longest name token | 70.7% | 12,825 | 4,475 | 134k |
| Any name token (excl. legal words) | 85.0% | 48,120 | 30,559 | 317k |
| Name, first 3 chars | 77.7% | 12,136 | 8,207 | 102k |
| Any address number | 80.0% | 121,467 | 8,976 | 566k |
| Longest address number | 72.1% | 7,741 | 2,091 | 94k |
| Name token **or** address number (from §5) | **97.5%** | — | — | — |

**Takeaways**

- No single exact key works. The keys with good recall produce thousands to hundreds of
  thousands of candidates per entity, because business-name words (`Cardiology`, `Shree`,
  `Global`) and house numbers repeat heavily.
- A **union of keys** reaches about 97.5% pair recall but is far too large to score
  exhaustively. Candidates therefore need **ranking inside blocks**, e.g. TF-IDF char
  n-gram cosine or ANN top-k (FAISS or sparse kNN) over name + address, keeping the top
  ~20–50 per S1.
- The remaining ~2.5% (unrelated names, web domains, truncated numbers) is only reachable
  by address-text similarity. That's also the key route for Indic-script names.

**What distractors look like:** only **5.6%** of unmatched S2/S3 records have a normalised
name that appears verbatim in S1, against 29% of matched records. Distractors are mostly
*other* businesses in the same cities (for example `Vandenburg Services`,
`Lotech Group`), not near-copies of S1 records. They carry the same noise (caps, `null`,
misspelled cities), so noise alone can't identify a record as a distractor.

## 7. Train → test shift (important)

| | Train | Test |
|---|---|---|
| Country mix of S1 | US 60%, India 40% | India 47%, US 38%, **France 15%** |
| S2+S3 records per S1 | **4.68** (US 4.67, India 4.68) | **5.75** (US 5.76, India 5.82, France 5.53) |

- **France is unseen.** Its records are French-language: legal forms `SARL`, `SAS`,
  `S.A.S`, `EURL`, `EI`, `Ets`/`Établissements`, `& Fils`, `& Cie`; street types `Rue`/`R`,
  `Avenue`/`Ave`, `Impasse`/`IMP.`, `Boulevard`/`BOULEVRD`, `Chemin`, `Cours`; `Nº`;
  accents dropped (`Mérignac`/`MERIGNAC`); the region vs département
  (`Nouvelle-Aquitaine` vs `Gironde`, `Pays de la Loire` vs `Loire-Atlantique`,
  `Hauts-de-France` vs `Nord`). Records cluster in a few cities (Bordeaux, Lille, Nantes,
  Dunkerque, Saint-Nazaire, Mérignac, Pessac, Tourcoing), so city-level blocks will be
  large. Noise styles match the other countries: S2 addresses ~32% uppercase, junk,
  domains and bracketed suffixes at similar rates. France has no Indic script and almost
  no postcodes. **Build normalisation and features from country-agnostic signals.** Keep
  any per-country abbreviation tables open-ended, with a French table added. Don't
  one-hot `country`.
- **Test has about 23% more S2/S3 records per S1.** Either the average match count is
  higher or (more likely) there are more distractors: at the train rate of 3.46 matches,
  test would have about 2.3 distractors per S1 instead of 1.2. **Expect more candidate
  false positives on test than on validation**, so set thresholds on the conservative
  side. Consider validating on a train split with extra distractors injected.

## 8. Modelling implications (summary)

- **Pipeline:** block within country → pair features → classifier → per-S1 threshold →
  enforce "each S2/S3 record goes to at most one S1" → write both TSVs.
- **Normalise:** Unicode NFKD plus accent stripping, lowercase, punctuation to space;
  strip junk prefixes, bracketed suffixes and `null`/`N/A` tokens; collapse repeated words;
  map digit swaps (0→o, 1→l, 5→s) inside words; canonicalise legal suffixes and street
  types (US, India, France); map state codes to state names; split domains into words.
- **Blocking:** use a union of keys (name tokens, address numbers, char n-gram / TF-IDF
  nearest neighbours on name and address) so the Indic-name and made-up-name cases survive.
  Handle Indic-script names by transliterating to Latin, or by blocking on the address only.
- **Features:** name and address similarity scores (token set, ratio, Jaro-Winkler, TF-IDF
  cosine), number-set overlap (house or plot numbers), whether the name is non-Latin,
  whether the address is empty, whether the name is a domain, and IDF/rarity of shared
  tokens (a name like "Cardiology Specialists" is common, so a match on it counts for
  less). Also the rank of this candidate among the S1's candidates, and whether the
  record's best S1 is this one.
- **Decisions:** F0.5 macro rewards leaving uncertain entities empty. Tune the threshold on
  macro F0.5 (not pair-level F1) with a held-out train split. Singletons are 5.6% and each
  one wrongly matched costs a full 1.0.
