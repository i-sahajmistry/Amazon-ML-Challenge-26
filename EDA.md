# EDA — ML Challenge 2026: Business Entity Resolution

Exploratory analysis of `student_resource/`. Every number below was measured on the actual
files. The analysis was done in two passes:

- **Quick pass** ([eda/](eda/), laptop): profiling, ground truth, noise patterns and
  similarity on samples, exact-key blocking.
- **Deep pass** ([eda/deep/](eda/deep/), PADUM, 24 CPUs + 1 A100): noise taxonomy and
  similarity over **all 7.6M true pairs**, entity-level difficulty, vocabulary inventories,
  exact TF-IDF nearest-neighbour retrieval on GPU (train and test), F0.5 of simple
  decision rules, and a measured train→test shift.

How to rerun either pass is in the [Appendix](#appendix-reproducing-the-numbers).

## Key findings

1. **The address decides a match, not the name.** 39% of S1 entities share their
   normalised name with other S1 entities (up to 253 entities called "Primary Care
   Group"). A different business with the identical name has address similarity around
   33, against about 94 for a true match (§6).
2. **Structure to exploit:** each S2/S3 record belongs to at most one S1 entity; country
   always agrees; 5.6% of S1 entities are singletons; about 26% of S2/S3 records are
   distractors that match nothing (§3).
3. **Most entities have at least one hard match.** Only 43% of S1 entities have *all*
   their matches with name and address similarity both ≥ 80. 15% have a match with an
   empty address, 15% a web-domain name, 13% an Indic-script name and 6% a made-up name
   (§7). With the score macro-averaged per entity, recall on these hard matches is what
   separates good pipelines.
4. **Transliteration fixes Indic names.** Median name similarity for Indic-script records
   goes from about 9 (raw) to about 59 after `anyascii` transliteration (Tamil lowest at
   43). Legal words then need their transliterated forms (`praivet`, `limitet`, `elelpi`)
   (§5, §8).
5. **Only 0.06% of true pairs are weak on both name and address**, so almost every true
   pair has at least one strong signal a matcher can use. When the name fails, the
   address is usually near-identical (§6).
6. **TF-IDF nearest-neighbour search on name + address is a strong blocker.** In the top
   20 candidates it finds 98.1% of true pairs for the US and 90.3% for India. A perfect
   matcher on those candidates would score macro F0.5 0.994 / 0.963. Indian recall is
   held back by empty addresses (36% found in the top 20) and Indic names (77%) (§9.2).
7. **Simple rules plateau around macro F0.5 0.72–0.74**, far below that ceiling. The
   mutual-best rule (the record's best S1 is this entity) keeps 98% of true candidates
   and only 6.5% of false ones, but it can't reject **distractors**, and they are the
   core difficulty (§10).
8. **Test has about 39% distractors, against 26% in train** (US and India, estimated
   from score distributions; the method recovers the train rate exactly). This accounts
   for the 23% more records per entity in test. Validate on a train split with extra
   distractors injected, or thresholds tuned on train will over-match on test (§11).

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
  business**. The counts from S2 and S3 are close to independent: 13% of entities have
  no S2 match and 12% have no S3 match.
- **Within-source duplicates are near copies of each other.** For 2.35M pairs of two
  S2 (or two S3) records of the same entity, the median name and address similarity are
  both 97; 6.5% are identical after normalisation. Once one record is matched confidently,
  its siblings in the same source are easy to pick up.
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
repeat heavily: **39% of S1 entities share their normalised name with another S1 entity**.
There are 1,340,680 unique names, 100,854 names used twice and 11,943 names used 10 or
more times. The biggest groups are generic medical names (`Primary Care Group` × 253,
`Ear Nose Throat Group` × 251, `Pediatric Group` × 222). Only 4.5% of same-name groups
stay within one state.

**S2 and S3 (noisy).** Both have the same families of noise, with different styles
(record-level rates; §5 has the same patterns measured on matched pairs):

| Pattern | S2 | S3 | Notes |
|---|---|---|---|
| Address in ALL CAPS | US 93%, India 27% | ~3% | S2 US addresses: `607 VIRGINIA ST, TERRELL, TX` |
| Name in ALL CAPS | US 22%, India 38% | US 3%, India 14% | |
| Name in an Indic script (India) | **23%** | **13%** | Devanagari 13%/7.4%, then Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi |
| Address in an Indic script (India) | 24% | 22% | often only the state is native (`ಕರ್ನಾಟಕ`, `తెలంగాణ`) |
| Empty address | 3.4% | 3.3% | |
| `null` / `NULL` / `N/A` tokens in the address | 2.9–3.9% | 2.7–3.7% | |
| Name is a web domain | 2.6–3.9% | 2.8–3.6% | `kimb1eolvas.com`, `bfprivate.com` (acronym), `familybrighthealth.com` |
| Junk prefix | 0.9–1.3% | 0.9–1.3% | `>>`, `...`, `***`, `--` in roughly equal shares |
| Digit/letter swaps | ~1.5% | ~1.5% | `Cardi0logy`, `5afe`, `D0mbivli` |
| Double spaces | ~11% | ~11% | |
| Bracketed suffix | ~6% | ~6% | `[INC]`, `(PC)`, `[Club-EURL]` |
| PO BOX / PMB added (US) | 3.3% | 3.1% | |

**State format depends on the source**, which matters for address similarity:

| | S1 | S2 | S3 |
|---|---|---|---|
| US | code (`TX`) 99% | code (`TX`) | full name (`Texas`) |
| India | full name (`Maharashtra`) 99.9% | full name, or native script (`महाराष्ट्र`) | code (`MH`), or native script |

Other patterns seen in the samples:

- **Name changes:** a legal suffix swapped, dropped or expanded (`Inc`↔`Incorporated`,
  `Pvt Ltd`↔`Private Limited`); words duplicated (`Family Family Bright Health`); words
  reordered (`India Trust Charitable`); typos (`Sepciialsist`); a generic word added
  (`Dr`, `Center`, `Services`); and occasionally a **completely unrelated made-up name**
  (`Iriecto`, `Evohalovio`) that links to the entity only through its address.
- **Mixed scripts:** some names mix Latin and Indic script (`Al Tech प्राइवेट लिमिटेड`).
- **Address changes:** street-type abbreviations (Rd/Road, St/Street, Pkwy); state name
  vs code; components reordered; components dropped (keeping only `Flat 203, Hyderabad`);
  a changed or truncated house number (`607`→`60`, `405`→`05`); a wrong district or state
  (`Jalna`, `Andhra Pradesh` for a Telangana address); city-name variants
  (`Howrah`/`Haora`, `Mumbai`/`Greater Bombay`).
- **Postcodes are rare:** US ZIP appears in about 10% of addresses, Indian PIN in about
  1% (0.3% in S1), French code postal in 0.5%. **Postcodes can't serve as a main
  blocking key.**
- **Landmarks:** Indian addresses carry landmark text (`Near PNB Bank`, `Opp ...`) in
  8–13% of records, including in S1.

## 5. Noise taxonomy over all 7.6M true pairs

Each true pair (S1 record, matched S2/S3 record) was checked for every pattern below.
"Core name" = transliterated, normalised name with legal-form words removed and repeated
words collapsed. Rates are the share of true pairs.

**Name side**

| Pattern | All | US S2 | US S3 | India S2 | India S3 |
|---|---:|---:|---:|---:|---:|
| Normalised name identical | 25.8% | 30.5% | 30.5% | 17.5% | 20.0% |
| Core name identical | 50.4% | 56.4% | 52.1% | 43.8% | 45.5% |
| …only the legal form differs | 24.6% | 25.9% | 21.7% | 26.2% | 25.4% |
| Typo-like (core differs, similarity ≥ 85) | 16.0% | 17.9% | 17.0% | 13.6% | 14.1% |
| Word added | 9.1% | 4.2% | 9.3% | 10.3% | 14.4% |
| Word dropped | 5.1% | 7.3% | 7.7% | 0.9% | 1.9% |
| Words reordered | 5.6% | 6.2% | 5.8% | 5.0% | 5.0% |
| Repeated word | 4.3% | 5.5% | 5.0% | 2.9% | 2.9% |
| Indic-script name | 7.2% | 0 | 0 | 23.3% | 13.1% |
| Web domain | 4.4% | 5.1% | 4.8% | 3.5% | 3.7% |
| …domain = whole name concatenated | 2.3% | | | | |
| …domain = initials + word | 0.2% | | | | |
| Made-up name (Latin, not a domain, similarity < 40) | 1.8% | 1.5% | 1.7% | 1.1% | 3.0% |
| Bracketed suffix / junk prefix / digit swap | 8.1% / 1.3% / 1.9% | | | | |

**Address side**

| Pattern | All | US S2 | US S3 | India S2 | India S3 |
|---|---:|---:|---:|---:|---:|
| Address empty | 4.4% | 4.9% | 4.6% | 3.8% | 4.0% |
| Components only dropped (tokens ⊂ S1 tokens) | 6.4% | 7.8% | 0.2% | 20.7% | 0.4% |
| Same tokens, reordered | 3.5% | 5.9% | 2.0% | 4.1% | 1.8% |
| House/plot numbers identical | 60.6% | 57.0% | 66.4% | 61.1% | 56.4% |
| At least one number shared | 79.9% | 76.3% | 79.7% | 83.5% | 81.7% |
| All numbers missing in the record | 9.3% | 13.0% | 11.4% | 4.3% | 5.7% |
| Extra numbers added (unit, PO box, H No) | 20.1% | 19.1% | 17.9% | 23.6% | 21.4% |
| A number truncated (`607`→`60`) | 4.9% | 5.6% | 5.3% | 4.3% | 3.9% |
| Same state (after code/name canonicalisation) | 86.7% | 94.9% | 95.2% | 73.5% | 74.8% |
| Different state | 0.6% | 0.2% | 0.2% | 1.3% | 1.3% |

The record's address usually adds 1–2 tokens that aren't in S1 (76% of pairs add at least
one) and drops 1–2 S1 tokens (83% drop at least one).

**Sources differ in style.** S2 India often keeps only a prefix of the S1 address (20.7%
strict subsets); S3 rewrites it (reordered components, state codes). US S3 has full state
names where S1 has codes, which is why its raw address similarity is lower (median 88 vs 95
for US S2). Canonicalising state names should close most of that gap.

**Transliteration.** Name similarity (token set, core names) for records in each script,
without → with `anyascii` transliteration, median:

| Script | Pairs | Raw | Transliterated | Consonant skeleton |
|---|---:|---:|---:|---:|
| Latin | 7,087,125 | 100 | 100 | 100 |
| Devanagari | 312,725 | 10 | **59** | 67 |
| Telugu | 45,620 | 9 | 60 | 67 |
| Kannada | 43,379 | 9 | 60 | 67 |
| Tamil | 39,252 | 9 | 43 | 53 |
| Bengali | 35,910 | 10 | 57 | 67 |
| Gujarati | 35,819 | 10 | 62 | 70 |
| Malayalam | 21,930 | 9 | 49 | 55 |
| Odia | 8,681 | 9 | 60 | 67 |
| Gurmukhi | 7,924 | 10 | 48 | 56 |

"Consonant skeleton" drops vowels and `h` before comparing (`maharashtra`→`mrstr`),
which absorbs the vowel loss of transliteration (`महाराष्ट्र` → `mharastr`).

## 6. How matched pairs differ from non-matches

**Full population, by source** (7.6M true pairs; transliterated; p5 / p10 / p25 / p50):

| | Name token-set | Name Jaro-Winkler | Address token-set | Address token-sort |
|---|---|---|---|---|
| US S2 | 60 / 77 / 95 / 100 | 81 / 88 / 93 / 100 | 76 / 83 / 91 / 95 | 68 / 75 / 85 / 93 |
| US S3 | 60 / 77 / 96 / 100 | 66 / 81 / 92 / 100 | 68 / 75 / 82 / 88 | 64 / 70 / 78 / 85 |
| India S2 | 43 / 51 / 68 / 100 | 61 / 67 / 83 / 95 | 86 / 90 / 95 / 100 | 60 / 74 / 87 / 93 |
| India S3 | 42 / 54 / 79 / 100 | 56 / 66 / 84 / 96 | 71 / 80 / 89 / 95 | 40 / 47 / 67 / 87 |

Address token *order* is unreliable (India S3 token-sort median 87 but p10 47), so use
order-insensitive address similarity.

**Against negatives** (quick pass; 200k sampled true pairs vs three kinds of same-country
negatives: *random*, *shared-token* = shares the longest name token with the S1 record,
*same-name* = a different entity with an identical normalised name; no transliteration):

| Signal | True pairs | US | India | Random neg | Shared-token neg | Same-name neg |
|---|---|---|---|---|---|---|
| Normalised name exactly equal | 25.7% | 30.4% | 18.8% | 0% | 0.6% | 100% |
| Share ≥1 name token (excl. legal words) | 84.8% | 91.2% | 75.3% | 0.9% | 100% | 100% |
| Share ≥1 address token | 95.6% | | | 21% | 23% | 25% |
| Share ≥1 address number | 79.8% | 78.0% | 82.5% | 2.7% | 2.9% | 3.2% |

| (p10 / p25 / p50) | Name token-set | Address token-set |
|---|---|---|
| True pairs, all | 47 / 88 / 100 | 79 / 87 / 94 |
| Shared-token negatives | 55 / 60 / 67 | 29 / 33 / 37 |
| Same-name negatives | 100 / 100 / 100 | **30 / 33 / 37** |

**Takeaways**

1. **Name alone is not enough.** 39% of sampled S1 entities have a non-matching S2/S3
   record with the *identical* normalised name. With a name threshold of 80, 22% of
   shared-token negatives still pass.
2. **The address is the strongest signal.** Every negative type sits around 30–37
   address similarity. Name ≥ 60 **and** address ≥ 60 keeps 82.5% of true pairs and lets
   through only 1.5% of shared-token negatives and 1.1% of same-name negatives.
3. **When the name fails, the address still works.** 5.0% of true pairs have name
   similarity < 50 even after transliteration; their address similarity median is 96
   (p10 81), 85% share a house number, and only 0.2% of them have an empty address.
   **Only 0.06% of all true pairs are weak on both** (name < 50 and address < 60 or
   empty). Typical examples: a made-up name plus a heavily rewritten address
   (`Taraesh Strips LLP` → `Miraecto`, `H No 548 A 2205 Greater Bombay`).
4. **Empty addresses (4.4%) depend on the name alone.** Their name similarity median is
   100 (p5 77), so they are matchable, but only with a high name threshold.
5. **Made-up names (1.8%) and domains (4.4%) have median name similarity 23 and 63**
   respectively, but address similarity like any other pair (median 96 / 94).

## 7. Entity-level difficulty

The score is per S1 entity, so what matters is the hardest match of each entity:

| Share of S1 entities (with ≥1 match) that have… | |
|---|---:|
| all matches "easy" (name ≥ 80 **and** address ≥ 80) | 42.7% |
| at least one match with an empty address | 15.0% |
| at least one web-domain name | 15.0% |
| at least one Indic-script name | 13.1% |
| at least one made-up name | 6.4% |

The worst match per entity has name similarity p10 44 / p25 61 / median 84 and address
similarity p10 71 / p25 80 / median 87. A pipeline that only matches "easy" pairs would
still get perfect precision, but most entities would lose some recall. With F0.5, missing
1 of 4 matches still scores 0.94, so partial recall is cheap and wrong matches are
expensive.

## 8. Normalisation inventories

From [04_vocab.py](eda/deep/04_vocab.py) (full files, per country and source).

**Legal forms (last name token, S1):**

- US: `llc` 27%, `inc` 18%, `c` (from `P.C.`/`L.C.`) 3.4%, `corp`, `group`, `pc`, `pllc`,
  `lp`.
- India: `limited` 59%, `ltd` 17%, `llp` 4.4%, `co`, `trust`, `company`.
- France (test): `sarl` 28%, `sas` 20%, `eurl` 6.5%, `sa` 4.9%, `sasu` 4.1%, `sci` 3.2%,
  `ei` 1.6%.
- In S2/S3 India, the legal words also appear in native script (`लिमिटेड`, `లిమిటెడ్`,
  `ಲಿಮಿಟೆಡ್`, `லிமிடெட்`) and transliterate to `limitet`, `praivet`/`praibhet`, `pra`,
  `li`, `elelpi` (LLP). Add these to the legal-word list, or they'll count as name tokens.
- `com` is the last token of 3.4–4.3% of S2/S3 names (web domains).

**Street types and fillers:**

- US S1 writes them in full (`street`, `road`, `drive`, `avenue`, `lane`, `court`),
  S2/S3 mostly abbreviated (`st`, `rd`, `dr`, `ave`, `ln`, `ct`).
- India: `no`, `road`, `floor`, `nagar`, `plot`, `flat`, `sector`, `colony`, `block`,
  `door`, `near`, `h` (`H No`).
- France: `rue`/`r`, `avenue`/`ave`, `allee`, `boulevard`/`boulevrd`, `impasse`/`imp`,
  `chemin`, `cours`, `no`/`nº`, `bis`.

**Vocabulary size shows the typo volume.** S1 US has 71k distinct core name tokens, while
S2 and S3 US have 534k and 559k. Most of that is misspellings, so exact token matching
misses a lot. Character n-grams or edit distance are needed.

**Name-token frequency:** in S1 India, 40% of names consist only of the 200 most common
tokens (`india`, `services`, `solutions`, `trading`, `technologies`, …); US 13%, France 32%.
These names are ambiguous on their own and need IDF weighting plus the address.

**City concentration (S1, component before the state or region):**

| | Distinct | Top-10 share | Top-100 share | Largest |
|---|---:|---:|---:|---|
| US (train) | 100,207 | 6% | 25% | Houston 12k |
| India (train) | 74,907 | 34% | 73% | Bangalore 47k, Mumbai 36k + "Mumbai City" 31k |
| France (test) | 17,340 | **73%** | **93%** | Bordeaux 37k, Nantes 32k, Lille 30k |

French S1 addresses end in one of just 3 regions (Hauts-de-France, Nouvelle-Aquitaine,
Pays de la Loire), while S2/S3 France also use the département (`Nord`, `Gironde`,
`Loire-Atlantique`) or an uppercase city as the last component. **City-level blocks for
France will be very large**, so French blocking has to rely on street/name n-grams.

## 9. Blocking

### 9.1 Exact keys (quick pass)

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
| Name token **or** address number | **97.5%** | — | — | — |

No single exact key works: the keys with good recall produce thousands to hundreds of
thousands of candidates per entity. Candidates need ranking, which §9.2 measures.

**What distractors look like:** only **5.6%** of unmatched S2/S3 records have a normalised
name that appears verbatim in S1, against 29% of matched records. Distractors are mostly
*other* businesses in the same cities (for example `Vandenburg Services`,
`Lotech Group`), not near-copies of S1 records. They carry the same noise (caps, `null`,
misspelled cities), so noise alone can't identify a record as a distractor.

### 9.2 TF-IDF nearest neighbours on GPU (deep pass)

**Method** ([02_retrieval.py](eda/deep/02_retrieval.py)):

- Text is transliterated with `anyascii` and normalised.
- Character 3-gram TF-IDF (`char_wb`, sublinear TF, min_df 3; 11k–22k features),
  fitted separately per split and country.
- Exact cosine similarity, computed as a sparse × dense matrix product on the A100.
- 50,000 random S1 queries per country, each searched against **all** S2+S3 records of
  that country (6.2M US, 4.1M India). Top 100 kept per view.
- Three views: name only, address only, and name + address ("both").

**Pair recall** (share of true pairs found in the top k):

| View | US @5 | @10 | @20 | @50 | @100 | India @5 | @10 | @20 | @50 | @100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Name only | 51.8 | 61.1 | 67.7 | 74.3 | 78.7 | 43.8 | 51.0 | 56.4 | 62.5 | 66.1 |
| Address only | 69.8 | 81.3 | 86.5 | 89.9 | 91.2 | 69.5 | 77.4 | 80.7 | 83.5 | 85.2 |
| **Name + address** | 88.0 | **97.1** | **98.1** | 98.8 | 99.1 | 79.6 | **87.8** | **90.3** | 92.5 | 93.8 |
| Union of all three views | 93.7 | 98.1 | 98.8 | 99.2 | 99.4 | 87.5 | 92.0 | 93.8 | 95.5 | 96.4 |

(The union at k takes the top k of each view, so up to 3k candidates.)

- **Name-only search is weak** (68% / 56% at 20): same-name businesses in other cities
  crowd the true match out of the top k. The combined view is the right default.
- **Entities fully covered** (all matches in the top k, name + address): US 90.7% at 10,
  93.8% at 20; India 68.4% at 10, 74.0% at 20, 82.6% at 100.
- **Macro F0.5 ceiling:** the score a perfect matcher would reach using only these
  candidates (singletons count 1.0):

  | View | US @5 | @10 | @20 | @100 | India @5 | @10 | @20 | @100 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | Name + address | 0.975 | 0.991 | **0.994** | 0.997 | 0.933 | 0.954 | **0.963** | 0.977 |
  | Address only | 0.868 | 0.918 | 0.944 | 0.966 | 0.870 | 0.902 | 0.919 | 0.941 |
  | Name only | 0.712 | 0.771 | 0.815 | 0.898 | 0.622 | 0.677 | 0.724 | 0.811 |

**Which true pairs the combined view misses** (recall at 20, by pair category from §5):

| Category | US | India |
|---|---:|---:|
| Name identical after normalisation | 99.9% | 98.3% |
| Other (ordinary noise) | 99.3% | 96.0% |
| Web-domain name | 98.2% | 93.0% |
| Empty address | 80.7% | **35.9%** |
| Made-up name | 76.4% | 82.6% |
| Indic-script name | – | **77.3%** |

Two gaps stand out. **Empty-address records in India** need a name-only channel, and
Indian names are generic. **Indic names** are still behind even after transliteration.
Adding the name-only and address-only views (the union row) recovers some of both.

**Cosine alone does not separate matches from non-matches** (name + address view):

| (p10 / p25 / p50) | US | India |
|---|---|---|
| True pairs | 0.64 / 0.75 / 0.84 | 0.52 / 0.71 / 0.85 |
| Best non-match in the query's top 100 | 0.61 / 0.68 / 0.76 | 0.61 / 0.69 / 0.79 |
| Singletons: their top-1 candidate | 0.66 / 0.72 / 0.79 | 0.66 / 0.75 / 0.83 |

Only 71% (US) / 64% (India) of true pairs score above their query's best non-match, and
a singleton's best candidate usually looks as close as a real match. Ranking is
good enough for blocking but not for the final decision.

## 10. What simple decision rules score

Rules applied to the top-20 name + address candidates of the same 50k queries per country
(1M candidate pairs each, 17% / 16% of them true matches), scored with the challenge's
macro F0.5 (singletons included). "Mutual best" means the record's highest-cosine S1
among **all** S1 entities of the country is this query. Precision and recall are pooled
over all candidates.

| Rule | US F0.5 | P / R | India F0.5 | P / R |
|---|---:|---|---:|---|
| Cosine ≥ t (best t = 0.75) | **0.744** | 0.82 / 0.74 | 0.678 | 0.72 / 0.69 |
| Name ≥ a and address ≥ b (best a, b) | 0.706 | 0.75 / 0.81 | 0.660 | 0.71 / 0.70 |
| …plus mutual best | 0.729 | 0.76 / 0.92 | 0.719 | 0.76 / 0.84 |
| 0.4·name + 0.6·address ≥ t, plus mutual best | 0.729 | 0.76 / 0.93 | **0.724** | 0.77 / 0.84 |
| *Perfect matcher on the same candidates* | *0.994* | | *0.963* | |

Name and address similarity here are transliterated token-set scores (0–100); the best
thresholds were a = 80, b = 70 for the US and a = 50, b = 90 for India, and 40 / 60 once
mutual best is required.

**Mutual best is a very strong signal.** 98.3% (US) / 97.8% (India) of true candidate
pairs are mutual best, against only 6.9% / 6.2% of false candidates. It removes records
that belong to *another* S1 entity almost perfectly.

**What mutual best can't do is reject distractors.** A distractor has no true S1, so its
nearest S1 is usually the query itself. Their best-S1 cosine overlaps heavily with that
of real matches:

| Record's best-S1 cosine | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Matched records, US | 0.64 | 0.75 | 0.84 | | |
| Distractors, US | | | 0.73 | 0.81 | 0.86 |
| Matched records, India | 0.54 | 0.71 | 0.85 | | |
| Distractors, India | | | 0.75 | 0.84 | 0.89 |

That overlap, and singletons (5.6%, each scoring 0 on any false match), is why simple
rules plateau around 0.72–0.74 while the candidate ceiling is 0.96–0.99. **The gap is
the job of the learned matcher**, with features that tell a lookalike from the same
business: house numbers, state, token rarity, and the margin to the next candidate.

## 11. Train → test shift

| | Train | Test |
|---|---|---|
| Country mix of S1 | US 60%, India 40% | India 47%, US 38%, **France 15%** |
| S2+S3 records per S1 | **4.68** (US 4.67, India 4.68) | **5.75** (US 5.76, India 5.82, France 5.53) |

- **France is unseen.** Its records are French-language: legal forms `SARL`, `SAS`,
  `S.A.S`, `EURL`, `SASU`, `SCI`, `EI`, `Ets`/`Établissements`, `& Fils`, `& Cie`; street
  types `Rue`/`R`, `Avenue`/`Ave`, `Impasse`/`IMP.`, `Boulevard`/`BOULEVRD`, `Chemin`,
  `Cours`; `Nº`; accents dropped (`Mérignac`/`MERIGNAC`); region vs département. Names
  are often associations (`club`, `amicale`, `comité`, `école`, `sportive`) rather than
  companies. Noise styles match the other countries: S2 addresses ~32% uppercase, junk,
  domains and bracketed suffixes at similar rates. France has no Indic script and almost
  no postcodes. **Build normalisation and features from country-agnostic signals.** Keep
  per-country abbreviation tables open-ended, with a French table added. Don't one-hot
  `country`.
- **Test has about 23% more S2/S3 records per S1, because it has more distractors.**
  Measured with the same retrieval on 50k test queries per country and a random 200k
  records per country:

  | | Train US | Test US | Train India | Test India | Test France |
  |---|---:|---:|---:|---:|---:|
  | Top-1 cosine, median | 0.920 | 0.921 | 0.935 | 0.935 | 0.944 |
  | Candidates with cosine ≥ 0.7, per S1 | 3.7 | 4.3 | 4.4 | 5.2 | **10.0** |
  | Candidates with cosine ≥ 0.6, per S1 | 5.8 | 5.9 | 11.3 | 12.7 | **33.6** |
  | Estimated distractor share | 26.0% (true 25.9%) | **39.1%** | 25.9% (true 25.9%) | **39.4%** | 14.6%, unreliable |
  | Implied matched records per S1 | 3.46 | 3.51 | 3.47 | 3.53 | – |

  The distractor share is estimated by fitting each test split's histogram of
  "record → best S1 cosine" as a mixture of the train histograms for matched records and
  distractors. On train it recovers the known share almost exactly. For US and India
  test it gives about 39%, which implies the **same number of matches per entity as
  train (3.5)**. So the extra records in test are distractors.

  France has no training shapes, so its estimate is unreliable. If French entities have
  the usual 3.46 matches, its distractor share is about 37%.

- **Consequences:**
  1. A threshold tuned on train will let more false matches through on test. **Validate
     on a train split with injected distractors.** Remove about 18% of S1 entities from
     the validation S1 but keep their S2/S3 records. That lifts the distractor share from
     26% to about 39% without changing anything else.
  2. **Cosine levels don't transfer to France.** Its concentrated vocabulary (few cities,
     `rue`, `club`, `sarl`) gives 3× more candidates above 0.6 than India and 6× more
     than the US. Features and thresholds should be scale-free: rank, margin to the next
     candidate, mutual best, share of the query's candidates above the score.

## 12. Modelling implications

- **Pipeline:** block within country (TF-IDF top-k) → pair features → classifier →
  decision per S1 → enforce "each S2/S3 record goes to at most one S1" → write both
  TSVs. `candidate_pairs.tsv` is the top-k list that the classifier scores.
- **Normalise:**
  - Transliterate everything (`anyascii`; ISC licence).
  - Unicode NFKD plus accent stripping, lowercase, punctuation to space.
  - Strip junk prefixes, bracketed suffixes and `null`/`N/A` tokens; collapse repeated
    words; map digit swaps (0→o, 1→l, 5→s) inside words.
  - Canonicalise legal forms, including transliterated Indic forms (`praivet`, `limitet`,
    `elelpi`) and French forms (`sarl`, `sas`, `sasu`, `eurl`, `sci`, `ei`).
  - Canonicalise street types (US, India, France); map state codes and native-script
    state names to one form; split domain names into words.
- **Blocking:**
  - Character 3-gram TF-IDF on transliterated name + address, top **20** per S1 for the
    US (98.1% pair recall) and top **50–100** for India (92.5–93.8%).
  - Add the name-only and address-only top-k lists (union: US 98.8%, India 93.8% at 20
    each).
  - Add a name-only channel for records with an empty address (India: only 36% found
    at 20 otherwise).
  - Check French recall on a pseudo-labelled sample. France's dense vocabulary may need
    a larger k.
- **Features:**
  - Name and address similarities (token set, Jaro-Winkler, TF-IDF cosine, consonant
    skeleton).
  - House-number overlap, equality and truncation; state agreement after
    canonicalisation.
  - Flags: empty address, domain name, non-Latin name.
  - IDF/rarity of shared name tokens ("Primary Care Group" is shared by 253 S1
    entities).
  - Relative features: rank, margin to the next candidate, whether the pair is mutual
    best, how many of the query's candidates score higher. Siblings from the same source
    (within-source duplicates have median similarity 97).
- **Decisions:**
  - Optimise macro F0.5 directly on a validation split with ~39% distractors, including
    singletons (each false match on a singleton costs 1.0).
  - Missing one of four matches still scores 0.94, so lean towards precision.
  - Apply the one-S1-per-record constraint after scoring (mutual best keeps 98% of true
    pairs).

## Appendix: reproducing the numbers

**Quick pass** ([eda/](eda/), laptop, about 15 minutes in total): `cd eda && python
<script>.py` in the order listed at the top. Needs pandas, pyarrow and rapidfuzz.

**Deep pass** ([eda/deep/](eda/deep/)):

| Script | What it produces | Runtime on scai03 (24 CPUs, A100) |
|---|---|---|
| `01_pair_noise.py` | §5–§7: flags and similarities for all 7.6M true pairs, entity-level stats | 14 min |
| `04_vocab.py` | §4, §8: legal forms, street types, states, cities, same-name groups | 1 min |
| `02_retrieval.py` | §9.2, §11: GPU TF-IDF top-100 for 50k queries × 3 views, per split and country; reverse search | 50 min |
| `03_blocking_eval.py` | §9.2, §10, §11: recall, F0.5 ceilings, rule grid, distractor mixture estimate | 3 min |

- Shared code (loading, transliteration, normalisation, state tables, parallel map) is
  in `common.py`.
- Settings are environment variables: `DATA_DIR` (default `student_resource/dataset`),
  `OUT_DIR` (default `eda/deep/out/`, gitignored), `WORKERS`, and for 02 `Q_N`
  (queries per country), `K`, `REV_SAMPLE`.
- Pinned packages are in [eda/deep/requirements.txt](eda/deep/requirements.txt); torch
  with CUDA is only needed by 02.
- On PADUM, `qsub -v PY=/path/to/env/bin/python eda/deep/run_deep_eda.pbs` from the repo
  root runs everything in one job (24 CPUs, 1 GPU, about an hour). The `scai_q` queue
  requires at least one GPU per job.
- Results land in `eda/deep/out/`: `01_pair_noise.json`, `03_blocking_eval.json`,
  `04_vocab.json`, `pair_features.parquet` (per-pair flags) and `retrieval/`
  (top-k lists). The three JSON files from the run behind this document are committed
  in [eda/deep/results/](eda/deep/results/).
