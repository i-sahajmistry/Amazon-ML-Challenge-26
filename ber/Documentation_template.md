# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** SSM  
**Team Members:** Sahaj Mistry, Sarvesh Nikas, Mohanish Baviskar  
**Submission Date:** 25 September 2026

---

## 1. Executive Summary
We fine-tune a small multilingual bi-encoder (multilingual-e5-small, MIT, 118M parameters) on the training pairs and
use exact GPU nearest-neighbour search as blocking. It finds the true Source 1 entity among the top 5 candidates for
99.0% of Source 2/3 records. A LightGBM matcher then scores each candidate pair on ~30 string and retrieval
features. Each record is assigned to at most one Source 1 entity, only above a threshold tuned for F0.5.
Macro F0.5 on a held-out 10% of Source 1 entities is **0.9890**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **One owner per record.** In the training ground truth, 7.64M S2/S3 ids are matched and none is matched to two
  S1 entities. 73–75% of S2/S3 records have a match; the rest are distractors.
- **Few singletons.** Only 5.6% of S1 entities have no match (same rate in US and India). The average is ~3.5
  matches per entity (up to 10), split between S2 and S3.
- **Noise.**
  - Names: typos ("Treoasubr,y"), legal-suffix changes and reordering ("Llc Raab Modern Treasury,"), accented
    letters ("Fírst", "Prógram"), bracketed words ("Superior Blue [Rising]"), website names ("raabmoderntreasury.com"),
    DBA prefixes, and names written in Devanagari, Gujarati, Tamil or Kannada script.
  - Addresses: abbreviations (ST/Street, TX/Texas), reordered components, changed street numbers ("2415"→"02415",
    "3231-3233"), and a missing address in ~3% of S2/S3 records.
- **Distribution shift.** The test set adds France (15% of test S1), which does not appear in training. Country is
  therefore treated as an open set of labels.
- **Metric.** Macro F0.5 per S1 entity. A single wrong merge on a singleton costs a full point, so precision matters,
  but with ~3.5 true matches per entity, recall is also worth a lot.

### 2.2 Solution Strategy
**Approach Type:** Blocking (fine-tuned bi-encoder retrieval) + pairwise classifier (LightGBM) + constrained assignment  
**Core Innovation:** Because every S2/S3 record has at most one owner, the problem is solved per record: each record
is given to its best S1 candidate only if the model is confident enough. This removes most multi-assignment false
merges. A contrastively fine-tuned multilingual encoder, applied after transliteration, gives near-perfect
blocking recall across Latin and Indic scripts.

Train S1 ids are split into 10 folds by `crc32(id) % 10`:
- folds 0–3 train the encoder;
- folds 4–8 train the matcher;
- fold 9 is held out for threshold selection and all reported scores.

The matcher and the reported numbers therefore never use pairs the encoder was trained on.

---

## 3. Candidate Generation (Blocking)
- **Text normalisation:**
  - Every script is transliterated to ASCII with `anyascii` ("व्हाइट बिल्डर्स प्राइवेट लिमिटेड" →
    "vhait bildrs praivet limited") and lowercased.
  - Abbreviations are expanded (Pvt→private, Rd→road, TX→texas, R.→rue) and website suffixes are removed.
  - A "core" name is built without legal words (LLC, Pvt Ltd, SARL, SAS, …).
- **Blocking model:** `intfloat/multilingual-e5-small`, fine-tuned for one epoch on 1.6M
  (S2/S3 record, S1 entity) pairs with symmetric in-batch-negative contrastive loss (batch 1024, lr 1e-4,
  max 64 tokens, 8 minutes on one A100). Input text is `name | address`.
- **Blocking keys used:** exact equality of the country string (open set, so France is handled like any other
  label), then cosine similarity of the embeddings. For every S2/S3 record we take the top-20 S1 records by exact
  matrix multiplication on the GPU (no approximate search).
- **Candidate pairs generated:** the top-5 per S2/S3 record go to the matcher. On test: 9.97M records ×
  5 = 49.8M pairs (≈29 candidates per S1 entity, versus 1.7M × 10M possible pairs, a reduction ratio above
  99.999%). This set is `candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** on S1 folds the encoder never saw, the true S1 is ranked
  1st for 97.8% of matched records, in the top 5 for 99.0% and in the top 20 for 99.5%. Recall is identical on the
  training folds, so the encoder does not overfit. A perfect matcher on our candidate set would score F0.5 = 0.9970.

---

## 4. Matching Model

**Features used (31):**
- **Retrieval context:** cosine score, rank, gap to the record's best and next candidate, the record's rank
  among all records retrieved for the same S1, and how many records retrieve that S1 (and rank it first).
- **Name features:** rapidfuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler, computed on both the
  normalised name and the core name. Also: similarity of the core names with spaces removed (catches website-style
  names), exact core-name equality, and name lengths.
- **Address features:** ratio, token-set and partial token-set ratio on normalised addresses; token-set
  similarity of all numbers; equality of the first number (house or street number); PIN/ZIP equality (or
  missing).
- **Other:** empty-address and no-number flags, non-Latin-script flag for the record's name, and which source the
  record came from (S2 or S3).

**Model type:** LightGBM binary classifier (255 leaves, lr 0.05, early stopping on fold 9 at 274 rounds), trained
on 25.8M pairs from S1 folds 4–8 (14.7% positive). The most important features by gain are partial name ratio,
no-space core-name ratio, retrieval rank, gap to the next candidate, and core-name token-set ratio.  
**Threshold selection method:** each S2/S3 record goes to its highest-probability S1 candidate if p ≥ t. The
threshold t is swept from 0.05 to 0.95 and chosen by macro F0.5 on the held-out fold-9 S1 entities; t = 0.20.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9890** on held-out fold 9 (220k S1 entities, 1.03M S2/S3 records).
  - v1 (lr 0.1, t = 0.30) scored 0.9884.
  - The ceiling with the same candidates is 0.9970.
- **Wrong merges (28.0k accepted pairs, 3.6%):**
  - 81% attach a record that truly has no owner. Typical cases:
    - the record has a missing address and a near-identical name ("Zesoft Ínc", "Ribeiro Pioneer Royalties Llc");
    - names that share a generic core ("Laxmi Consulting" vs "Laxmi Marketing" at the same address);
    - names in Tamil script whose only overlap is the legal suffix.
  - 19% give a truly matched record to the wrong S1: neighbouring businesses with swapped words and a nearby
    house number ("Vava Unified Better, 2600 Ponderosa Dr" vs "Better Unified Vioavi, 2607 Ponderosa Dr").
- **Missed matches (11.5k true pairs below threshold, 1.5%):**
  - Mostly records with no address and a truncated name ("Bhardwaj Consultants Pvt", "New Delhi Software [Private]").
  - Rebranded or DBA names with nothing in common ("Brixveravio" for Strategic Business Union LLC).
  - Heavily corrupted names (OCR-like "The Fírst 8ay Sui").
  - Tamil-script names where transliteration diverges from the English spelling.

---

## 6. Conclusion
Treating entity resolution as "which owner, if any, does each record have" fits the one-owner structure of the
data. Paired with a contrastively fine-tuned multilingual encoder for blocking, it gives 99% blocking recall and
0.989 macro F0.5 with a lightweight, fully reproducible pipeline (about 1.5 GPU-hours end to end). The remaining
gap to the 0.997 ceiling comes from records with missing addresses and generic names. A cross-encoder re-ranker on
uncertain pairs is the natural next step.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/common.py`: data loading (TSV→parquet cache), transliteration and normalisation, fold split, macro F0.5 scorer
- `src/prep_norm.py`: normalises all records once (multiprocessing)
- `src/train_embed.py`: fine-tunes multilingual-e5-small on S1 folds 0–3
- `src/retrieve.py`: embeds all records and runs the same-country top-20 GPU search (`train` | `test`)
- `src/match.py`: features, LightGBM training and threshold tuning (`train`), inference and output writing (`test`),
  threshold re-tuning from cached predictions (`tune`)
- `src/error_analysis.py`: the fold-9 error breakdown above

Entry points, in order: `prep_norm.py → train_embed.py → retrieve.py train → retrieve.py test → match.py train →
match.py test`, which writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. See `README.md`.

### B. Additional Results

| Threshold | 0.05 | 0.10 | 0.15 | **0.20** | 0.30 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| Fold-9 F0.5 | 0.9869 | 0.9885 | 0.9889 | **0.9890** | 0.9885 | 0.9864 | 0.9826 | 0.9713 |

Blocking recall of the true S1 by rank (folds 4–9): @1 97.84%, @2 98.48%, @3 98.76%, @5 99.03%, @10 99.29%, @20 99.51%.

Models used: `intfloat/multilingual-e5-small` (MIT licence, 118M parameters) and LightGBM. No external data, APIs
or lookup services are used.
