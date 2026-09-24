"""Shared helpers for the deep EDA: data loading, text normalisation, parallel map.

Paths come from env vars so the same code runs on the laptop and on PADUM:
  DATA_DIR  folder containing train/ and test/   (default: <repo>/student_resource/dataset)
  OUT_DIR   where results and parquet caches go  (default: eda/deep/out)
  WORKERS   process count for parallel maps      (default: all CPUs in this job's affinity)
"""
import json
import os
import re
import unicodedata
from multiprocessing import get_context

import numpy as np
import pandas as pd
import pyarrow.csv as pc
import pyarrow.parquet as pq
from anyascii import anyascii

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(HERE, "..", "..", "student_resource", "dataset"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
CACHE = os.path.join(OUT_DIR, "cache")
WORKERS = int(os.environ.get("WORKERS", len(os.sched_getaffinity(0))))

FILES = {
    "train_s1": "train/train_source1.tsv", "train_s2": "train/train_source2.tsv",
    "train_s3": "train/train_source3.tsv", "train_gt": "train/train_ground_truth.tsv",
    "test_s1": "test/test_source1.tsv", "test_s2": "test/test_source2.tsv",
    "test_s3": "test/test_source3.tsv",
}


# ---------------------------------------------------------------- loading

def load(name):
    """Return one challenge file as a DataFrame, cached as parquet under OUT_DIR/cache."""
    cached = os.path.join(CACHE, name + ".parquet")
    if os.path.exists(cached):
        return pq.read_table(cached).to_pandas()
    table = pc.read_csv(
        os.path.join(DATA_DIR, FILES[name]),
        parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pc.ConvertOptions(strings_can_be_null=False),
    )
    os.makedirs(CACHE, exist_ok=True)
    pq.write_table(table, cached)
    return table.to_pandas()


def records(split):
    """S2 and S3 of a split stacked into one frame, with a `src` column."""
    s2, s3 = load(f"{split}_s2"), load(f"{split}_s3")
    return pd.concat([s2.assign(src="S2"), s3.assign(src="S3")], ignore_index=True)


def train_pairs():
    """Every ground-truth pair joined with both records' fields (cached)."""
    cached = os.path.join(CACHE, "train_pairs_joined.parquet")
    if os.path.exists(cached):
        return pd.read_parquet(cached)
    gt = load("train_gt")
    lists = gt.matched_entity_ids.str.strip()
    p = (gt.assign(id=lists.where(lists != "", None).str.split(","))
           .explode("id").dropna(subset=["id"])
           .rename(columns={"source1_entity_id": "s1"})[["s1", "id"]])
    s1 = load("train_s1").rename(columns={"entity_id": "s1", "business_name": "name1",
                                          "business_address": "addr1"})
    r = records("train").rename(columns={"entity_id": "id", "business_name": "name2",
                                         "business_address": "addr2"}).drop(columns="country")
    j = p.merge(s1, on="s1").merge(r, on="id")
    j.to_parquet(cached)
    return j


# ---------------------------------------------------------------- text

LEGAL = set("""inc incorporated llc l l c ltd limited pvt private corp corporation co company plc llp lp pc
pllc lc pa sarl sas sasu eurl sa sci snc ei ets etablissements""".split())
GENERIC = set("""the and of services service group center centre solutions enterprises enterprise
international global""".split())
NULLS = {"null", "n/a", "na", "none", "nan"}

SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF), ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF), ("Odia", 0x0B00, 0x0B7F), ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F),
]
INDIC_RE = re.compile(r"[ऀ-ൿ]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
DOMAIN_RE = re.compile(r"(?i)^\s*(?:www\.)?([a-z0-9\-]+)\.(com|in|net|org|co\.in|co|biz|info|fr)\s*$")
JUNK_PREFIX_RE = re.compile(r"^\s*(>>|\.\.\.|\*\*\*|--|<<)\s")
BRACKET_RE = re.compile(r"[\[\(][^\]\)]*[\]\)]")
LEET_RE = re.compile(r"(?i)[a-z][013458][a-z]")
NUM_RE = re.compile(r"\d+")
POBOX_RE = re.compile(r"(?i)\b(p\.?\s?o\.?\s?box|pmb)\b")
LANDMARK_RE = re.compile(r"(?i)\b(near|opp|opposite|behind|nr)\b")


def script_of(s):
    for ch in s:
        o = ord(ch)
        if 0x0900 <= o <= 0x0D7F:
            for name, a, b in SCRIPT_RANGES:
                if a <= o <= b:
                    return name
    return "Latin"


def norm(s):
    """NFKD, strip accents, lowercase, punctuation -> space, collapse whitespace (keeps Indic)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c) or INDIC_RE.match(c)).lower()
    s = re.sub(r"[^\w\s\u0900-\u0D7F]", " ", s)  # keep Indic vowel signs (not \w)
    return re.sub(r"\s+", " ", s).strip()


def translit(s):
    """ASCII transliteration (anyascii handles Indic scripts and accents), then norm."""
    return norm(anyascii(s))


def skeleton(s):
    """Consonant skeleton of an ASCII string: drops vowels, h and repeats (maharashtra -> mrstr)."""
    s = re.sub(r"[aeiouyh\s]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def core_tokens(s_norm):
    """Name tokens minus legal-form words, with consecutive duplicates collapsed."""
    out = []
    for t in s_norm.split():
        if t in LEGAL or (out and out[-1] == t):
            continue
        out.append(t)
    return out


def addr_tokens(s_norm):
    return [t for t in s_norm.split() if t not in NULLS]


# US and India state names / codes, lower-case, mapped to one canonical key.
US_STATES = dict(al="alabama", ak="alaska", az="arizona", ar="arkansas", ca="california", co="colorado",
                 ct="connecticut", de="delaware", fl="florida", ga="georgia", hi="hawaii", id="idaho",
                 il="illinois", in_="indiana", ia="iowa", ks="kansas", ky="kentucky", la="louisiana",
                 me="maine", md="maryland", ma="massachusetts", mi="michigan", mn="minnesota",
                 ms="mississippi", mo="missouri", mt="montana", ne="nebraska", nv="nevada",
                 nh="new hampshire", nj="new jersey", nm="new mexico", ny="new york",
                 nc="north carolina", nd="north dakota", oh="ohio", ok="oklahoma", or_="oregon",
                 pa="pennsylvania", ri="rhode island", sc="south carolina", sd="south dakota",
                 tn="tennessee", tx="texas", ut="utah", vt="vermont", va="virginia", wa="washington",
                 wv="west virginia", wi="wisconsin", wy="wyoming", dc="district of columbia")
IN_STATES = {
    "andhra pradesh": ["ap"], "arunachal pradesh": ["ar"], "assam": ["as"], "bihar": ["br"],
    "chhattisgarh": ["cg", "ct"], "goa": ["ga"], "gujarat": ["gj"], "haryana": ["hr"],
    "himachal pradesh": ["hp"], "jharkhand": ["jh"], "karnataka": ["ka"], "kerala": ["kl"],
    "madhya pradesh": ["mp"], "maharashtra": ["mh"], "manipur": ["mn"], "meghalaya": ["ml"],
    "mizoram": ["mz"], "nagaland": ["nl"], "odisha": ["od", "or", "orissa"], "punjab": ["pb"],
    "rajasthan": ["rj"], "sikkim": ["sk"], "tamil nadu": ["tn"], "telangana": ["tg", "ts"],
    "tripura": ["tr"], "uttar pradesh": ["up"], "uttarakhand": ["uk", "ut", "uttaranchal"],
    "west bengal": ["wb"], "delhi": ["dl", "new delhi"], "jammu and kashmir": ["jk", "jammu kashmir"],
    "chandigarh": ["ch"], "puducherry": ["py", "pondicherry"], "ladakh": ["la"],
}
STATE_LOOKUP = {"US": {}, "India": {}}
for code, name in US_STATES.items():
    STATE_LOOKUP["US"][code.rstrip("_")] = name
    STATE_LOOKUP["US"][name] = name
for name, codes in IN_STATES.items():
    STATE_LOOKUP["India"][name] = name
    for c in codes:
        STATE_LOOKUP["India"][c] = name


def address_state(addr, country):
    """(canonical_state, form) from the comma-separated components; form is code/name/native/None."""
    table = STATE_LOOKUP.get(country)
    if not table:
        return None, None
    native = False
    for comp in addr.split(","):
        if INDIC_RE.search(comp):
            native = True
            continue
        c = norm(comp)
        if c in table:
            return table[c], ("code" if len(c) <= 2 else "name")
    return (None, "native") if native else (None, None)


# ---------------------------------------------------------------- parallel helpers

def pmap_frame(fn, df, workers=WORKERS, chunks_per_worker=4):
    """Apply fn(DataFrame chunk) -> DataFrame over row chunks in a process pool; concat results."""
    n = max(1, workers * chunks_per_worker)
    parts = np.array_split(np.arange(len(df)), n)
    chunks = [df.iloc[p] for p in parts if len(p)]
    with get_context("fork").Pool(workers) as pool:
        out = pool.map(fn, chunks)
    return pd.concat(out, ignore_index=True)


def save_json(obj, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)
