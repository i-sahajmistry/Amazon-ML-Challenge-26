import os, re, zlib
import numpy as np, pandas as pd
from anyascii import anyascii

# default: the folder that contains src/ (put or symlink student_resource/ there), override with AMLC_ROOT
ROOT = os.path.expanduser(os.environ.get("AMLC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = f"{ROOT}/student_resource/dataset"
WORK = f"{ROOT}/work"
os.makedirs(WORK, exist_ok=True)


def load(split, src):
    """split: train|test, src: 1|2|3|'ground_truth'. Cached as parquet."""
    name = f"{split}_source{src}" if src != "ground_truth" else f"{split}_ground_truth"
    pq = f"{WORK}/pq/{name}.parquet"
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    df = pd.read_csv(f"{DATA}/{split}/{name}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    os.makedirs(os.path.dirname(pq), exist_ok=True)
    df.to_parquet(pq)
    return df


def s1_fold(ids):
    """Deterministic split of train S1 ids: 0..9 by crc32. Folds 0-3 -> embedder, 4-8 -> GBM, 9 -> validation."""
    return np.array([zlib.crc32(x.encode()) % 10 for x in ids], dtype=np.int8)


_NAME_ABBR = {
    "pvt": "private", "pte": "private", "ltd": "limited", "ltda": "limited", "corp": "corporation", "co": "company",
    "inc": "incorporated", "intl": "international", "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "mgmt": "management", "assoc": "associates", "bros": "brothers", "ent": "enterprises", "tech": "technologies",
    "&": "and", "n": "and",
}
_ADDR_ABBR = {  # US / India / France
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ct": "court",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "hwy": "highway", "pkwy": "parkway", "pl": "place",
    "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "no": "number", "nr": "near", "opp": "opposite",
    "r": "rue", "all": "allee", "chem": "chemin", "rte": "route", "imp": "impasse", "fbg": "faubourg",
    "n.": "north", "s.": "south", "e.": "east", "w.": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "mg": "mahatma gandhi",
}
_LEGAL = set("private limited llp llc incorporated corporation company plc pllc pc lp llp sarl sas sasu eurl sa sci snc "
             "the dba gmbh ltd pvt inc corp co and of".split())
_STATES = {  # expand abbreviations to full names so "TX" == "Texas"; open set, unknown tokens pass through
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "up": "uttar pradesh", "mh": "maharashtra", "tn.": "tamil nadu", "ka": "karnataka", "kl": "kerala",
    "wb": "west bengal", "mp": "madhya pradesh", "hr": "haryana", "pb": "punjab", "rj": "rajasthan",
    "gj": "gujarat", "ap": "andhra pradesh", "ts": "telangana", "dl": "delhi", "br": "bihar", "or.": "odisha",
    "jh": "jharkhand", "uk": "uttarakhand", "hp": "himachal pradesh", "jk": "jammu and kashmir",
}
_TOK = re.compile(r"[a-z0-9]+")


def translit(s: pd.Series) -> pd.Series:
    return s.map(lambda x: anyascii(x).lower())


def norm_name(s: pd.Series) -> pd.Series:
    def f(x):
        x = anyascii(x).lower().replace("&", " and ")
        x = re.sub(r"\.(com|in|net|org|co|fr|io|biz)\b", " ", x)  # website-style names
        return " ".join(_NAME_ABBR.get(t, t) for t in _TOK.findall(x))
    return s.map(f)


def core_name(s: pd.Series) -> pd.Series:
    return s.map(lambda x: " ".join(t for t in x.split() if t not in _LEGAL))


def norm_addr(s: pd.Series) -> pd.Series:
    def f(x):
        toks = _TOK.findall(anyascii(x).lower())
        out = []
        for t in toks:
            t = _ADDR_ABBR.get(t, t)
            out.append(_STATES.get(t, t) if len(t) == 2 else t)
        return " ".join(out)
    return s.map(f)


def embed_text(df: pd.DataFrame) -> list:
    return ("query: " + translit(df.business_name) + " | " + translit(df.business_address)).tolist()


def f05(pred: dict, truth: dict) -> float:
    """Macro F0.5 over truth keys (S1 ids). pred/truth: id -> set."""
    tot = 0.0
    for k, t in truth.items():
        p = pred.get(k, set())
        if not t:
            tot += 1.0 if not p else 0.0
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(truth)


if __name__ == "__main__":
    t = {"a": {"x", "y"}, "b": set(), "c": {"z"}}
    assert abs(f05({"a": {"x", "y", "w"}}, {"a": {"x", "y"}}) - 0.714285) < 1e-4  # README example
    assert f05({"b": set()}, {"b": set()}) == 1.0 and f05({"b": {"q"}}, {"b": set()}) == 0.0
    assert f05({}, t) == 1 / 3
    print(norm_name(pd.Series(["Raab Modern Treoasubr,y LLC", "व्हाइट बिल्डर्स प्राइवेट लिमिटेड", "wilfordhancock.com", "Grain & Fils"])).tolist())
    print(norm_addr(pd.Series(["2670- DUMBLE ST, ALVIN, TX", "63 R. DE DIEPPE, LILLE, Hauts-de-France"])).tolist())
    print("ok")
