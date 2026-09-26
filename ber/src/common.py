import os, re, zlib, multiprocessing
import numpy as np, pandas as pd
from anyascii import anyascii

multiprocessing.set_start_method("fork", force=True)   # python >= 3.14 defaults to forkserver; our pools share globals

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


HOLDOUT = os.environ.get("HOLDOUT", "")   # leave-one-country-out: this train country is never trained on
HELD = 10                                  # its fold: outside 0-9, so no training or validation step selects it
_held = None


def s1_fold(ids):
    """Deterministic split of train S1 ids: 0..9 by crc32. Folds 0-3 -> embedder, 4-8 -> GBM, 9 -> validation.
    With HOLDOUT=<country>, every train S1 / S2 / S3 id of that country gets fold HELD instead."""
    f = np.array([zlib.crc32(x.encode()) % 10 for x in ids], dtype=np.int8)
    if HOLDOUT:
        global _held
        if _held is None:
            _held = set()
            for src in (1, 2, 3):
                d = load("train", src)
                _held.update(d.entity_id.values[d.country.values == HOLDOUT])
            assert _held, f"no train records of HOLDOUT={HOLDOUT}"
        f[np.fromiter((x in _held for x in ids), bool, len(ids))] = HELD
    return f


_NAME_ABBR = {
    "pvt": "private", "pte": "private", "ltd": "limited", "ltda": "limited", "corp": "corporation", "co": "company",
    "inc": "incorporated", "intl": "international", "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "mgmt": "management", "assoc": "associates", "bros": "brothers", "ent": "enterprises", "tech": "technologies",
    "&": "and", "n": "and",
}
# Generic English defaults only (the noise the problem statement lists: Rd/Road, St/Street, Pvt/Private, Ltd/Limited).
# Everything country- or language-specific (state codes, French street words, legal forms, transliteration
# variants) is MINED per country label from the data by mine_dicts.py, and takes precedence over these defaults.
_ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ct": "court",
    "blvd": "boulevard", "ln": "lane", "hwy": "highway", "pkwy": "parkway", "pl": "place",
    "cir": "circle", "trl": "trail", "ter": "terrace", "sq": "square", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "no": "number", "nr": "near", "opp": "opposite",
    "n.": "north", "s.": "south", "e.": "east", "w.": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
}
_LEGAL = set("private limited llp llc incorporated corporation company plc pllc pc lp llp "
             "the dba ltd pvt inc corp co and of "
             # transliterated legal words found in the TRAINING data (EDA section 8)
             "praivet praibhet prayvet limitet limted lmtd elelpi elelpee".split())
_JUNK = {"null", "none", "nan", "na"}
_TOK = re.compile(r"[a-z0-9]+")
_NONE = {"name": {}, "address": {}, "legal": frozenset()}
_MINED = None


def mined():
    """country label -> {"name": {short: long}, "address": {short: long}, "legal": set} (work/dicts, mine_dicts.py).
    Missing dictionaries are an error (a silent fallback would change the normalisation); NO_DICTS=1 skips them."""
    global _MINED
    if _MINED is None:
        _MINED = {}
        if not os.environ.get("NO_DICTS"):
            ab, lg = pd.read_parquet(f"{WORK}/dicts/abbr.parquet"), pd.read_parquet(f"{WORK}/dicts/legal.parquet")
            for c in set(ab.country) | set(lg.country):
                a = ab[ab.country == c]
                _MINED[c] = {f: dict(zip(a.short[a.field == f], a.long[a.field == f])) for f in ("name", "address")}
                _MINED[c]["legal"] = frozenset(lg.tok[lg.country == c])
    return _MINED


def _countries(s, c):
    return c.tolist() if c is not None else [None] * len(s)


def translit(s: pd.Series) -> pd.Series:
    return s.map(lambda x: anyascii(x).lower())


def norm_name(s: pd.Series, c: pd.Series = None) -> pd.Series:
    """c: country labels aligned with s (selects the mined dictionary; None = generic defaults only)."""
    M = mined()
    def f(x, cc):
        m = M.get(cc, _NONE)["name"]
        x = anyascii(x).lower().replace("&", " and ")
        x = re.sub(r"\.(com|net|org|io|biz|[a-z]{2})\b", " ", x)  # website-style names (generic + any 2-letter TLD)
        return " ".join(m.get(t) or _NAME_ABBR.get(t, t) for t in _TOK.findall(x))
    return pd.Series([f(x, cc) for x, cc in zip(s.tolist(), _countries(s, c))], index=s.index)


def core_name(s: pd.Series, c: pd.Series = None) -> pd.Series:
    M = mined()
    def f(x, cc):
        legal = M.get(cc, _NONE)["legal"]
        out = []
        for t in x.split():
            if t not in _LEGAL and t not in legal and t not in _JUNK and (not out or out[-1] != t):
                out.append(t)                                  # drop legal words, collapse repeats
        return " ".join(out)
    return pd.Series([f(x, cc) for x, cc in zip(s.tolist(), _countries(s, c))], index=s.index)


_VOW = re.compile(r"[aeiouh]")


def skeleton(s: pd.Series) -> pd.Series:
    """consonant skeleton per token: absorbs vowel loss of transliteration (maharashtra -> mrstr)."""
    return s.map(lambda x: " ".join(filter(None, (t[0] + _VOW.sub("", t[1:]) for t in x.split()))))


def norm_addr(s: pd.Series, c: pd.Series = None) -> pd.Series:
    M = mined()
    def f(x, cc):
        m = M.get(cc, _NONE)["address"]
        return " ".join(m.get(t) or _ADDR_ABBR.get(t, t) for t in _TOK.findall(anyascii(x).lower()) if t not in _JUNK)
    return pd.Series([f(x, cc) for x, cc in zip(s.tolist(), _countries(s, c))], index=s.index)


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
    names = pd.Series(["Raab Modern Treoasubr,y LLC", "व्हाइट बिल्डर्स प्राइवेट लिमिटेड", "wilfordhancock.com", "Grain & Fils"])
    print(norm_name(names, pd.Series(["US", "India", "US", "France"])).tolist())
    print(norm_addr(pd.Series(["2670- DUMBLE ST, ALVIN, TX", "63 R. DE DIEPPE, LILLE, Hauts-de-France"]),
                    pd.Series(["US", "France"])).tolist())
    print("ok")
