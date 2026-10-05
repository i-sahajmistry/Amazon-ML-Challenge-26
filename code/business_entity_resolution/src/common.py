import os, re, json, zlib
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


def unlabelled():
    """test countries without a single labelled training record (France here), found from the data, so no country is
    named in the pipeline: FROM= / THR_C= (x_final.py), COUNTRY= (x_llm.py) and rule_fr.py take 'unlabelled' for them"""
    return sorted(set(load("test", 1).country.unique()) - set(load("train", 1).country.unique()))


def countries(spec):
    """a comma-separated country list in which 'unlabelled' stands for unlabelled()"""
    out = []
    for c in filter(None, spec.split(",")):
        out += unlabelled() if c == "unlabelled" else [c]
    return out


def s1_fold(ids):
    """Deterministic split of train S1 ids: 0..9 by crc32. Folds 0-3 -> embedder, 4-8 -> GBM, 9 -> validation."""
    return np.array([zlib.crc32(x.encode()) % 10 for x in ids], dtype=np.int8)


_JUNK = {"null", "none", "nan", "na"}   # empty-field markers in the raw files
_TOK = re.compile(r"[a-z0-9]+")
_LEX = None


def lexicon():
    """Per-country abbreviations and legal forms learned from the records by lexicon.py (no hand-written tables:
    the same abbreviation means different things in different countries). Empty until lexicon.py has run."""
    global _LEX
    if _LEX is None:
        f = f"{WORK}/lexicon.json"
        _LEX = json.load(open(f)) if os.path.exists(f) else {"abbr": {}, "legal": {}}
    return _LEX


def translit(s: pd.Series) -> pd.Series:
    return s.map(lambda x: anyascii(x).lower())


_LEAD = re.compile(r"([a-z]+)(\d+)$")   # letters glued in front of a number: "ndeg29" (N°29), "b12"


def tokens(x):
    """transliterated, lowercased word tokens. Letters glued in front of a number are split off it (N°29 -> ndeg 29;
    "12b", "1st" stay whole), and single letters separated only by dots / spaces form one token (L.L.C. -> llc,
    J P -> jp, S.A.S. -> sas, P O Box -> po box), so numbers and initialisms match however they are punctuated."""
    s, out, end, single = anyascii(x).lower(), [], 0, False
    for m in _TOK.finditer(s):
        lead = _LEAD.match(m.group())
        for i, t in enumerate(lead.groups() if lead else (m.group(),)):
            one = len(t) == 1 and t.isalpha()
            if one and single and set(s[end:m.start()] if i == 0 else "") <= {".", " "}:
                out[-1] += t
            else:
                out.append(t)
            single = one
        end = m.end()
    return out


def _expand(s, country, field):
    ab = lexicon()["abbr"]
    out = []
    for x, c in zip(s.values, country.values):
        m = ab.get(c, {}).get(field, {})
        out.append(" ".join(m.get(t, t) for t in tokens(x) if t not in _JUNK))
    return pd.Series(out, index=s.index)


def norm_name(s: pd.Series, country: pd.Series) -> pd.Series:
    """transliterate, lowercase, tokenise, expand the country's learned name abbreviations."""
    return _expand(s, country, "name")


def norm_addr(s: pd.Series, country: pd.Series) -> pd.Series:
    """same with the country's learned address abbreviations (street types, state codes...)."""
    return _expand(s, country, "addr")


def core_name(s: pd.Series, country: pd.Series) -> pd.Series:
    """normalised name without the country's learned legal-form words, repeats collapsed."""
    legal = {c: set(v) for c, v in lexicon()["legal"].items()}
    out = []
    for x, c in zip(s.values, country.values):
        lg, o = legal.get(c, ()), []
        for t in x.split():
            if t not in lg and (not o or o[-1] != t):
                o.append(t)
        out.append(" ".join(o))
    return pd.Series(out, index=s.index)


_VOW = re.compile(r"[aeiouh]")


def skeleton(s: pd.Series) -> pd.Series:
    """consonant skeleton per token: absorbs vowel loss of transliteration (maharashtra -> mrstr)."""
    return s.map(lambda x: " ".join(filter(None, (t[0] + _VOW.sub("", t[1:]) for t in x.split()))))


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
    assert tokens("J.P. Morgan L.L.C., P O Box 12 B") == ["jp", "morgan", "llc", "po", "box", "12", "b"]
    assert tokens("A & B Traders") == ["a", "b", "traders"]
    assert tokens("N°29 R. du Port, 12B, 1st Ave") == ["ndeg", "29", "r", "du", "port", "12b", "1st", "ave"]
    c = pd.Series(["US", "India", "US", "France"])
    print(norm_name(pd.Series(["Raab Modern Treoasubr,y LLC", "व्हाइट बिल्डर्स प्राइवेट लिमिटेड", "wilfordhancock.com", "Grain & Fils"]), c).tolist())
    print(norm_addr(pd.Series(["2670- DUMBLE ST, ALVIN, TX", "12 MG Road, Chennai, TN", "NULL", "63 R. DE DIEPPE, LILLE, Hauts-de-France"]), c).tolist())
    print("ok")
