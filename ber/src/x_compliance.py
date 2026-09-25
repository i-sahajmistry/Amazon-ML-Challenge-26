"""What does going fully compliant touch?
(1) State table: addresses / train pair similarities changed by removing common._STATES (old caches in work/pq/oldnorm
    vs the rebuilt norm2 caches).
(2) anyascii: records that need transliteration, per script, and how often a stdlib-only transliterator (Unicode
    character names + NFKD, no package data) gives the same tokens as anyascii.
  python x_compliance.py"""
import re, unicodedata as ud, numpy as np, pandas as pd
from collections import Counter, defaultdict
from anyascii import anyascii
from common import WORK, load
from harness import truth_arrays

TOK = re.compile(r"[a-z0-9]+")
INDIC = {"DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM"}
VOW = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e", "AI": "ai", "O": "o",
       "OO": "o", "AU": "au", "VOCALIC R": "r", "VOCALIC RR": "r", "VOCALIC L": "l", "VOCALIC LL": "l",
       "CANDRA E": "e", "CANDRA O": "o", "SHORT E": "e", "SHORT O": "o"}
SIGN = {"ANUSVARA": "n", "CANDRABINDU": "n", "VISARGA": "h"}


def std_char(ch):
    """stdlib-only transliteration of one character (Indic: consonants without the inherent vowel, like anyascii)."""
    if ch < "\x80":
        return ch
    if ud.category(ch) == "Nd":
        return str(ud.digit(ch))
    base = "".join(c for c in ud.normalize("NFKD", ch) if not ud.combining(c))
    if base and base.isascii():                                   # é -> e, ﬁ -> fi, full-width forms
        return base
    n = ud.name(ch, "")
    if n.startswith("LATIN"):                                     # ø, œ, æ, ß, ł ...
        m = re.search(r"(?:LETTER|LIGATURE) (.+?)(?: WITH .*)?$", n)
        s = m.group(1) if m else ""
        return {"SHARP S": "ss", "DOTLESS I": "i"}.get(s, s.lower() if s.isalpha() and len(s) <= 2 else "")
    if n.split(" ")[0] not in INDIC:
        return ""
    for key in (" VOWEL SIGN ", " LETTER ", " SIGN "):
        if key in n:
            rest = n.split(key, 1)[1]
            break
    else:
        return ""
    if key == " SIGN ":
        return SIGN.get(rest, "")
    if rest in VOW:
        return VOW[rest]
    c = rest.split()[-1]                                          # KA -> k, TTHA -> th, CHILLU NN -> n
    c = c[:-1] if len(c) > 1 and c.endswith("A") else c
    return re.sub(r"(.)\1+", r"\1", c).lower()


def std(s):
    return "".join(map(std_char, s))


def script(s):
    c = Counter(ud.name(ch, "?").split(" ")[0] for ch in s if ch >= "\x80" and ud.category(ch)[0] in "LM")
    return c.most_common(1)[0][0] if c else "OTHER"


def states():
    print("== state table", flush=True)
    for split in ("train", "test"):
        for src in (1, 2, 3):
            f = f"pq/norm2_{split}_{src}.parquet"
            new = pd.read_parquet(f"{WORK}/{f}", columns=["na"]).na.values
            old = pd.read_parquet(f"{WORK}/pq/oldnorm/norm2_{split}_{src}.parquet", columns=["na"]).na.values
            c = load(split, src).country.values
            print(f"  {split} source{src}: addresses changed " +
                  "  ".join(f"{k} {(new[c == k] != old[c == k]).mean():.2%}" for k in ("US", "India", "France") if (c == k).any()),
                  flush=True)
    s1, other, ts, s1f, rf = truth_arrays()
    rd = lambda d, s: pd.read_parquet(f"{WORK}/pq/{d}norm2_train_{s}.parquet", columns=["na"]).na.values
    a = {d: (rd(d, 1), np.concatenate([rd(d, 2), rd(d, 3)])) for d in ("", "oldnorm/")}
    cand = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
    top = cand[(cand["rank"] == 0) & (ts[cand.rid.values] < 0)]
    rng = np.random.default_rng(0)
    pick = lambda n: rng.choice(n, min(n, 400000), replace=False)
    true = np.flatnonzero(ts >= 0)[pick((ts >= 0).sum())]
    i = pick(len(top))
    for label, r, s in (("true pairs", true, ts[true]), ("distractor vs top-1 S1", top.rid.values[i], top.sid.values[i])):
        jac = {}
        for d, (A, B) in a.items():
            jac[d] = np.array([len(x & y) / max(len(x | y), 1) for x, y in
                               ((set(p.split()), set(q.split())) for p, q in zip(A[s], B[r]))])
        up, dn = (jac["oldnorm/"] > jac[""] + 1e-9), (jac["oldnorm/"] < jac[""] - 1e-9)
        print(f"  {label} ({len(r)}): address Jaccard higher with the table {up.mean():.2%}, lower {dn.mean():.2%}, "
              f"mean change {np.mean(jac['oldnorm/'] - jac['']):+.4f}", flush=True)


def translit():
    print("== anyascii", flush=True)
    per = defaultdict(Counter)
    uniq = set()
    for split in ("train", "test"):
        for src in (1, 2, 3):
            d = load(split, src)
            txt = (d.business_name + " | " + d.business_address).values
            na = np.array([not t.isascii() for t in txt])
            for k in ("US", "India", "France"):
                m = d.country.values == k
                if m.any():
                    per[f"{split} source{src}"][k] = f"{na[m].mean():.2%}"
            uniq.update(t for col in (d.business_name, d.business_address) for t in col.values if not t.isascii())
    for k, v in per.items():
        print(f"  {k}: records with non-ASCII text {dict(v)}", flush=True)
    by = defaultdict(lambda: [0, 0, 0])
    bad = defaultdict(list)
    for s in uniq:
        sc = script(s)
        a, b = TOK.findall(anyascii(s).lower()), TOK.findall(std(s).lower())
        by[sc][0] += 1
        by[sc][1] += a == b
        sk = lambda t: [x[0] + re.sub("[aeiouh]", "", x[1:]) for x in t]
        by[sc][2] += sk(a) == sk(b)
        if a != b and len(bad[sc]) < 4:
            bad[sc].append((s, " ".join(a), " ".join(b)))
    print(f"  unique non-ASCII strings {len(uniq)}; per script: count, same tokens as anyascii, same consonant skeleton",
          flush=True)
    for sc, (n, same, sk) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"    {sc:12s} {n:8d}  {same / n:7.2%}  {sk / n:7.2%}", flush=True)
    for sc, ex in bad.items():
        for s, a, b in ex:
            print(f"    [{sc}] {s!r}\n       anyascii: {a}\n       stdlib:   {b}", flush=True)


if __name__ == "__main__":
    for s, want in (("Société Générale Œuvre", "societe generale oeuvre"), ("१२३", "123"), ("Straße", "strasse")):
        assert std(s).lower() == want, (s, std(s))
    translit()
    states()
