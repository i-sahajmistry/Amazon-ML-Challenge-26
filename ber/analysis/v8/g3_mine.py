"""Can the hand-written dictionaries (state codes, street abbreviations, legal forms) be MINED from the data, per
country, without labels? That would replace every country-specific list with one generic procedure.

Per (split, country), on confident top-1 retrieved pairs (cosine >= CONF and gap to the next candidate >= GAP):
tokens of the raw transliterated text (anyascii, lowercase, [a-z0-9]+, no normalisation), name and address apart.
  - abbreviations: a token only in one side ("a") and a longer token only in the other ("b") with the same first
    letter and a as a subsequence of b (st/street, tx/texas, r/rue). Kept if count >= 30 and precision
    (share of a's one-sided occurrences explained by b) >= 0.25.
  - symmetric noise tokens (legal forms): often present on one side only, in BOTH directions.
Train: precision of the confident pairs against the labels. Writes gen/abbr_{split}_{country}.parquet.
  python g3_mine.py"""
import os, re, numpy as np, pandas as pd
from collections import Counter
from multiprocessing import get_context
Pool = get_context("fork").Pool  # py3.14 defaults to forkserver; workers must inherit _G
from anyascii import anyascii
from common import WORK, load
from harness import truth_arrays

OUT = os.environ.get("GEN_DIR", f"{WORK}/gen")
CONF, GAP = float(os.environ.get("CONF", 0.90)), float(os.environ.get("GAP", 0.05))
_TOK = re.compile(r"[a-z0-9]+")
_G = {}


def toks(x):
    return _TOK.findall(anyascii(x).lower()) if isinstance(x, str) else []


def _tok_many(xs):
    return [" ".join(toks(x)) for x in xs]


def tokenize(series):
    xs = series.tolist()
    step = len(xs) // 256 + 1
    with Pool(16) as p:
        parts = p.map(_tok_many, [xs[i:i + step] for i in range(0, len(xs), step)])
    return np.array([t for part in parts for t in part], dtype=object)


def subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def _mine(bounds):
    lo, hi = bounds
    ra, sa = _G["r"], _G["s"]
    one_r, one_s, pair, present = Counter(), Counter(), Counter(), Counter()
    for i in range(lo, hi):
        R, S = set(ra[i].split()), set(sa[i].split())
        ar, bs = R - S, S - R
        present.update(R | S)
        one_r.update(ar); one_s.update(bs)
        for x, ys in ((ar, bs), (bs, ar)):          # abbreviation on either side
            for a in x:
                if len(a) > 5 or a.isdigit():
                    continue
                for b in ys:
                    if len(b) > len(a) and b[0] == a[0] and not b.isdigit() and subseq(a, b):
                        pair[(a, b)] += 1
    return one_r, one_s, pair, present


def run(split):
    s1, other = load(split, 1), pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    k = pd.read_parquet(f"{WORK}/cand_{split}.parquet", columns=["rid", "sid", "score", "rank"])
    top = k[k["rank"] == 0].set_index("rid")
    sec = k[k["rank"] == 1].set_index("rid").score
    top = top.assign(gap=top.score - sec.reindex(top.index).fillna(0).values)
    conf = top[(top.score >= CONF) & (top.gap >= GAP)]
    rid, sid = conf.index.values, conf.sid.values
    if split == "train":
        ts = truth_arrays()[2]
        print(f"{split}: confident pairs {len(rid)} of {len(top)}; label precision {(ts[rid] == sid).mean():.4f}", flush=True)
    else:
        print(f"{split}: confident pairs {len(rid)} of {len(top)}", flush=True)
    ctry = other.country.values[rid]
    for field in ("business_name", "business_address"):
        R, S = tokenize(other[field].iloc[rid]), tokenize(s1[field].iloc[sid])
        for c in sorted(set(ctry)):
            m = ctry == c
            _G.update(r=R[m], s=S[m])
            n = int(m.sum()); step = n // 96 + 1
            with Pool(16) as p:
                parts = p.map(_mine, [(i, min(i + step, n)) for i in range(0, n, step)])
            one_r, one_s, pair, present = Counter(), Counter(), Counter(), Counter()
            for a, b, c2, d in parts:
                one_r.update(a); one_s.update(b); pair.update(c2); present.update(d)
            rows = [(a, b, v, v / max(one_r[a] + one_s[a], 1)) for (a, b), v in pair.items() if v >= 30]
            ab = pd.DataFrame(rows, columns=["short", "long", "n", "prec"]).sort_values("n", ascending=False)
            ab = ab[ab.prec >= 0.25]
            # best expansion per short token
            ab = ab.sort_values(["short", "n"], ascending=[True, False]).drop_duplicates("short").sort_values("n", ascending=False)
            ab.to_parquet(f"{OUT}/abbr_{split}_{c}_{field[9:]}.parquet")
            sym = pd.DataFrame([(t, present[t], one_r[t], one_s[t]) for t in present if present[t] >= 500],
                               columns=["tok", "present", "only_rec", "only_s1"])
            sym["edit_rate"] = (sym.only_rec + sym.only_s1) / sym.present
            sym["balance"] = np.minimum(sym.only_rec, sym.only_s1) / np.maximum(sym.only_rec + sym.only_s1, 1)
            noise = sym[(sym.edit_rate >= 0.3) & (sym.balance >= 0.15)].sort_values("present", ascending=False)
            print(f"\n== {split} {c} {field[9:]}: pairs {n}; mined abbreviations {len(ab)} (top 40):", flush=True)
            print("  " + ", ".join(f"{a}->{b} ({v}, {p:.2f})" for a, b, v, p in ab.head(40).itertuples(index=False)), flush=True)
            print(f"  symmetric noise tokens (edit rate >= 0.3, both directions): "
                  + ", ".join(f"{t} ({r:.2f})" for t, r in zip(noise.tok.head(40), noise.edit_rate.head(40))), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for sp in ("train", "test"):
        run(sp)
