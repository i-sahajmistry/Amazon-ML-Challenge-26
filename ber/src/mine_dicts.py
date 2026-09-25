"""Mine normalisation dictionaries from the data, one per country label (an open set: every label found in the
data gets its own dictionaries; nothing is written by hand for any country).

Unit: confident record -> S1 pairs from the bi-encoder retrieval (top-1 cosine >= CONF and >= GAP above the
runner-up), train and test pooled, no labels used. Tokens: raw text, anyascii + lowercase, [a-z0-9]+.
  abbr    a one-sided token "a" and a longer one-sided token "b" in the same pair, same first letter, a is a
          subsequence of b (rd/road, tx/texas, r/rue, mh/maharashtra). Kept if seen >= MIN_N times and it explains
          >= MIN_PREC of a's one-sided occurrences; names also need len(a) >= 2.
  legal   short name tokens (<= 4 letters, as legal abbreviations are; "group" is a distractor word) that are often on one side only, in BOTH directions (llc, sas, ltd).
  python mine_dicts.py   -> work/dicts/abbr.parquet (country, field, short, long, n, prec), work/dicts/legal.parquet"""
import os, re, numpy as np, pandas as pd
from collections import Counter
from multiprocessing import get_context
from anyascii import anyascii
from common import WORK, load

CONF, GAP = float(os.environ.get("CONF", 0.90)), float(os.environ.get("GAP", 0.05))
MIN_N, MIN_PREC = int(os.environ.get("MIN_N", 50)), float(os.environ.get("MIN_PREC", 0.5))
NP = int(os.environ.get("NPROC", 16))
OUT = f"{WORK}/dicts"
_TOK = re.compile(r"[a-z0-9]+")
_G = {}


def _tok_many(xs):
    return [" ".join(_TOK.findall(anyascii(x).lower())) for x in xs]


def tokenize(xs):
    step = len(xs) // (NP * 8) + 1
    with get_context("fork").Pool(NP) as p:
        parts = p.map(_tok_many, [xs[i:i + step] for i in range(0, len(xs), step)])
    return np.array([t for part in parts for t in part], dtype=object)


def _subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def _mine(bounds):
    lo, hi = bounds
    ra, sa = _G["r"], _G["s"]
    one, pair, present, one_r, one_s = Counter(), Counter(), Counter(), Counter(), Counter()
    for i in range(lo, hi):
        R, S = set(ra[i].split()), set(sa[i].split())
        ar, bs = R - S, S - R
        present.update(R | S); one_r.update(ar); one_s.update(bs)
        for x, ys in ((ar, bs), (bs, ar)):
            for a in x:
                if len(a) > 5 or a.isdigit():
                    continue
                for b in ys:
                    if len(b) > len(a) and b[0] == a[0] and not b.isdigit() and _subseq(a, b):
                        pair[(a, b)] += 1
    return pair, present, one_r, one_s


def confident_pairs(split):
    s1, other = load(split, 1), pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    k = pd.read_parquet(f"{WORK}/cand_{split}.parquet", columns=["rid", "sid", "score", "rank"])
    top = k[k["rank"] == 0].set_index("rid")
    sec = k[k["rank"] == 1].set_index("rid").score.reindex(top.index).fillna(0).values
    top = top[(top.score >= CONF) & (top.score.values - sec >= GAP)]
    rid, sid = top.index.values, top.sid.values
    return {f: (other[f"business_{f}"].values[rid], s1[f"business_{f}"].values[sid]) for f in ("name", "address")}, \
        other.country.values[rid]


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [confident_pairs(sp) for sp in ("train", "test")]
    ctry = np.concatenate([c for _, c in parts])
    abbr, legal = [], []
    for field in ("name", "address"):
        R = tokenize(np.concatenate([p[field][0] for p, _ in parts]).tolist())
        S = tokenize(np.concatenate([p[field][1] for p, _ in parts]).tolist())
        for c in sorted(set(ctry)):
            m = ctry == c
            _G.update(r=R[m], s=S[m])
            n = int(m.sum()); step = n // (NP * 6) + 1
            with get_context("fork").Pool(NP) as p:
                res = p.map(_mine, [(i, min(i + step, n)) for i in range(0, n, step)])
            pair, present, one_r, one_s = Counter(), Counter(), Counter(), Counter()
            for a, b, x, y in res:
                pair.update(a); present.update(b); one_r.update(x); one_s.update(y)
            rows = [(a, b, v, v / max(one_r[a] + one_s[a], 1)) for (a, b), v in pair.items() if v >= MIN_N]
            d = pd.DataFrame(rows, columns=["short", "long", "n", "prec"])
            d = d[(d.prec >= MIN_PREC) & ((field == "address") | (d.short.str.len() >= 2))]
            d = d.sort_values(["short", "n"], ascending=[True, False]).drop_duplicates("short")
            abbr.append(d.assign(country=c, field=field))
            print(f"{c} {field}: {n} pairs, {len(d)} abbreviations: "
                  + ", ".join(f"{a}->{b}" for a, b in d.sort_values('n', ascending=False)[["short", "long"]].head(25).values),
                  flush=True)
            if field == "name":
                t = pd.DataFrame([(w, present[w], one_r[w], one_s[w]) for w in present if present[w] >= 500],
                                 columns=["tok", "present", "only_rec", "only_s1"])
                t["edit"] = (t.only_rec + t.only_s1) / t.present
                t["balance"] = np.minimum(t.only_rec, t.only_s1) / np.maximum(t.only_rec + t.only_s1, 1)
                t = t[(t.edit >= 0.3) & (t.balance >= 0.15) & (t.tok.str.len() <= 4) & t.tok.str.isalpha()]
                legal.append(t.assign(country=c))
                print(f"{c} legal-like tokens: {', '.join(t.sort_values('present', ascending=False).tok)}", flush=True)
    pd.concat(abbr, ignore_index=True).to_parquet(f"{OUT}/abbr.parquet")
    pd.concat(legal, ignore_index=True).to_parquet(f"{OUT}/legal.parquet")


if __name__ == "__main__":
    main()
