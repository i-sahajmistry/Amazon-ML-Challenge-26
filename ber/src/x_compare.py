"""Compare test submissions without labels:
  - per country: matches per S1, empty S1s, and how far the set-size distribution is from the train truth's;
  - pairwise macro F0.5 of one file (row) scored as if another file (column) were the truth.
  python x_compare.py"""
import numpy as np, pandas as pd
from common import ROOT, load

FILES = {"v4": "output_v4/matching_results.tsv", "v5": "output_v5/matching_results.tsv",
         "v5f": "output_v5f/matching_results.tsv", "sarvesh_old": "teammates/sarvesh_old.tsv",
         "sarvesh_gap": "teammates/sarvesh_gap.tsv", "mohanish": "teammates/mohanish_fix.tsv"}
s1 = load("test", 1)
IDX, C, N = pd.Index(s1.entity_id), s1.country.values, len(s1)


def read(path):
    d = pd.read_csv(f"{ROOT}/{path}", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    assert len(d) == N and IDX.get_indexer(d.iloc[:, 0]).min() >= 0
    e = pd.DataFrame({"s": IDX.get_indexer(d.iloc[:, 0]), "r": d.iloc[:, 1].str.split(",")}).explode("r")
    e = e[e.r != ""]
    o = pd.Series(e.s.values.astype(np.int64), index=e.r.values)
    if o.index.duplicated().any():
        print(f"  {path}: {o.index.duplicated().sum()} records claimed by 2+ S1s (kept first)")
        o = o[~o.index.duplicated()]
    return o


def hist(sz):
    return np.bincount(np.minimum(sz, 7), minlength=8) / len(sz)


def f05(a, b):
    """per-S1 F0.5 of prediction a against 'truth' b (record -> S1 index)"""
    com = a.index.intersection(b.index)
    av = a.reindex(com).values
    tp = np.bincount(av[av == b.reindex(com).values], minlength=N)
    pa, tb = np.bincount(a.values, minlength=N), np.bincount(b.values, minlength=N)
    P = np.divide(tp, pa, out=np.zeros(N), where=pa > 0)
    R = np.divide(tp, tb, out=np.zeros(N), where=tb > 0)
    f = np.divide(1.25 * P * R, 0.25 * P + R, out=np.zeros(N), where=tp > 0)
    return np.where((pa == 0) & (tb == 0), 1.0, f)


gt = load("train", "ground_truth")
tsz = gt.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0).values
truth = hist(tsz)
print("train truth: per S1 %.3f  empty %.2f%%  sizes 0..7+ %s" % (tsz.mean(), 100 * (tsz == 0).mean(),
                                                               np.round(truth, 3).tolist()), flush=True)
O = {k: read(v) for k, v in FILES.items()}
print("\nfile           accepted   " + "   ".join(f"{c}: per S1 / empty / TV" for c in ("US", "India", "France")))
for k, o in O.items():
    sz = np.bincount(o.values, minlength=N)
    row = [f"{sz[C == c].mean():.3f} / {100 * (sz[C == c] == 0).mean():.2f}% / "
           f"{0.5 * np.abs(hist(sz[C == c]) - truth).sum():.3f}" for c in ("US", "India", "France")]
    print(f"{k:13s} {len(o):9d}   " + "   ".join(row), flush=True)
for c in (None, "US", "India", "France"):
    m = np.ones(N, bool) if c is None else C == c
    print(f"\nmacro F0.5 of row scored against column as truth ({c or 'all'})")
    print(" " * 13 + "".join(f"{k:>13s}" for k in O))
    for a in O:
        print(f"{a:13s}" + "".join(f"{f05(O[a], O[b])[m].mean():13.4f}" for b in O), flush=True)
