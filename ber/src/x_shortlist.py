"""Blocking size vs recall on the retrieved top-20 (work/cand_train.parquet), records of S1 folds 7-9:
fixed top-k, Sarvesh's gap:d and softmax:T:c (match.py SHORTLIST), and a calibrated shortlist: LightGBM on
retrieval-only features (fit on folds 4-6) giving P(candidate is the record's S1); keep candidates with P >= tau.
Recall = real records whose true S1 is kept; size = kept candidates per record (distractors included) and per S1.
  python x_shortlist.py"""
import numpy as np, pandas as pd, lightgbm as lgb
from common import WORK
from harness import truth_arrays

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet")
c = c[rf[c.rid.values] >= 4].sort_values(["rid", "rank"]).reset_index(drop=True)   # records the embedder never saw
r, sc = c.rid.values, c.score.values.astype(np.float64)
g = c.groupby("rid").score
top1 = g.transform("max").values
c["gap_top1"] = top1 - sc
c["gap_next"] = sc - g.shift(-1).fillna(-1).values
c["top1"] = top1
for d in (0.02, 0.05, 0.1):
    c[f"n{int(d * 100):02d}"] = pd.Series(sc >= top1 - d).groupby(r).transform("sum").values
e = np.exp((sc - top1) / 0.05)
c["soft"] = e / pd.Series(e).groupby(r).transform("sum").values
c["y"] = ts[r] == c.sid.values
fold = rf[r]
F = ["score", "rank", "gap_top1", "gap_next", "top1", "n02", "n05", "n10", "soft"]
tr, ev = np.isin(fold, [4, 5, 6]) & (r % 4 == 0), np.isin(fold, [7, 8, 9])   # fit on a quarter of 4-6
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, verbose=-1,
                   num_threads=32), lgb.Dataset(c.loc[tr, F], c.y.values[tr]), 300)
E = c[ev].copy()
E["P"] = m.predict(E[F], num_threads=32)
real = np.unique(r[ev & (ts[r] >= 0)])
n_rec, n_s1 = len(np.unique(r[ev])), (s1f >= 7).sum()


def report(name, keep):
    k = E[keep]
    hit = np.isin(real, k.rid.values[k.y.values])
    print(f"{name:22s} recall {hit.mean():.4%}   per record {len(k) / n_rec:5.2f}   per S1 {len(k) / n_s1:6.2f}", flush=True)


print(f"records {n_rec} (real {len(real)}), S1 {n_s1}; recall ceiling of the top-20 below", flush=True)
for k in (1, 2, 3, 5, 10, 20):
    report(f"top-{k}", E["rank"].values < k)
for d in (0.05, 0.1, 0.15):
    report(f"gap:{d}", E.gap_top1.values <= d)
s = E.soft.values
before = pd.Series(s).groupby(E.rid.values).cumsum().values - s
for cut in (0.9, 0.95, 0.99):
    report(f"softmax:0.05:{cut}", before < cut)
for tau in (0.3, 0.1, 0.05, 0.02, 0.01, 0.005):
    report(f"calibrated P>={tau}", E.P.values >= tau)
