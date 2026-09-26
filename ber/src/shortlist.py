"""Calibrated blocking model for match.py SHORTLIST=model:tau. A small LightGBM on retrieval-only features of each
record's top-20 (cosine, rank, gaps to the best / next candidate, how many candidates are nearly tied, softmax share)
predicts whether a candidate is the record's S1. No text or language features, so it transfers to any country.
Fit on train records of S1 folds 4-6 (the bi-encoder never saw them); recall / size by tau reported on folds 7-9,
next to fixed top-k and Sarvesh's gap / softmax rules.
  python shortlist.py     -> work/shortlist.txt"""
import numpy as np, pandas as pd, lightgbm as lgb
from common import WORK
from harness import truth_arrays
from match import retrieval_feats, SHORT_F

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet")
c = retrieval_feats(c[rf[c.rid.values] >= 4].sort_values(["rid", "rank"]).reset_index(drop=True))
r = c.rid.values
y = ts[r] == c.sid.values
tr, ev = np.isin(rf[r], [4, 5, 6]) & (r % 4 == 0), np.isin(rf[r], [7, 8, 9])
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, verbose=-1,
                   num_threads=32), lgb.Dataset(c.loc[tr, SHORT_F], y[tr]), 300)
m.save_model(f"{WORK}/shortlist.txt")
E = c[ev].assign(P=m.predict(c.loc[ev, SHORT_F], num_threads=32), y=y[ev])
real = np.unique(r[ev & (ts[r] >= 0)])
n_rec, n_s1 = len(np.unique(r[ev])), (s1f >= 7).sum()


def report(name, keep):
    k = E[keep]
    hit = np.isin(real, k.rid.values[k.y.values]).mean()
    print(f"{name:22s} true S1 kept {hit:.4%}   per record {len(k) / n_rec:5.2f}   per S1 {len(k) / n_s1:6.2f}", flush=True)


for k in (1, 5, 20):
    report(f"top-{k}", E["rank"].values < k)
report("gap:0.1", E.gap_top1.values <= 0.1)
before = pd.Series(E.soft.values).groupby(E.rid.values).cumsum().values - E.soft.values
report("softmax:0.05:0.95", before < 0.95)
for tau in (0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001):
    report(f"model:{tau}", E.P.values >= tau)
