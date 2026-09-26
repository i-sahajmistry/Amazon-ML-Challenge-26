"""How a record's name differs from its best S1's (core names, typo-tolerant token alignment): same / insert /
delete / swap, crossed with house-number agreement. Train (S1 folds 8-9, labelled, v7 stage 1): share of true records
and acceptance per type. Test (v7w stage 2): share of records per type and acceptance, per country. Shows which
edits the model mis-handles on an unseen country.
  python x_edits.py"""
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from common import WORK
from match import normed
from harness import truth_arrays


def etype(a, b):
    A, B = a.split(), b.split()
    ua = sum(not any(fuzz.ratio(t, u) >= 80 for u in B) for t in A)
    ub = sum(not any(fuzz.ratio(t, u) >= 80 for u in A) for t in B)
    return "same" if ua + ub == 0 else "insert" if ua == 0 else "delete" if ub == 0 else "swap"


def ntype(a, b):
    a, b = a.split()[:1], b.split()[:1]
    return "num?" if not a or not b else "num=" if a == b else "num≠"


def table(split, rid, sid, extra):
    n1, n2 = normed(split, 1), pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    t = pd.DataFrame({"e": [etype(a, b) for a, b in zip(n1.cn.values[sid], n2.cn.values[rid])],
                      "n": [ntype(a, b) for a, b in zip(n1.num.values[sid], n2.num.values[rid])]}, **extra)
    t["type"] = t.e + " " + t.n
    return t


rng = np.random.default_rng(0)
s1, other, ts, s1f, rf = truth_arrays()
k = pd.read_parquet(f"{WORK}/x/oof5_train_all.parquet").sort_values(["rid", "p"], ascending=[True, False])
k = k[~k.rid.duplicated()]
k = k[(s1f[k.sid.values] >= 8) & (rf[k.rid.values] >= 4)]
k = k.iloc[rng.choice(len(k), 300000, replace=False)]
t = table("train", k.rid.values, k.sid.values, {})
t["true"] = ts[k.rid.values] == k.sid.values
t["acc"] = k.p.values >= 0.5
g = t.groupby("type")
print("TRAIN folds 8-9 (v7 stage 1 argmax, p >= 0.5)")
print(pd.DataFrame({"share": g.size() / len(t), "true_share": g.true.mean(), "accept_true": g.apply(
    lambda x: x.acc[x.true].mean()), "accept_false": g.apply(lambda x: x.acc[~x.true].mean())}).round(4).to_string(),
      flush=True)
d = pd.read_parquet(f"{WORK}/x/test_q_allw.parquet")
for c in ("US", "India", "France"):
    x = d[d.c == c]
    x = x.iloc[rng.choice(len(x), 200000, replace=False)]
    t = table("test", x.rid.values, x.sid.values, {})
    t["acc"] = x.q.values >= 0.7
    t["unsure"] = (x.q.values > 0.02) & (x.q.values < 0.98)
    g = t.groupby("type")
    print(f"TEST {c} (v7w stage 2, q >= 0.7)")
    print(pd.DataFrame({"share": g.size() / len(t), "accept": g.acc.mean(), "uncertain": g.unsure.mean()})
          .round(4).to_string(), flush=True)
