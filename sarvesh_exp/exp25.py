"""E34: per-record-type thresholds on US / India validation (bge stack, + rescue as base). Types: name-only with a
core name shared by >= 2 S1s (amb), name-only unique name (uniq), name-only name matching no S1 (typo), invented name,
normal. One type's threshold moved at a time (others 0.70), grid incl. 1.01 (= reject all of that type);
cross-fitted fold 8 <-> 9; then all best-per-type together, cross-fitted."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load
from harness import truth_arrays, wscore
from match import normed
W_ = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
c2 = pd.concat([load("train", 2), load("train", 3)], ignore_index=True).country.values
cnt = pd.Series(n1.cn.values + "|" + s1.country.values).value_counts()
amb = cnt.reindex(n2.cn.values + "|" + c2).fillna(0).values
vocab = set(" ".join(n1.cn.values).split())
inv = np.array([bool(x) and all(t not in vocab for t in x.split()) for x in n2.cn.values])
empty = n2.na.values == ""
typ = np.select([empty & (amb >= 2), empty & (amb == 1), empty, inv], ["amb", "uniq", "typo", "invented"], "normal")
v = pd.read_parquet(f"{W_}/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"])
sid, q, y, wt = v.sid.values, v.q.values, v.y.values.astype(bool), v.wt.values; ty = typ[v.rid.values]
def F(acc, e): return wscore(sid, acc, y, wt, T, e)
top = pd.Series(q).groupby(sid).transform("max").values
isbest = (q == top) & ~pd.Series(q == top).groupby(sid).cumsum().gt(1).values
def decide(th):
    t = np.full(len(q), 0.70)
    for k, x in th.items(): t[ty == k] = x
    a = q >= t; na = np.bincount(sid[a], minlength=n)
    return a | (isbest & (na[sid] == 0) & (q >= 0.5) & (t <= 1.0))
base = decide({})
print(f"base (0.70 + rescue) {F(base, ents):.5f}", flush=True)
for k in ("amb", "uniq", "typo", "invented", "normal"):
    m = ty == k; a = base & m
    print(f"type {k:8s}: rows {m.sum():>9,}  accepted {a.sum():>9,}  precision of accepted (unweighted) {y[a].mean() if a.any() else 0:.4f}", flush=True)
grid = (0.5, 0.6, 0.65, 0.75, 0.8, 0.9, 0.95, 0.99, 1.01)
best = {}
for k in ("amb", "uniq", "typo", "invented", "normal"):
    accs = {t: decide({k: t}) for t in grid}; g = []
    for fit, ev in ((8, 9), (9, 8)):
        ef, ee = ents[s1f[ents] == fit], ents[s1f[ents] == ev]
        sc = {t: F(a, ef) - F(base, ef) for t, a in accs.items()}; b = max(sc, key=sc.get)
        if sc[b] <= 0: b = 0.70
        g.append(F(accs[b], ee) - F(base, ee) if b != 0.70 else 0.0); best[(k, fit)] = b
    print(f"type {k:8s}: fold8 best t {best[(k, 8)]}, fold9 best t {best[(k, 9)]}; cross-fitted {np.mean(g):+.6f}   all t: " +
          " ".join(f"{t}:{F(a, ents) - F(base, ents):+.6f}" for t, a in accs.items()), flush=True)
g = []
for fit, ev in ((8, 9), (9, 8)):
    ee = ents[s1f[ents] == ev]; a = decide({k: best[(k, fit)] for k in ("amb", "uniq", "typo", "invented", "normal")})
    g.append(F(a, ee) - F(base, ee))
print(f"E34 all types together, cross-fitted mean gain: {np.mean(g):+.6f}", flush=True)
