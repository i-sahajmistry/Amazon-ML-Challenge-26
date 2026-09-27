"""E25: does a second-model veto (France-style min) help US / India? Validation folds 8-9, distractors weighted.
Base = bge stack q >= 0.70 (+ rescue). Veto: also require the main stack's q (same record, same S1) >= v; a record whose
main-stack best S1 differs counts as main q = 0 (as x_final's :min). v cross-fitted fold 8 <-> 9. Also the plain
min(bge, main) >= t for t in a grid, and the veto applied only above / below a bge-q band."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]; n = len(s1f)
X = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"
b = pd.read_parquet(f"{X}/val_q_v10bpw.parquet"); m = pd.read_parquet(f"{X}/val_q_v10pw.parquet", columns=["rid", "sid", "q"])
d = b.merge(m, on="rid", how="left", suffixes=("", "_m"))
qm = np.where(d.sid_m.values == d.sid.values, d.q_m.values, 0.0)
print(f"rows {len(d):,}; main-stack best S1 differs for {(d.sid_m.values != d.sid.values).mean():.4f}", flush=True)
sid, q, y, wt = d.sid.values, d.q.values, d.y.values.astype(bool), d.wt.values
def F(acc, e): return wscore(sid, acc, y, wt, T, e)
top = pd.Series(q).groupby(sid).transform("max").values
isbest = (q == top) & ~pd.Series(q == top).groupby(sid).cumsum().gt(1).values
def with_rescue(acc):
    na = np.bincount(sid[acc], minlength=n); return acc | (isbest & (na[sid] == 0) & (q >= 0.5))
base = with_rescue(q >= 0.70)
print(f"base (bge 0.70 + rescue) {F(base, ents):.5f}", flush=True)
cands = {}
for v in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
    cands[f"veto main<{v}"] = with_rescue((q >= 0.70) & (qm >= v))
for t in (0.6, 0.65, 0.7):
    cands[f"min(bge,main)>={t}"] = with_rescue(np.minimum(q, qm) >= t)
for v in (0.1, 0.3):
    for hi in (0.9, 0.99):
        cands[f"veto main<{v} only if bge<{hi}"] = with_rescue((q >= 0.70) & ((qm >= v) | (q >= hi)))
cands["mean(bge,main)>=0.70"] = with_rescue((q + qm) / 2 >= 0.70)
for k, a in cands.items():
    print(f"{k:32s} all {F(a, ents) - F(base, ents):+.6f}  accepted {a.sum() - base.sum():+,}", flush=True)
g = []
for fit, evf in ((8, 9), (9, 8)):
    ef, ee = ents[s1f[ents] == fit], ents[s1f[ents] == evf]
    sc = {k: F(a, ef) - F(base, ef) for k, a in cands.items()}; kb = max(sc, key=sc.get)
    g.append(F(cands[kb], ee) - F(base, ee)); print(f"fit {fit}: best '{kb}' ({sc[kb]:+.6f}) -> fold {evf} {g[-1]:+.6f}", flush=True)
print(f"E25 cross-fitted mean gain over bge 0.70 + rescue: {np.mean(g):+.6f}", flush=True)
