"""E15 on US / India validation (bge stack): recall rules for the biggest loss (missed matches, 66% of the loss).
  sibling: a record with t <= q < 0.70 whose best S1 already has an accepted record with q >= A is accepted.
  plus E13 rescue (empty S1 takes its best claimant if q >= r), alone and combined. Cross-fitted fold 8 <-> 9."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]; n = len(s1f)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet")
sid, q, y, wt = v.sid.values, v.q.values, v.y.values.astype(bool), v.wt.values
def F(acc, e): return wscore(sid, acc, y, wt, T, e)
base = q >= 0.70
top = pd.Series(q).groupby(sid).max().reindex(np.arange(n)).fillna(0).values
n_acc = np.bincount(sid[base], minlength=n)
order = np.lexsort((-q, sid)); first = np.r_[True, sid[order][1:] != sid[order][:-1]]; best_row = order[first]
def sibling(t, A):
    acc = base.copy(); acc |= (q >= t) & (q < 0.70) & (top[sid] >= A)
    return acc
def rescue(acc, r):
    acc = acc.copy(); b = best_row[(n_acc[sid[best_row]] == 0) & (q[best_row] >= r)]; acc[b] = True
    return acc
cands = {}
for t in (0.4, 0.5, 0.6, 0.65):
    for A in (0.9, 0.95, 0.99):
        cands[("sib", t, A)] = sibling(t, A)
for r in (0.5,):
    cands[("rescue", r)] = rescue(base, r)
    for t in (0.5, 0.6):
        for A in (0.95, 0.99):
            cands[("sib+rescue", t, A, r)] = rescue(sibling(t, A), r)
for group in ("sib", "sib+rescue", "rescue"):
    keys = [k for k in cands if k[0] == group]
    gains = []
    for fit, ev in ((8, 9), (9, 8)):
        ef, ee = ents[s1f[ents] == fit], ents[s1f[ents] == ev]
        sc = {k: F(cands[k], ef) - F(base, ef) for k in keys}; b = max(sc, key=sc.get)
        g = F(cands[b], ee) - F(base, ee); gains.append(g)
        print(f"E15 {group}: fit fold {fit} best {b[1:]} (fit {sc[b]:+.5f}) -> fold {ev} {g:+.5f}", flush=True)
        if group == "sib":
            print("   all on fit: " + " ".join(f"{k[1:]}:{x:+.5f}" for k, x in sc.items()), flush=True)
    print(f"E15 {group}: cross-fitted mean gain {np.mean(gains):+.5f}", flush=True)
