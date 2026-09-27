"""E13 / E14 on US / India validation (bge stack, _v10bpw; entity folds 8-9; distractors weighted; macro F0.5 per S1).
E14: where F0.5 is lost (per S1 kind). E13: S1-level rules that exploit the per-S1 metric, cross-fitted fold 8 <-> 9:
  (a) rescue: an S1 with no accepted record takes its best claimant if that q >= t (t < 0.70)
  (b) lone drop: an S1 whose only accepted record has q < t (t > 0.70) is left empty."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]; n = len(s1f)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet")
sid, q, y, wt = v.sid.values, v.q.values, v.y.values.astype(bool), v.wt.values
isd = ts[v.rid.values] < 0
def F(acc, e): return wscore(sid, acc, y, wt, T, e)
base = q >= 0.70
# ---- E14: loss breakdown at 0.70
tp = np.bincount(sid[base & y], minlength=n); fpd = np.bincount(sid[base & isd], weights=wt[base & isd], minlength=n)
fpw = np.bincount(sid[base & ~y & ~isd], minlength=n); t_ = T
pr = np.divide(tp, tp + fpd + fpw, out=np.zeros(n), where=tp + fpd + fpw > 0); rc = tp / np.maximum(t_, 1)
f = np.where(t_ == 0, (tp + fpd + fpw == 0).astype(float), np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros(n), where=tp > 0))
e = ents; loss = 1 - f[e]
kinds = {"singleton S1 given a record": (t_[e] == 0) & (tp[e] + fpd[e] + fpw[e] > 0),
         "S1 left empty but has matches": (t_[e] > 0) & (tp[e] + fpd[e] + fpw[e] == 0),
         "distractor merged": (t_[e] > 0) & (fpd[e] > 0),
         "wrong-S1 record merged": (t_[e] > 0) & (fpd[e] == 0) & (fpw[e] > 0),
         "missed some matches only": (t_[e] > 0) & (fpd[e] == 0) & (fpw[e] == 0) & (tp[e] < t_[e]) & (tp[e] > 0)}
print(f"E14 loss breakdown, US+India validation F0.5 {F(base, e):.5f} (total loss {loss.mean():.5f})", flush=True)
for k, m in kinds.items():
    print(f"  {k:32s} S1s {m.mean():.4f}   F0.5 lost {loss[m].sum() / len(e):.5f}", flush=True)
# ---- E13 rules
best_q = pd.Series(q).groupby(sid).max(); best_row = pd.Series(np.arange(len(q))).groupby(sid).apply(lambda i: i.values[np.argmax(q[i.values])])
n_acc = np.bincount(sid[base], minlength=n)
def rescue(t):
    acc = base.copy()
    emp = np.flatnonzero((n_acc == 0))
    emp = emp[np.isin(emp, best_row.index.values)]
    r = best_row.reindex(emp).values.astype(int); acc[r[q[r] >= t]] = True
    return acc
def lone(t):
    acc = base.copy()
    one = np.flatnonzero(n_acc == 1)
    rows = np.flatnonzero(base & np.isin(sid, one))
    acc[rows[q[rows] < t]] = False
    return acc
fold = s1f
for name, fn, grid in (("rescue", rescue, [0.4, 0.5, 0.55, 0.6, 0.65]), ("lone drop", lone, [0.75, 0.8, 0.85, 0.9, 0.95])):
    accs = {t: fn(t) for t in grid}
    gains = []
    for fit, ev in ((8, 9), (9, 8)):
        ef, ee = ents[fold[ents] == fit], ents[fold[ents] == ev]
        sc = {t: F(a, ef) - F(base, ef) for t, a in accs.items()}; b = max(sc, key=sc.get)
        g = F(accs[b], ee) - F(base, ee); gains.append(g)
        print(f"E13 {name}: fit fold {fit} best t={b} (fit gain {sc[b]:+.5f}) -> fold {ev} gain {g:+.5f}   all t on fit: " +
              " ".join(f"{t}:{x:+.5f}" for t, x in sc.items()), flush=True)
    print(f"E13 {name}: cross-fitted mean gain {np.mean(gains):+.5f}", flush=True)
