"""E12: consistent grouping + cross-fitted threshold per group.
Group = number of strong claimants of the S1 (rows whose best S1 it is with q >= 0.5; unweighted), counted the same way
on validation and on test. (a) validation F0.5 per group and test-like reweighting; (b) threshold per group fitted on
entity fold 8 and scored on fold 9 and vice versa (honest gain); (c) that rule's test-like gain."""
import os, zlib, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
from common import load
O = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); cty = s1.country.values
v = pd.read_parquet(f"{O}/val_q_v10bpw.parquet"); y = v.y.values.astype(bool)
t = pd.read_parquet(f"{O}/test_q_v10bplw.parquet"); tc = load("test", 1).country.values
t = t[np.isin(tc[t.sid.values], ["US", "India"])]
CAP = 6
def grp(sid, q, n):
    return np.minimum(np.bincount(sid[q >= 0.5], minlength=n), CAP)
gv, gt = grp(v.sid.values, v.q.values, len(s1f)), grp(t.sid.values, t.q.values, len(tc))
ents = np.where(s1f >= 8)[0]
THRS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def f(acc, e): return wscore(v.sid.values, acc, y, v.wt.values, T, e)
base_acc = v.q.values >= 0.7
for c in ["US", "India", None]:
    e_all = ents if c is None else ents[cty[ents] == c]
    t_s1 = np.where(np.isin(tc, ["US", "India"] if c is None else [c]))[0]
    fa = f(base_acc, e_all); est = 0
    print(f"\n== {c or 'US+India'}: validation {fa:.5f}", flush=True)
    print(" strong claimants  val share  test share  val F0.5", flush=True)
    for k in range(CAP + 1):
        e = e_all[gv[e_all] == k]; sv, st = len(e) / len(e_all), (gt[t_s1] == k).mean()
        fk = f(base_acc, e) if len(e) else np.nan
        if len(e): est += st * fk
        print(f"  {k}{'+' if k == CAP else ' '}              {sv:.4f}     {st:.4f}     {fk:.5f}", flush=True)
    print(f"  test-like (reweighted): {est:.5f}  ({est - fa:+.5f})", flush=True)
# cross-fitted threshold per group: pick on one entity fold, score on the other
fold = np.full(len(s1f), -1); fold[ents] = s1f[ents]
row_g = gv[v.sid.values]
tot = {}
for fit, ev in ((8, 9), (9, 8)):
    e_fit, e_ev = ents[fold[ents] == fit], ents[fold[ents] == ev]
    best = {}
    for k in range(CAP + 1):
        ek = e_fit[gv[e_fit] == k]
        if len(ek) < 500: best[k] = 0.7; continue
        sc = {th: f(v.q.values >= th, ek) for th in THRS}; best[k] = max(sc, key=sc.get)
    thr_row = np.array([best[k] for k in row_g])
    f0, f1 = f(base_acc, e_ev), f(v.q.values >= thr_row, e_ev)
    tot[ev] = (f0, f1, best)
    print(f"\nfit fold {fit} -> score fold {ev}: thresholds {best}   F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)
    # test-like gain of this rule on the scored fold: per group gain weighted by test shares
    t_s1 = np.where(np.isin(tc, ["US", "India"]))[0]; g_tl = 0
    for k in range(CAP + 1):
        ek = e_ev[gv[e_ev] == k]
        if len(ek): g_tl += (gt[t_s1] == k).mean() * (f(v.q.values >= thr_row, ek) - f(base_acc, ek))
    print(f"   test-like gain of the rule: {g_tl:+.5f}", flush=True)
print(f"\ncross-fitted mean gain on validation: {np.mean([b - a for a, b, _ in tot.values()]):+.5f}", flush=True)
