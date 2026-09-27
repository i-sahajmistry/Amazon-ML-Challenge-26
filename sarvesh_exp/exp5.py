"""E11: does test's higher decoy density explain validation (0.9928) > leaderboard (~0.9915) for US / India?
Group S1s by how many records claim them (their best S1): label-free, so it exists on test too. Validation F0.5 per
group, then reweight the groups to test's mix -> a "test-like" validation estimate. Also: threshold per group."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
O = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]; cty = s1.country.values
v = pd.read_parquet(f"{O}/val_q_v10bpw.parquet")
from common import load
t = pd.read_parquet(f"{O}/test_q_v10bplw.parquet")
tc = load("test", 1).country.values
t = t[np.isin(tc[t.sid.values], ["US", "India"])]
CAP = 8
def groups(sid, n_s1, w=None):
    n = np.bincount(sid, weights=w, minlength=n_s1)
    return np.minimum(np.round(n).astype(int), CAP)
# claimants per S1 (rows whose best S1 it is); validation distractors count with their weight (test share)
gv = groups(v.sid.values, len(s1f), v.wt.values)
gt = groups(t.sid.values, len(tc))
acc = v.q.values >= 0.7
y = v.y.values.astype(bool)
print("claimants per S1 (validation weighted to the test distractor share) vs test, US + India", flush=True)
rows = []
for c in ["US", "India", None]:
    e_all = ents if c is None else ents[cty[ents] == c]
    t_s1 = np.where(np.isin(tc, ["US", "India"] if c is None else [c]))[0]
    f_all = wscore(v.sid.values, acc, y, v.wt.values, T, e_all)
    est = 0.0; parts = []
    for k in range(CAP + 1):
        e = e_all[gv[e_all] == k]
        share_v = len(e) / len(e_all); share_t = (gt[t_s1] == k).mean()
        f = wscore(v.sid.values, acc, y, v.wt.values, T, e) if len(e) else np.nan
        if len(e): est += share_t * f
        parts.append((k, share_v, share_t, f))
    name = c or "US+India"
    print(f"\n== {name}: validation {f_all:.5f}  test-like (reweighted to test's claimant mix) {est:.5f}  diff {est - f_all:+.5f}", flush=True)
    print(" k  val share  test share   val F0.5", flush=True)
    for k, a, b, f in parts:
        print(f"{k:>2}{'+' if k == CAP else ' '} {a:9.4f}  {b:9.4f}   {f:.5f}", flush=True)
# best threshold per claimant group (validation) and what it would give test-like
print("\n== best threshold per claimant group (US+India validation)", flush=True)
for k in range(CAP + 1):
    e = ents[gv[ents] == k]
    if len(e) < 1000: continue
    cur = {th: wscore(v.sid.values, v.q.values >= th, y, v.wt.values, T, e) for th in (0.5, 0.6, 0.7, 0.8, 0.9)}
    b = max(cur, key=cur.get)
    print(f"k={k}: n={len(e):,} " + " ".join(f"{th}:{x:.5f}" for th, x in cur.items()) + f"  best {b} (+{cur[b]-cur[0.7]:.5f})", flush=True)
