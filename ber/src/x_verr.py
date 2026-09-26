"""Where does a model lose F0.5 on the copy-free validation (x/val_q{TAG}w.parquet, written by x_nocopy.py test)?
For each error type, the score if it were fixed (oracle): missed true matches, accepted distractors, records given to
the wrong S1, split by record buckets (empty address, namesake S1s, name edits, source). Also checks whether a
threshold per bucket would beat the single threshold.
  python x_verr.py [TAG] [THR]"""
import sys, numpy as np, pandas as pd
from common import WORK
from harness import truth_arrays, wscore

TAG = sys.argv[1] if len(sys.argv) > 1 else "_v8"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70
s1, other, ts, s1f, rf = truth_arrays()
v = pd.read_parquet(f"{WORK}/x/val_q{TAG}w.parquet")
f = pd.read_parquet(f"{WORK}/feats2_train.parquet", columns=["rid", "sid", "b_addr_empty", "a_cn_s1_count", "src3"])
e = pd.read_parquet(f"{WORK}/x/extra_train.parquet", columns=["ed_add", "ed_del"])
f = pd.concat([f, e], axis=1)
v = v.merge(f, on=["rid", "sid"], how="left")
T = np.bincount(ts[ts >= 0], minlength=len(s1))
ents = np.where(s1f >= 8)[0]
sid, y, wt, q = v.sid.values, v.y.values, v.wt.values, v.q.values
real = ts[v.rid.values] >= 0
acc = q >= THR
base = wscore(sid, acc, y, wt, T, ents)
print(f"{TAG} copy-free F0.5 at {THR}: {base:.5f}", flush=True)
kinds = {"missed true match": y & ~acc, "accepted distractor": ~y & ~real & acc, "accepted wrong S1": ~y & real & acc}
buckets = {"all": np.ones(len(v), bool), "empty address": v.b_addr_empty.values > 0,
           "namesake S1s (>1 same core name)": v.a_cn_s1_count.values > 1,
           "name adds a word": v.ed_add.values > 0, "name drops a word": v.ed_del.values > 0,
           "source 3": v.src3.values > 0,
           "S1 with a single claiming record": v.groupby("sid").rid.transform("size").values == 1}
print(f"{'error type':22s} {'bucket':34s} {'rows':>8s}  gain if fixed", flush=True)
for kn, km in kinds.items():
    for bn, bm in buckets.items():
        m = km & bm
        fixed = acc.copy(); fixed[m] = ~fixed[m]
        print(f"{kn:22s} {bn:34s} {m.sum():8d}  {wscore(sid, fixed, y, wt, T, ents) - base:+.5f}", flush=True)
print("threshold per bucket (the rest keep the single threshold):", flush=True)
for bn, bm in list(buckets.items())[1:]:
    best = max(((wscore(sid, np.where(bm, q >= t, acc), y, wt, T, ents), t) for t in np.arange(0.3, 0.96, 0.05)))
    print(f"  {bn:34s} best thr {best[1]:.2f}: {best[0] - base:+.5f}", flush=True)
