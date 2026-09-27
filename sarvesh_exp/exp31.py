"""E42: is France's strictness on number-mismatched records caused by the veto chain? On US / India validation (labels),
score the France chain = min(main stack v10, self-trained a1, a2, a3) (same S1 required) vs the main stack alone, per
number bucket: true rate, acceptance, precision, and how many TRUE matches the chain rejects that the main stack
accepts. Also overall F0.5 of main vs chain (does the veto chain help or hurt on labelled data)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
from match import normed
V = "/scratch/scai/mtech/aib262144/amlc_variant/work/x"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10pw.parquet", columns=["rid", "sid", "q", "y", "wt"])
qc = v.q.values.copy()
for t in ("a1", "a2", "a3"):
    e = pd.read_parquet(f"{V}/val_q_{t}pw.parquet", columns=["rid", "sid", "q"])
    m = v[["rid", "sid"]].merge(e, on="rid", how="left", suffixes=("", "_b"))
    qt = np.where(m.sid.values == m.sid_b.values, m.q.values, 0.0)
    print(f"{t}: best S1 differs from main for {(m.sid.values != m.sid_b.values).mean():.4f} of rows", flush=True)
    qc = np.minimum(qc, qt)
sid, y, wt = v.sid.values, v.y.values.astype(bool), v.wt.values
am, ac = v.q.values >= 0.70, qc >= 0.70
print(f"US / India validation F0.5: main {wscore(sid, am, y, wt, T, ents):.5f}  chain(min of 4) {wscore(sid, ac, y, wt, T, ents):.5f}", flush=True)
def ints(a):
    out = np.full(len(a), -1, np.int64)
    for i, x in enumerate(a):
        t = x.split()[0] if x else ""
        if t.isdigit() and len(t) < 12: out[i] = int(t)
    return out
r = v.rid.values; a, b = ints(n2.num.values[r]), ints(n1.num.values[sid]); d = a - b
sn = n2.cn.values[r] == n1.cn.values[sid]
pe = pd.DataFrame({"s": sid, "a": a}).groupby(["s", "a"]).s.transform("size").values - 1
kind = np.select([(a >= 0) & (b >= 0) & (d < 0), (a >= 0) & (b >= 0) & (d >= 1) & (d <= 13), (a >= 0) & (b >= 0) & (d > 13), a == b],
                 ["down", "up1-13", "up>13", "same-num"], "no-num")
rows = []
for k in ("down", "up1-13", "up>13", "same-num", "no-num"):
    for nm in (True, False):
        for alone in (True, False):
            m = (kind == k) & (sn == nm) & ((pe == 0) == alone)
            if m.sum() == 0: continue
            rows.append(dict(kind=k, name="same" if nm else "edit", alone=alone, rows=m.sum(),
                             true_w=y[m].sum() / np.where(y[m], 1.0, wt[m]).sum(), acc_main=am[m].mean(), acc_chain=ac[m].mean(),
                             true_lost_by_chain=(m & y & am & ~ac).sum(), false_removed_by_chain=(m & ~y & am & ~ac).sum()))
print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
