"""Mix two variants' stage-2 scores on the held-out country of a leave-one-country-out run (loco_eval.py writes
x/loco_q{TAG}.parquet: one row per record, its argmax S1 and q). Where both variants pick the same S1 the score is
(1-a)*q_A + a*q_B; where they pick different S1s the row of A is kept for a < 0.5, of B for a > 0.5, and the higher
score at a = 0.5. a = 0 is variant A alone, a = 1 variant B alone.
  HOLDOUT=India A=_v8 B=_v8dce python loco_blend.py"""
import os
import zlib

import numpy as np
import pandas as pd

from common import HELD, WORK
from harness import truth_arrays
from x_judge import wscore

A, B = os.environ["A"], os.environ["B"]
s1_, _, ts, s1f, _ = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1_))
h = np.where(s1f == HELD)[0]
crc = np.array([zlib.crc32(x.encode()) % 10 for x in s1_.entity_id.values[h]])
groups = {"all S1": h, "crc folds 8-9": h[crc >= 8]}
qa = pd.read_parquet(f"{WORK}/x/loco_q{A}.parquet").set_index("rid")
qb = pd.read_parquet(f"{WORK}/x/loco_q{B}.parquet").set_index("rid").reindex(qa.index)
assert qb.sid.notna().all(), "the variants cover different records"
same = qa.sid.values == qb.sid.values
print(f"{len(qa):,} records; the variants pick the same S1 for {same.mean():.4f}", flush=True)
THRS = np.round(np.arange(0.50, 0.91, 0.05), 2)
res = []
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    q_mix = (1 - a) * qa.q.values + a * qb.q.values
    take_b = ~same & ((a > 0.5) | ((a == 0.5) & (qb.q.values > qa.q.values)))
    sid = np.where(take_b, qb.sid.values, qa.sid.values).astype(np.int64)
    y = np.where(take_b, qb.y.values, qa.y.values).astype(bool)
    wt = np.where(take_b, qb.wt.values, qa.wt.values)
    q = np.where(same, q_mix, np.where(take_b, qb.q.values, qa.q.values))
    for g, ents in groups.items():
        curve = {t: wscore(sid, q >= t, y, wt, T, ents) for t in THRS}
        best = max(curve, key=curve.get)
        res.append({"a (weight of B)": a, "group": g, "F0.5@0.70": curve[0.7], "best": curve[best], "best thr": best})
        print(f"a={a:.2f} {g:14s} " + "  ".join(f"{t:.2f}:{v:.5f}" for t, v in curve.items()), flush=True)
print(f"\nSUMMARY  A={A}  B={B}\n" + pd.DataFrame(res).round(5).to_string(index=False), flush=True)
