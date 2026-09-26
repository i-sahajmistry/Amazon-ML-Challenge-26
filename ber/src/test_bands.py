"""Label-free: stage-2 scores on test by country, in the bands loco_eval.py prints for trained vs held-out countries.
If France's bands look like the held-out country's in the leave-one-country-out run (and unlike US/India on test),
France behaves like an unseen train country, and the LOCO error breakdown is a fair picture of France's loss.
  TAG=_v8w python test_bands.py        (x/test_q{TAG}.parquet from x_nocopy.py test)"""
import os

import numpy as np
import pandas as pd

from common import WORK, load

TAG = os.environ.get("TAG", "_v8w")
THR = float(os.environ.get("THR", 0.70))
BANDS = ((0, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 0.95), (0.95, 1.01))
pd.set_option("display.width", 220)

d = pd.read_parquet(f"{WORK}/x/test_q{TAG}.parquet")
s1 = load("test", 1)
q, c = d.q.values, d.c.values
rows = []
for g in sorted(set(c)):
    k = c == g
    n1 = (s1.country.values == g).sum()
    acc_s1 = np.bincount(d.sid.values[k & (q >= THR)], minlength=len(s1))[s1.country.values == g]
    rows.append({"country": g, "records": k.sum(), "S1": n1, f"accepted @ {THR}": (q[k] >= THR).mean(),
                 **{f"q [{a},{b})": ((q[k] >= a) & (q[k] < b)).mean() for a, b in BANDS},
                 "accepted per S1": acc_s1.mean(), "empty S1": (acc_s1 == 0).mean()})
print(f"Test stage-2 scores by country ({TAG}):")
print(pd.DataFrame(rows).round(4).to_string(index=False))
