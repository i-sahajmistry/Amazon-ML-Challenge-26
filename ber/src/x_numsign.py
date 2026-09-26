"""Does the house-number shift have a direction? Train records whose first number differs from their top-1 retrieved
S1's: share of positive shifts (record > S1) and shift histogram in -13..13, for distractors vs true matches.
  python x_numsign.py"""
import numpy as np, pandas as pd
from common import WORK
from harness import truth_arrays
from match import normed

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
c = c[c["rank"].values == 0]
first = lambda num: np.array([int(x.split()[0][:15]) if x else -1 for x in num])
kr = first(pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True).num.values)[c.rid.values]
ks = first(normed("train", 1).num.values)[c.sid.values]
ok = (kr >= 0) & (ks >= 0) & (kr != ks)
d = (kr - ks)[ok]
true = (ts[c.rid.values] == c.sid.values)[ok]
ctry = s1.country.values[c.sid.values][ok]
for name, m in (("true match", true), ("distractor", ~true)):
    for cc in ("US", "India"):
        k = m & (ctry == cc)
        near = k & (np.abs(d) <= 13)
        h = np.bincount(d[near] + 13, minlength=27) / max(near.sum(), 1)
        print(f"{name:10s} {cc:5s}: differ {k.sum():8d}, |shift|<=13 {near.sum():8d}, positive share {np.mean(d[near] > 0):.3f}; "
              f"-13..-1 {h[:13].sum():.3f}  +1..+13 {h[14:].sum():.3f}", flush=True)
