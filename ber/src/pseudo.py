"""Pseudo-labels for a country without training labels (self-training): pairs the current stage 1 is sure about become
extra stage-1 training rows (x_stage_multi.py PSEUDO=...), so the judges learn how that country writes names and
addresses, and separate its unsure records better.
  SPLIT=train  the held-out country of a leave-one-country-out run (records with fold HELD; needs HOLDOUT)
  SPLIT=test   every test country absent from the training data (the same rule for any such country)
  python pseudo.py   -> x/pseudo{TAG}_{SPLIT}.parquet  (rid, sid, y)
Rule, per record, on the mean of stage-1 models A and B: best candidate p >= P_HI and runner-up <= P2_MAX -> the best
pair is positive and the record's other candidates negative (a record has at most one owner); best p <= P_LO -> all
its candidates negative. Other records get no pseudo-label. On a held-out train country it also reports how accurate
the pseudo-labels are against the true labels (never used for training)."""
import os

import numpy as np
import pandas as pd

from common import HELD, WORK, load

XD = f"{WORK}/x"
TAG = os.environ.get("TAG", "")
SPLIT = os.environ.get("SPLIT", "train")
P_HI = float(os.environ.get("P_HI", 0.97))
P_LO = float(os.environ.get("P_LO", 0.03))
P2_MAX = float(os.environ.get("P2_MAX", 0.05))

f = f"{XD}/oof5_train{TAG}.parquet" if SPLIT == "train" else f"{XD}/p5_test{TAG}.parquet"
k = pd.read_parquet(f, columns=["rid", "sid", "p"])
other = pd.concat([load(SPLIT, 2), load(SPLIT, 3)], ignore_index=True)
if SPLIT == "train":
    from harness import truth_arrays
    _, _, ts, _, rf = truth_arrays()
    k = k[rf[k.rid.values] == HELD]
    assert len(k), "no held-out records: run with HOLDOUT=<country>"
else:
    unseen = sorted(set(other.country.values) - set(load("train", 1).country.values))
    k = k[np.isin(other.country.values[k.rid.values], unseen)]
    print("countries without training labels:", unseen, flush=True)
k = k.sort_values(["rid", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
g = k.groupby("rid").cumcount().values
first = g == 0
rid1, p1 = k.rid.values[first], k.p.values[first]
p2 = pd.Series(k.p.values[g == 1], index=k.rid.values[g == 1]).reindex(rid1).fillna(0.0).values
pos_rec, neg_rec = rid1[(p1 >= P_HI) & (p2 <= P2_MAX)], rid1[p1 <= P_LO]
n_rec = int(k.rid.max()) + 1
is_pos, is_neg = np.zeros(n_rec, bool), np.zeros(n_rec, bool)
is_pos[pos_rec], is_neg[neg_rec] = True, True
r = k.rid.values
lab = is_pos[r] | is_neg[r]
P = pd.DataFrame({"rid": r[lab], "sid": k.sid.values[lab], "y": (is_pos[r] & first)[lab]})
out = f"{XD}/pseudo{TAG}_{SPLIT}.parquet"
P.to_parquet(out)
print(f"{len(rid1):,} records: {len(pos_rec):,} sure matches (p >= {P_HI}, runner-up <= {P2_MAX}), {len(neg_rec):,} sure "
      f"non-matches (p <= {P_LO}), {len(rid1) - len(pos_rec) - len(neg_rec):,} left unlabelled; {len(P):,} pseudo-labelled "
      f"pairs ({P.y.mean():.1%} positive) -> {out}", flush=True)
if SPLIT == "train":   # how good are the pseudo-labels? (true labels only for this report)
    tru = ts[P.rid.values] == P.sid.values
    yv = P.y.values
    print(f"pseudo-label accuracy: positives {tru[yv].mean():.4f} correct, negatives {(~tru[~yv]).mean():.4f} correct; "
          f"true pairs of the country covered as positives {yv[tru].sum() / max((ts[np.unique(r)] >= 0).sum(), 1):.1%}",
          flush=True)
