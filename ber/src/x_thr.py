"""Test-time per-country view of a stage-2 model: saves q for every record's argmax and sweeps thresholds, printing
per-country no-match rate and matches per S1 (train prior: 5.6% / 3.46; v5 predicts 5.75% / 3.43 on US/India).
Also counts accepted France records whose name adds a suspected distractor word.
  python x_thr.py [TAG]"""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
os.environ["TAG"] = TAG
if "llr" in TAG:
    os.environ["LLR"] = "1"
if "base" in TAG:
    os.environ["CE_TAGS"] = ",_base"
if "all" in TAG:
    os.environ["CE_TAGS"] = ",_base,_raw"
from common import WORK, load
from match import normed
import x_stage_multi as X
k = pd.read_parquet(f"{X.XD}/p5_test{TAG}.parquet")
d = X.context(X.rows("test", k))
d["q"] = lgb.Booster(model_file=f"{X.XD}/gbm5s2{TAG}.txt").predict(X.design(d), num_threads=32)
s1 = load("test", 1)
d["c"] = s1.country.values[d.sid.values]
d[["rid", "sid", "q", "c"]].to_parquet(f"{X.XD}/test_q{TAG}.parquet")
nS1 = s1.country.value_counts()
for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    a = d[d.q >= t]
    has = np.bincount(a.sid.values, minlength=len(s1)) > 0
    row = []
    for c in ("US", "India", "France"):
        m = s1.country.values == c
        row.append(f"{c} {1 - has[m].mean():.4f}/{(a.c == c).sum() / nS1[c]:.3f}")
    print(f"thr {t:.2f}  " + "  ".join(row), flush=True)
n1, n2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
f = d[(d.c == "France") & (d.q >= 0.65)]
added = [set(b.split()) - set(a.split()) for a, b in zip(n1.nn.values[f.sid.values], n2.nn.values[f.rid.values])]
for w in ("groupe", "holding", "france", "developpement", "participations", "distribution", "international",
          "services", "associes", "fils", "cie", "compagnie", "societe", "et", "centre", "sas", "sarl"):
    m = np.array([w in s for s in added])
    print(f"France accepted, name adds '{w}': {m.sum():6d}  (q mean {f.q.values[m].mean() if m.any() else 0:.3f})", flush=True)
