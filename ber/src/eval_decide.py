import numpy as np, pandas as pd, time
from common import WORK
from harness import truth_arrays, build, score
from decide import select
s1, other, ts, s1f, rf = truth_arrays()
k = pd.read_parquet(f"{WORK}/pred_all_train.parquet")
sid, p, tru, T, ents = build(k, ts, s1f, rf)
print("rows", len(sid), "entities", len(ents), flush=True)
for thr in [0.4, 0.5, 0.6]:
    print(f"global thr {thr}: {score(sid, p >= thr, tru, T, ents):.5f}", flush=True)
for lam in [1.0, 1.4, 1.8, 2.4]:
    for extra in [0.0, 0.03]:
        t = time.time(); acc = select(sid, p, lam, extra)
        print(f"expected-F  lam {lam} extra {extra}: {score(sid, acc, tru, T, ents):.5f}  ({time.time()-t:.0f}s)", flush=True)
