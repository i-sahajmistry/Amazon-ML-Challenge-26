"""E29: does file order or the id format carry cluster information? (train, labels)"""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load
from harness import truth_arrays
s1, _, ts, s1f, rf = truth_arrays()
o2 = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
n2 = len(load("train", 2))
print("id examples S1:", s1.entity_id.values[:3].tolist(), " S2:", o2.entity_id.values[:3].tolist(), " S3:", o2.entity_id.values[n2:n2 + 3].tolist(), flush=True)
for name, sl in (("source 2", slice(0, n2)), ("source 3", slice(n2, len(o2)))):
    t = ts[sl]; same = (t[1:] == t[:-1]) & (t[1:] >= 0)
    print(f"{name}: consecutive rows with the same true S1 {same.mean():.5f} (random ~{1 / len(s1):.1e})", flush=True)
    m = np.flatnonzero(t >= 0); c = np.corrcoef(m, t[m])[0, 1]
    print(f"{name}: corr(row index, true S1 row index) {c:.4f}", flush=True)
g = pd.read_parquet if False else None
gt = load("train", "ground_truth")
print("ground truth columns:", list(gt.columns), "rows", len(gt), flush=True)
pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
print("ground-truth row order vs S1 file order corr:", np.corrcoef(np.arange(len(gt)), pos.reindex(gt.source1_entity_id).values)[0, 1], flush=True)
# matched pairs: do ids share a prefix / numeric closeness?
m = np.flatnonzero(ts >= 0)[:200000]
a = s1.entity_id.values[ts[m]]; b = o2.entity_id.values[m]
pre = np.mean([x[:4] == y[:4] for x, y in zip(a, b)])
rnd = np.mean([x[:4] == y[:4] for x, y in zip(a, np.random.default_rng(0).permutation(b))])
print(f"matched pairs sharing the first 4 id chars {pre:.4f} vs shuffled {rnd:.4f}", flush=True)
t2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True); t1 = load("test", 1)
print("test ids S1:", t1.entity_id.values[:3].tolist(), " S2/3:", t2.entity_id.values[:3].tolist(), flush=True)
