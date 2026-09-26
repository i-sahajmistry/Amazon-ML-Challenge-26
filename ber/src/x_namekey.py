"""v9 blocking study: hybrid dense + exact-key blocking. Besides each record's HNSW top-20, take the S1s of the same
country whose normalised core name equals the record's (a hash lookup, so it scales), when at most CAP S1s share
that name. How many true S1s missing from the top-20 does it recover, and how many pairs does it add?
Train records of S1 folds 4-9.
  python x_namekey.py"""
import numpy as np, pandas as pd
from common import WORK
from match import normed
from harness import truth_arrays

s1, other, ts, s1f, rf = truth_arrays()
n1 = normed("train", 1)
n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid"])
recs = np.flatnonzero(rf >= 4)
real = recs[ts[recs] >= 0]
key1 = pd.Series(n1.cn.values + "|" + s1.country.values)
key2 = n2.cn.values + "|" + other.country.values
size = key1.map(key1.value_counts()).values                       # S1s sharing each S1's key
by_key = pd.DataFrame({"key": key1.values, "sid": np.arange(len(s1)), "size": size})
NS = len(s1)
cand = np.unique(c.rid.values.astype(np.int64) * NS + c.sid.values)      # (record, S1) pair keys, sorted
true = real.astype(np.int64) * NS + ts[real]
hit20 = np.isin(true, cand, assume_unique=True)
print(f"real records folds 4-9: {len(real)}; true S1 in HNSW top-20: {hit20.mean():.4%}", flush=True)
q = pd.DataFrame({"rid": recs, "key": key2[recs]})
q = q[n2.cn.values[recs] != ""]
for cap in (1, 2, 3, 5, 10):
    m = q.merge(by_key[by_key["size"] <= cap], on="key")
    pk = np.unique(m.rid.values.astype(np.int64) * NS + m.sid.values)
    add = pk[~np.isin(pk, cand, assume_unique=True)]
    got = np.isin(true, add, assume_unique=True)
    print(f"cap {cap:2d}: +{len(add):8d} pairs ({len(add) / len(recs):.3f} per record)  recovers {got.sum()} true S1s "
          f"-> top-20 + key {(hit20 | got).mean():.4%}", flush=True)
