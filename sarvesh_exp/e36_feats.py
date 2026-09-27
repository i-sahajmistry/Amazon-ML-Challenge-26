"""E36 features, per feats2 row (rid, sid), appended to the extra features: how many S1s (same country) share the
record's core name (amb_rec) / the S1's core name (amb_s1), record address empty (na_empty), record name invented
(no token in any S1 name; invented), record core name == S1 core name (cn_eq). Label-free, per split."""
import sys, numpy as np, pandas as pd
from common import load, WORK
from match import normed
C = sys.argv[1]   # control sandbox with the original extra_{split}.parquet
for split in ("train", "test"):
    s1 = load(split, 1); n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c2 = pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values
    k1 = n1.cn.values + "|" + s1.country.values; cnt = pd.Series(k1).value_counts()
    kr = n2.cn.values + "|" + c2
    amb_rec = np.where(n2.cn.values == "", -1, cnt.reindex(kr).fillna(0).values).astype(np.float32)
    amb_s1 = np.where(n1.cn.values == "", -1, cnt.reindex(k1).values).astype(np.float32)
    vocab = set(" ".join(n1.cn.values).split())
    inv = np.array([bool(x) and all(t not in vocab for t in x.split()) for x in n2.cn.values])
    F = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"]); r, s = F.rid.values, F.sid.values
    x = pd.read_parquet(f"{C}/work/x/extra_{split}.parquet")
    assert len(x) == len(F)
    x["z_amb_rec"] = amb_rec[r]; x["z_amb_s1"] = amb_s1[s]; x["z_na_empty"] = (n2.na.values == "")[r].astype(np.float32)
    x["z_invented"] = inv[r].astype(np.float32); x["z_cn_eq"] = (n2.cn.values[r] == n1.cn.values[s]).astype(np.float32)
    x.to_parquet(f"{WORK}/x/extra_{split}.parquet")
    print(split, len(x), x[["z_amb_rec", "z_amb_s1", "z_na_empty", "z_invented", "z_cn_eq"]].mean().round(4).to_dict(), flush=True)
