"""Blocking misses: real records whose true S1 is not in their top-5 candidates. What are they (empty address,
unrelated name) and would ranks 6-20 or an exact-address lookup find the true S1?  python x_block.py"""
import numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from common import WORK
from match import normed
from harness import truth_arrays

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
real = np.flatnonzero((ts >= 0) & (rf >= 4))
hit = c[ts[c.rid.values] == c.sid.values]                     # rank of the true S1 when retrieved (top-20)
rank = np.full(len(other), 99, np.int16); rank[hit.rid.values] = hit["rank"].values
r = real[rank[real] >= 5]
print(f"real records folds 4-9: {len(real)}  true S1 outside top-5: {len(r)} ({len(r) / len(real):.2%})  "
      f"of which in ranks 5-19: {(rank[r] < 20).mean():.1%}")
n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
a_n, b_n = n1.nn.values[ts[r]], n2.nn.values[r]
a_a, b_a = n1.na.values[ts[r]], n2.na.values[r]
name_sim = process.cpdist(list(a_n), list(b_n), scorer=fuzz.token_set_ratio, workers=-1)
addr_sim = process.cpdist(list(a_a), list(b_a), scorer=fuzz.token_set_ratio, workers=-1)
empty = b_a == ""
print(f"  empty address {empty.mean():.1%} | name token-set<50 (unrelated name) {(name_sim < 50).mean():.1%} "
      f"| addr token-set>=90 given non-empty {(addr_sim[~empty] >= 90).mean():.1%}")
# exact-address channel: same country + identical normalized address string
key1 = pd.Series(np.arange(len(s1)), index=n1.na.values + "|" + s1.country.values)
key1 = key1[~key1.index.duplicated(keep=False)]               # unique addresses only
k2 = n2.na.values[r] + "|" + other.country.values[r]
found = key1.reindex(k2).values
ok = ~np.isnan(found) & (found == ts[r])
print(f"  exact normalized-address lookup finds the true S1 for {ok.mean():.1%} of the misses; "
      f"among unrelated-name misses {ok[name_sim < 50].mean():.1%}")
rng = np.random.default_rng(0)
for i in rng.choice(len(r), 12, replace=False):
    print(f"rank {rank[r[i]]:2d} | S1: {s1.business_name.values[ts[r[i]]]} | {s1.business_address.values[ts[r[i]]]}\n"
          f"          rec: {other.business_name.values[r[i]]} | {other.business_address.values[r[i]]}")
