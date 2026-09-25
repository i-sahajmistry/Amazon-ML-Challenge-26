"""How resolvable are empty-address records? For real ones (folds 4-9): how many S1 entities in the country share the
true S1's core name, does the record's core name equal it, where does the true S1 rank.  python x_empty.py"""
import numpy as np, pandas as pd
from common import WORK
from match import normed
from harness import truth_arrays
s1, other, ts, s1f, rf = truth_arrays()
n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
empty = n2.na.values == ""
real = ts >= 0
print(f"empty-address share: real records {empty[real].mean():.2%}, distractors {empty[~real].mean():.2%}")
key = pd.Series(n1.cn.values + "|" + s1.country.values)
same = key.map(key.value_counts()).values                       # S1s sharing this S1's core name (incl. itself)
r = np.flatnonzero(real & empty & (rf >= 4))
amb = same[ts[r]]
print(f"real empty-address records (folds 4-9): {len(r)}  S1s sharing the true core name: "
      + str(pd.Series(np.minimum(amb, 5)).value_counts(normalize=True).sort_index().round(3).to_dict()) + "  (5 = 5+)")
eq = n2.cn.values[r] == n1.cn.values[ts[r]]
print(f"  record core name == true S1 core name: {eq.mean():.1%};  unique AND equal: {(eq & (amb == 1)).mean():.1%}")
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
hit = c[ts[c.rid.values] == c.sid.values]
rank = np.full(len(other), 99, np.int16); rank[hit.rid.values] = hit["rank"].values
rr = rank[r]
print("  true S1 rank: " + str({"0": round((rr == 0).mean(), 3), "1-4": round(((rr >= 1) & (rr < 5)).mean(), 3),
                                "5-19": round(((rr >= 5) & (rr < 20)).mean(), 3), "20+": round((rr >= 20).mean(), 3)}))
for lab, m in [("unique core name", amb == 1), ("shared core name", amb > 1)]:
    print(f"  {lab}: {m.mean():.1%} of them; true S1 in top-5 {(rr[m] < 5).mean():.1%}, rank 0 {(rr[m] == 0).mean():.1%}")
