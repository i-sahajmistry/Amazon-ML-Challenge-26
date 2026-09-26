"""Which name words do train distractors add to the S1 they imitate (top-1 retrieved S1), against true matches?
Per country: words ranked by how often distractors add them, with the rate at which true matches add them.
  python x_decoywords.py"""
from collections import Counter
import numpy as np, pandas as pd
from common import WORK
from harness import truth_arrays
from match import normed

s1, other, ts, s1f, rf = truth_arrays()
n1 = normed("train", 1).cn.values
n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True).cn.values
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
top = c[c["rank"].values == 0]
for ctry in ("US", "India"):
    t = top[s1.country.values[top.sid.values] == ctry]
    dr = t[ts[t.rid.values] < 0]; tr = t[ts[t.rid.values] == t.sid.values]
    add = lambda rows: Counter(w for s, r in zip(rows.sid.values, rows.rid.values) for w in set(n2[r].split()) - set(n1[s].split()))
    ad, at = add(dr), add(tr)
    print(f"== {ctry}: distractors {len(dr)}, true top-1 {len(tr)}; words distractors add (share of distractors / of true)")
    print("  " + ", ".join(f"{w} {v / len(dr):.3f}/{at[w] / len(tr):.4f}" for w, v in ad.most_common(40)), flush=True)
