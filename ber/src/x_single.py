"""Entities with exactly ONE accepted record, per country (v5f decisions): France excess would be distractors landing
on singletons. Top added name / address tokens for France single-record entities vs multi-record ones.
  python x_single.py"""
import numpy as np, pandas as pd
from collections import Counter
from common import load
from match import normed
s1 = load("test", 1).reset_index(drop=True)
m = pd.read_csv("/home/scai/mtech/aib262144/scratch/AmazonMLChallenge/output_v5f/matching_results.tsv", sep="\t",
                dtype=str, keep_default_na=False)
k = m.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0).values
c = s1.set_index("entity_id").country.loc[m.source1_entity_id].values
for ctry in ("US", "India", "France"):
    kk = k[c == ctry]
    print(ctry, {n: round((kk == n).mean(), 4) for n in range(0, 7)}, flush=True)
other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
rid_of = pd.Series(np.arange(len(other)), index=other.entity_id)
sid_of = pd.Series(np.arange(len(s1)), index=s1.entity_id)
n1, n2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
fr = (c == "France")
for lab, sel in (("single", fr & (k == 1)), ("multi", fr & (k >= 2))):
    sub = m[sel]
    pairs = sub.assign(r=sub.matched_entity_ids.str.split(",")).explode("r")
    a, b = sid_of.loc[pairs.source1_entity_id].values, rid_of.loc[pairs.r].values
    for col in ("nn", "na"):
        cnt = Counter(t for x, y in zip(n1[col].values[a], n2[col].values[b]) for t in set(y.split()) - set(x.split()) if not t.isdigit())
        print(f"France {lab} ({len(pairs)} records) added {col}: " + ", ".join(f"{t} {v / len(pairs):.3f}" for t, v in cnt.most_common(25)), flush=True)
