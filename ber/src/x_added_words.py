"""Most frequently ADDED name / address words (record vs its top-1 S1 candidate) per test country, to find the
French counterparts of the distractor vocabulary.  python x_added_words.py"""
import numpy as np, pandas as pd
from collections import Counter
from common import WORK, load
from match import normed
k = pd.read_parquet(f"{WORK}/feats2_test.parquet", columns=["rid", "sid", "rank"])
k = k[k["rank"].values == 0]
n1, n2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
c = load("test", 1).country.values[k.sid.values]
for col in ("nn", "na"):
    for ctry in ("France", "US", "India"):
        m = c == ctry
        A, B = n1[col].values[k.sid.values[m]], n2[col].values[k.rid.values[m]]
        cnt = Counter()
        for a, b in zip(A, B):
            cnt.update(t for t in set(b.split()) - set(a.split()) if not t.isdigit())
        tot = m.sum()
        print(f"{col} {ctry}: " + ", ".join(f"{t} {v / tot:.4f}" for t, v in cnt.most_common(45)), flush=True)
