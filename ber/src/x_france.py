"""France over-matching: after dropping records that add a known distractor word (groupe/france/developpement...),
which added name / address tokens are over-represented among France acceptances in the borderline band
(q 0.65-0.95) vs the confident band, and vs US?  python x_france.py [TAG]"""
import sys, numpy as np, pandas as pd
from collections import Counter
from common import load
from match import normed
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
d = pd.read_parquet(f"{__import__('common').WORK}/x/test_q{TAG}.parquet")
n1, n2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
DW = {"groupe", "france", "developpement", "participations", "distribution", "international", "holding"}
def added(col, a):
    return [set(y.split()) - set(x.split()) for x, y in zip(n1[col].values[a.sid.values], n2[col].values[a.rid.values])]
for ctry in ("France", "US"):
    a = d[(d.c == ctry) & (d.q >= 0.65)]
    an = added("nn", a)
    bad = np.array([bool(s & DW) for s in an])
    print(f"{ctry}: accepted {len(a)}, with a distractor word {bad.sum()}", flush=True)
    for col, sets in (("name", an), ("addr", added("na", a))):
        for lo, hi in ((0.65, 0.95), (0.95, 1.01)):
            m = (a.q.values >= lo) & (a.q.values < hi) & ~bad
            cnt = Counter(t for s, k in zip(sets, m) if k for t in s if not t.isdigit())
            print(f"  {col} q[{lo},{hi}) n={m.sum()}: " + ", ".join(f"{t} {v / max(m.sum(), 1):.3f}" for t, v in cnt.most_common(30)), flush=True)
