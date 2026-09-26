"""Label-free look at where a country's test decisions are unsure: q histogram per country, and sample S1 groups
(the S1 plus every record whose best candidate it is, with q) from the ambiguous band, to compare France with US / India.
  python x_frdiag.py [TAG] [N_GROUPS]"""
import sys, numpy as np, pandas as pd
from common import WORK, load

TAG = sys.argv[1] if len(sys.argv) > 1 else "_v8w"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
d = pd.read_parquet(f"{WORK}/x/test_q{TAG}.parquet")
s1 = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
print(d.columns.tolist(), s1.columns.tolist(), flush=True)
bins = [0, .02, .1, .3, .5, .7, .9, .98, 1.01]
for c, g in d.groupby("c"):
    h = np.histogram(g.q, bins)[0] / len(g)
    print(f"{c:8s} n {len(g):8d}  " + " ".join(f"{a:.2f}-{b:.2f}:{x:.3f}" for a, b, x in zip(bins, bins[1:], h)), flush=True)
txt = lambda df, i: f"{df.business_name.values[i]} | {df.business_address.values[i]}"
rng = np.random.default_rng(0)
for c in ("France", "US", "India"):
    g = d[d.c == c]
    amb = g[(g.q > .1) & (g.q < .9)].sid.unique()
    print(f"\n===== {c}: {len(amb)} S1s with a record in 0.1-0.9", flush=True)
    for sid in rng.choice(amb, min(N if c == "France" else N // 3, len(amb)), replace=False):
        print(f"S1  {txt(s1, sid)}")
        for r in g[g.sid == sid].sort_values("q", ascending=False).itertuples():
            print(f"  {r.q:.3f} {'ACC' if r.q >= .7 else '   '}  {txt(other, r.rid)}")
