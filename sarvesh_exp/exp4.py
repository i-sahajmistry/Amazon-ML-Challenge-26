"""E8: per-country threshold curves on US / India validation for the bge stack (labels)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]; cty = s1.country.values
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet")
THRS = np.round(np.arange(0.50, 0.91, 0.025), 3)
for c in ["US", "India"]:
    e = ents[cty[ents] == c]
    cur = {t: wscore(v.sid.values, v.q.values >= t, v.y.values.astype(bool), v.wt.values, T, e) for t in THRS}
    b = max(cur, key=cur.get)
    print(f"{c}: best {b} {cur[b]:.5f} vs 0.70 {cur[0.7]:.5f} (+{cur[b]-cur[0.7]:.5f})   " + " ".join(f"{t}:{x:.5f}" for t, x in cur.items()), flush=True)
