"""US / India validation study (labels) for stack choices, and E7 (mean of two stacks) as a test file.
Validation rows: x/val_q{TAG}.parquet (rid, sid, p, y, wt, q) from x_nocopy.py, entity folds 8-9, distractors weighted."""
import os, glob, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
O = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"
V = "/scratch/scai/mtech/aib262144/amlc_variant/work/x"
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]
cty = s1.country.values
avail = sorted(os.path.basename(f) for f in glob.glob(f"{O}/val_q_v10*.parquet") + glob.glob(f"{V}/val_q_*.parquet"))
print("validation files:", avail, flush=True)
def load(tag):
    for d in (O, V):
        f = f"{d}/val_q{tag}.parquet"
        if os.path.exists(f):
            return pd.read_parquet(f)
def score(v, thr, c=None):
    e = ents if c is None else ents[cty[ents] == c]
    return wscore(v.sid.values, v.q.values >= thr, v.y.values.astype(bool), v.wt.values, T, e)
THRS = [0.6, 0.65, 0.7, 0.75, 0.8]
res = {}
for tag in ["_v10pw", "_v10bpw", "_v10plw", "_v10bplw"]:
    v = load(tag)
    if v is None:
        print(f"{tag}: no validation file", flush=True); continue
    res[tag] = v
    print(f"{tag:10s} " + "  ".join(f"{t}:{score(v, t):.5f}" for t in THRS) + f"   US {score(v,0.7,'US'):.5f} India {score(v,0.7,'India'):.5f}", flush=True)
def mean2(a, b):
    m = a.merge(b[["rid", "sid", "q"]], on="rid", how="left", suffixes=("", "_b"))
    same = m.sid.values == m.sid_b.values
    q = np.where(same, (m.q.values + m.q_b.values) / 2, m.q.values)   # different S1: keep the first stack's row
    return m.assign(q=q)[a.columns], same.mean()
pairs = [("_v10pw", "_v10bpw"), ("_v10plw", "_v10bplw")]
for a, b in pairs:
    if a in res and b in res:
        m, sh = mean2(res[a], res[b])
        print(f"mean{a}+{b} (same S1 {sh:.4f}) " + "  ".join(f"{t}:{score(m, t):.5f}" for t in THRS) +
              f"   US {score(m,0.7,'US'):.5f} India {score(m,0.7,'India'):.5f}", flush=True)
# E7 test file input: mean of the LLM-blended stacks (bge and main) for every country; France is replaced later by the variant
ta, tb = pd.read_parquet(f"{O}/test_q_v10bplw.parquet"), pd.read_parquet(f"{V}/test_q_v10plw.parquet")
m = ta.merge(tb[["rid", "sid", "q"]], on="rid", how="left", suffixes=("", "_b"))
same = m.sid.values == m.sid_b.values
m["q"] = np.where(same, (m.q.values + m.q_b.values) / 2, m.q.values)
m[ta.columns].to_parquet(f"{os.environ['AMLC_ROOT']}/work/x/test_q_mean7.parquet")
print(f"wrote test_q_mean7 (same S1 {same.mean():.4f})", flush=True)
