"""restore_vempty: for the countries without training labels, an S1 the final lists leave empty takes its best claimant
(highest main-stack q) when the main stack + judge accepted it (q >= 0.70) and a self-training veto removed it, the
judge says yes, and it is not 1-13 numbers above the S1's (the decoy slot). An empty S1 filled right gains a whole
F0.5 point, so the fill pays from ~54% right. US / India validation (the same veto chain): 83% right, +0.00020.
  python x_vempty.py COMB MAIN LLM RESTORES REJECTS   e.g. _fin _v10plw _v10p restore_dd,...,restore_alias reject_dd,reject_decword
      -> x/restore_vempty.parquet"""
import glob, sys, numpy as np, pandas as pd
from common import WORK, load, unlabelled
from match import normed
from x_anatomy import first
XD = f"{WORK}/x"; T = 0.70; SHIFT = (1, 13)
comb, main, llm, restores, rejects = sys.argv[1:6]
d = pd.read_parquet(f"{XD}/test_q{comb}.parquet")
key = pd.MultiIndex.from_arrays([d.rid.values, d.sid.values])
rj = np.zeros(len(d), bool)
for f in restores.split(","):
    r = pd.read_parquet(f"{XD}/{f}.parquet"); d.loc[key.isin(pd.MultiIndex.from_frame(r[["rid", "sid"]])), "q"] = 1.0
for f in rejects.split(","):
    k = key.isin(pd.MultiIndex.from_frame(pd.read_parquet(f"{XD}/{f}.parquet")[["rid", "sid"]])); d.loc[k, "q"] = 0.0; rj |= k
acc = d.q.values >= T
ct1 = load("test", 1).country.values
has = np.bincount(d.sid.values[acc], minlength=len(ct1)) > 0
b = pd.read_parquet(f"{XD}/test_q{main}.parquet", columns=["rid", "sid", "q"])
qm = pd.Series(b.q.values, index=pd.MultiIndex.from_frame(b[["rid", "sid"]])).reindex(key).values
L = pd.read_parquet(f"{XD}/llm_test{llm}.parquet", columns=["rid", "sid", "llm"]); L = L[np.isfinite(L.llm.values)]
m = pd.Series(L.llm.values, index=pd.MultiIndex.from_frame(L[["rid", "sid"]])).reindex(key).values
cand = d.c.isin(unlabelled()).values & ~acc & (qm >= T) & ~has[d.sid.values] & ~rj
x = d[cand].assign(qm=qm[cand], llm=m[cand]).sort_values(["qm", "llm"], ascending=False).drop_duplicates("sid")
t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
a = np.array([first(v) for v in t1.num.values[x.sid.values]]); n = np.array([first(v) for v in t2.num.values[x.rid.values]])
slot = (a >= 0) & (n >= 0) & (n - a >= SHIFT[0]) & (n - a <= SHIFT[1])
sel = (x.llm.values > 0) & ~slot
x[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_vempty.parquet")
print("restore_vempty", int(sel.sum()), x[sel].c.value_counts().to_dict(), f"(empty S1s with a vetoed claimant {len(x)})", flush=True)
