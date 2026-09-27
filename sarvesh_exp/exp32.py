"""E43: France-threshold probe lists on top of E16fr3 (US / India untouched). France q = the variant's chain
(frchain) with restore_fr_dd = 1 / reject_fr_dd = 0. T65: add France pairs with 0.65 <= q < 0.70 (best S1 row, not in
reject_fr_dd); T75: remove France pairs with 0.70 <= q < 0.75 that E16fr3 accepted (restores have q = 1, so they stay);
T60 / T80 as the bolder versions."""
import sys, numpy as np, pandas as pd
from common import load, WORK
OUT = sys.argv[1]; X = f"{WORK}/x"
t1 = load("test", 1); o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
rj = pd.read_parquet(f"{X}/reject_fr_dd.parquet"); rjk = set(zip(rj.rid.values, rj.sid.values))
notrj = np.array([(a, b) not in rjk for a, b in zip(fr.rid.values, fr.sid.values)])
ids = lambda m: pd.DataFrame({"s1_id": t1.entity_id.values[fr.sid.values[m]], "rec_id": o.entity_id.values[fr.rid.values[m]]})
q = fr.q.values
for name, lo, hi in (("T65", 0.65, 0.70), ("T60", 0.60, 0.70)):
    m = (q >= lo) & (q < hi) & notrj; ids(m).to_parquet(f"{OUT}/add_{name}.parquet"); print(name, "add", m.sum(), flush=True)
for name, lo, hi in (("T75", 0.70, 0.75), ("T80", 0.70, 0.80)):
    m = (q >= lo) & (q < hi); ids(m).to_parquet(f"{OUT}/rem_{name}.parquet"); print(name, "remove", m.sum(), flush=True)
pd.DataFrame(columns=["s1_id", "rec_id"]).to_parquet(f"{OUT}/empty.parquet")
