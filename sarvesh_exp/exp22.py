"""E30 / E31 test q: E17 (US / India) + France with the variant's chain [+ bge 4th veto] + restore / reject, then the
empty-S1 rescue for France too (an S1 with no record at q >= 0.70 takes its best claimant if q >= T_RESC)."""
import sys, numpy as np, pandas as pd
from common import WORK
OUT_TAG, VETO, TR = sys.argv[1], sys.argv[2] == "1", float(sys.argv[3])
X = f"{WORK}/x"
ui = pd.read_parquet(f"{X}/test_q_resc17.parquet"); ui = ui[ui.c != "France"]
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
if VETO:
    e = pd.read_parquet(f"{X}/test_q_v10bplwS.parquet"); e = e[e.c == "France"]
    m = fr[["rid", "sid"]].merge(e[["rid", "sid", "q"]], on="rid", how="left", suffixes=("", "_b"))
    fr["q"] = np.minimum(fr.q.values, np.where(m.sid.values == m.sid_b.values, m.q.values, 0.0))
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
q, sid = fr.q.values, fr.sid.values
top = pd.Series(q).groupby(sid).transform("max").values
best = (q == top) & ~pd.Series(q == top).groupby(sid).cumsum().gt(1).values
lift = best & (top < 0.70) & (top >= TR)
print(f"France: accepted {(q >= 0.70).sum():,}; rescued S1s {lift.sum():,} (t = {TR})", flush=True)
fr.loc[lift, "q"] = 0.70
pd.concat([ui, fr], ignore_index=True)[["rid", "sid", "q", "c"]].to_parquet(f"{X}/test_q{OUT_TAG}.parquet")
