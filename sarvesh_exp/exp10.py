"""E17 input: the empty-S1 rescue of E16 (t = 0.5, countries with labels) applied to the bge + LLM stack's test q
from the E10 run on the smaller candidate set (US / India shortlist P >= 0.005). France rows are untouched."""
import os, numpy as np, pandas as pd
from common import WORK, unlabelled
d = pd.read_parquet(f"{WORK}/x/test_q_v10bplw.parquet")
lab = ~np.isin(d.c.values, unlabelled())
top = pd.Series(d.q.values).groupby(d.sid.values).transform("max").values
best = d.q.values == top
first = ~pd.Series(best & True).groupby(d.sid.values).cumsum().gt(1).values   # one row per S1 if ties
lift = lab & best & first & (top < 0.70) & (top >= 0.50)
print(f"rescued S1s: {lift.sum():,} (US {((d.c.values == 'US') & lift).sum():,}, India {((d.c.values == 'India') & lift).sum():,})", flush=True)
d.loc[lift, "q"] = 0.70
d.to_parquet(f"{WORK}/x/test_q_resc17.parquet")
