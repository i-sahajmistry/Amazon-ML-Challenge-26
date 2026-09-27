"""E16 input: the bge + LLM stack's test q (_v10bplw) with the empty-S1 rescue (validated on US / India, t = 0.5):
for an S1 of a country with training labels whose claimants all have q < 0.70, the best claimant is accepted if its
q >= 0.5 (its q is lifted to 0.70). France rows are untouched (x_final replaces them with the variant's chain)."""
import os, numpy as np, pandas as pd
from common import load, unlabelled
O = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"
d = pd.read_parquet(f"{O}/test_q_v10bplw.parquet")
lab = ~np.isin(d.c.values, unlabelled())
g = pd.Series(d.q.values).groupby(d.sid.values)
top = g.transform("max").values
best = d.q.values == top
first = ~pd.Series(best & True).groupby(d.sid.values).cumsum().gt(1).values   # one row per S1 if ties
lift = lab & best & first & (top < 0.70) & (top >= 0.50)
print(f"rescued S1s: {lift.sum():,} (US {((d.c.values == 'US') & lift).sum():,}, India {((d.c.values == 'India') & lift).sum():,})", flush=True)
d.loc[lift, "q"] = 0.70
d.to_parquet(f"{os.environ['AMLC_ROOT']}/work/x/test_q_resc16.parquet")
