"""Do distractors come in groups (a sibling business with several noisy records) that share one deviation from the
S1 entity they resemble? Prints group-size distributions and examples.  python x_dgroups.py"""
import numpy as np, pandas as pd
from common import WORK
from harness import truth_arrays

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
top = np.full(len(other), -1, np.int64)
t0 = c[c["rank"] == 0]
top[t0.rid.values] = t0.sid.values
dis = ts < 0
src3 = other.entity_id.str.startswith("S3").values
print(f"records {len(other)}  distractors {dis.sum()} ({dis.mean():.1%})  S3 share: distractors {src3[dis].mean():.3f}, owned {src3[~dis].mean():.3f}")
g = pd.Series(top[dis]).value_counts()
print("distractors per S1 they land on (S1s with >=1):", g.value_counts().sort_index().head(10).to_dict())
print("share of S1 entities hit by >=1 distractor:", round(len(g) / len(s1), 4))
print("true records per S1:", pd.Series(ts[~dis]).value_counts().value_counts().sort_index().to_dict())
T = np.bincount(ts[~dis], minlength=len(s1))
hit = np.zeros(len(s1), bool); hit[g.index.values] = True
print("mean true records: S1 hit by a distractor", T[hit].mean().round(3), " not hit", T[~hit].mean().round(3))
# pairs of distractors on the same S1: same source or one S2 + one S3?
multi = g[g >= 2].index.values
dm = pd.DataFrame({"sid": top[dis], "s3": src3[dis]})
dm = dm[np.isin(dm.sid.values, multi)]
print("S1s with 2+ distractors:", len(multi), " their distractors S3 share", dm.s3.mean().round(3),
      " groups mixing S2+S3", dm.groupby("sid").s3.agg(lambda x: 0 < x.mean() < 1).mean().round(3))
txt = lambda df, i: f"{df.business_name.values[i]} | {df.business_address.values[i]}"
rng = np.random.default_rng(1)
for sid in rng.choice(multi, 8, replace=False):
    print(f"\nS1 : {txt(s1, sid)}")
    for r in np.flatnonzero((top == sid) & dis):
        print(f"  D{'3' if src3[r] else '2'}: {txt(other, r)}")
    for r in np.flatnonzero(ts == sid):
        print(f"  T{'3' if src3[r] else '2'}: {txt(other, r)}")
