"""Is the fold split biased? Per S1 fold (crc32 % 10): S1s, US share, singletons, matched records per S1, distractors
(by their own hash fold), empty addresses. Then the entity view: each distractor's imitated S1 (its top-1 retrieved S1)
- how many distractors per S1 exist in train, how many a validation S1 (folds 8-9) keeps in stage-2 rows (hash fold
4-9), and how many test S1s have (records below q 0.5 in v9p), with the share of distractors that have another
distractor of the same S1 at the same house number (what PEERS reads).
  python x_folds.py"""
import numpy as np, pandas as pd
from common import WORK, load
from harness import truth_arrays
from match import normed

s1, other, ts, s1f, rf = truth_arrays()
NS = len(s1)
T = np.bincount(ts[ts >= 0], minlength=NS)
dis = ts < 0
emp = other.business_address.str.strip().eq("").values
print("fold  S1s     US    single  matched/S1  distractors  empty-addr(matched)", flush=True)
for f in range(10):
    m = s1f == f
    own = (ts >= 0) & (rf == f)
    print(f"{f}  {m.sum():7d}  {(s1.country.values[m] == 'US').mean():.3f}  {(T[m] == 0).mean():.4f}  {T[m].mean():.3f}"
          f"       {(dis & (rf == f)).sum():8d}     {emp[own].mean():.4f}", flush=True)

c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
top = c[c["rank"].values == 0]
im = np.full(len(other), -1); im[top.rid.values] = top.sid.values        # imitated S1 = top-1 retrieved S1
d = np.where(dis & (im >= 0))[0]
print(f"\ndistractors {dis.sum()}, with a retrieved S1 {len(d)}; hash fold = imitated S1's fold for "
      f"{(rf[d] == s1f[im[d]]).mean():.3f} of them", flush=True)
num = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True).num.values
key = np.array([int(x.split()[0][:15]) if x else -1 for x in num[d]])


def peers(rows):
    """share of these distractors with another distractor of the same imitated S1 at the same first number."""
    g = pd.DataFrame({"s": im[rows], "k": key[np.searchsorted(d, rows)]})
    g = g[g.k >= 0]
    return (g.groupby(["s", "k"]).s.transform("size") > 1).mean()


val = s1f[im[d]] >= 8
kept = val & (rf[d] >= 4)
n_val = (s1f >= 8).sum()
print(f"distractors per S1, train all S1s: {len(d) / NS:.3f}; validation S1s, all: {val.sum() / n_val:.3f}, kept in "
      f"stage-2 rows (hash fold 4-9): {kept.sum() / n_val:.3f}", flush=True)
print(f"with a same-number peer: train all {peers(d):.3f}, validation kept {peers(d[kept]):.3f}", flush=True)

q = pd.read_parquet(f"{WORK}/x/test_q_v9pw.parquet")
t1 = load("test", 1)
tnum = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True).num.values
for ctry in ("US", "India", "France"):
    g = q[(q.c == ctry) & (q.q < 0.5)]
    k = np.array([int(x.split()[0][:15]) if x else -1 for x in tnum[g.rid.values]])
    h = pd.DataFrame({"s": g.sid.values, "k": k})
    h = h[h.k >= 0]
    pr = (h.groupby(["s", "k"]).s.transform("size") > 1).mean()
    print(f"test {ctry:6s}: records below q 0.5 per S1 {len(g) / (t1.country == ctry).sum():.3f}, with a same-number "
          f"peer {pr:.3f}", flush=True)
