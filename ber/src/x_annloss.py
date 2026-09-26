"""What the blocking change cost on validation (entity folds 8-9): real records whose true S1 exact search finds (v7's
exact top-20, work/archive_v7/cand_train.parquet, same embeddings) but the current candidates lack, split by where
they were lost (not in the HNSW top-20 / dropped by the shortlist) and by their exact rank. For each group, the
validation F0.5 if those records were matched right (upper bound), and that gain scaled by how often the matcher
gets records of the same exact rank right when it has their S1 (expected).
  python x_annloss.py [QTAG]      (reads x/val_q{QTAG}w.parquet, default _v9p)"""
import sys, numpy as np, pandas as pd
from common import WORK
from harness import truth_arrays, wscore

QT = sys.argv[1] if len(sys.argv) > 1 else "_v9p"
THR = 0.70
s1, other, ts, s1f, rf = truth_arrays()
NS = len(s1)
T = np.bincount(ts[ts >= 0], minlength=NS)
ents = np.where(s1f >= 8)[0]
v = pd.read_parquet(f"{WORK}/x/val_q{QT}w.parquet")
base = wscore(v.sid.values, v.q.values >= THR, v.y.values, v.wt.values, T, ents)
R = np.where((ts >= 0) & (rf >= 8))[0]                       # real records owned by a fold 8-9 S1
tk = R.astype(np.int64) * NS + ts[R]


def pairs(f, cols=("rid", "sid")):
    c = pd.read_parquet(f, columns=list(cols))
    return c[np.isin(c.rid.values, R)]


ex = pairs(f"{WORK}/archive_v7/cand_train.parquet", ("rid", "sid", "rank"))
exk = ex.rid.values.astype(np.int64) * NS + ex.sid.values
hn = pairs(f"{WORK}/cand_train.parquet")
cur = pairs(f"{WORK}/feats2_train.parquet")
in_ex = np.isin(tk, exk)
in_hn = np.isin(tk, hn.rid.values.astype(np.int64) * NS + hn.sid.values)
in_cur = np.isin(tk, cur.rid.values.astype(np.int64) * NS + cur.sid.values)
rank = pd.Series(ex["rank"].values, index=exk).reindex(tk).values           # exact rank of the true S1 (NaN if > 20)
print(f"val real records {len(R)}; true S1 in exact top-20 {in_ex.mean():.4%}, HNSW top-20 {in_hn.mean():.4%}, "
      f"current candidates {in_cur.mean():.4%}; base F0.5 {base:.5f}", flush=True)
print(f"HNSW finds, exact top-20 misses: {(in_hn & ~in_ex).sum()}", flush=True)

row = pd.Series(np.arange(len(v)), index=v.rid.values)
acc = v.q.values >= THR
ok = acc & v.y.values                                          # matched right
got = pd.Series(ok, index=v.rid.values).reindex(R).fillna(False).values.astype(bool)


def fixed(m):
    """F0.5 with the records R[m] matched to their true S1."""
    sid, a, y, w = v.sid.values.copy(), acc.copy(), v.y.values.copy(), v.wt.values.copy()
    r = R[m]
    have = row.reindex(r).values
    h = ~np.isnan(have)
    i = have[h].astype(int)
    sid[i], a[i], y[i] = ts[r[h]], True, True
    n = (~h).sum()
    return wscore(np.r_[sid, ts[r[~h]]], np.r_[a, np.ones(n, bool)], np.r_[y, np.ones(n, bool)], np.r_[w, np.ones(n)],
                  T, ents)


buckets = (("exact rank 0", rank == 0), ("exact rank 1-4", (rank >= 1) & (rank <= 4)), ("exact rank 5-19", rank >= 5))
tot_up = tot_exp = 0.0
for where, lost in (("not in HNSW top-20", in_ex & ~in_hn), ("dropped by the shortlist", in_ex & in_hn & ~in_cur)):
    for name, b in buckets:
        m = lost & b
        if not m.any():
            continue
        rate = got[in_cur & b].mean()                          # how often the matcher gets such records right
        up = fixed(m) - base
        tot_up += up; tot_exp += up * rate
        print(f"{where:25s} {name:16s} records {m.sum():6d} ({m.mean():.4%})  upper +{up:.5f}  "
              f"matcher right on such records {rate:.3f} -> expected +{up * rate:.5f}", flush=True)
print(f"total: upper +{tot_up:.5f}, expected +{tot_exp:.5f}", flush=True)
# unreachable-node check: are HNSW's rank-0 misses whole S1s (every record of that S1 misses it)?
m = in_ex & ~in_hn & (rank == 0)
lost_s = ts[R[m]]
u, cnt = np.unique(lost_s, return_counts=True)
own = np.bincount(ts[R], minlength=NS)[u]                      # real val records of those S1s
print(f"rank-0 misses: {m.sum()} records over {len(u)} S1s; S1s losing all their records {np.mean(cnt == own):.3f}, "
      f"records per such S1 {cnt.mean():.2f}", flush=True)
