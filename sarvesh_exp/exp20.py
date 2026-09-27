"""E28: fuzzy name channel for name-only records the pipeline misses. Query = a validation record (folds 8-9) with an
empty normalised address that is not accepted anywhere (q < 0.70 or no row). Pool = all train S1s of the same country.
Similarity = cosine of char 3-gram TF-IDF on the core name (rare trigrams only, for speed). Rule: accept the top S1
if sim >= tau and sim - second >= delta (row moved / added with q = 1). Cross-fitted fold 8 <-> 9."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from sklearn.feature_extraction.text import TfidfVectorizer
from common import load
from harness import truth_arrays, wscore
from match import normed
W_ = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
o2 = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
v = pd.read_parquet(f"{W_}/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"]).astype(
    {"sid": np.int64, "q": np.float64, "y": bool, "wt": np.float64})
W = v.wt.values[(ts[v.rid.values] < 0)].mean()
acc_r = np.zeros(len(n2), bool); acc_r[v.rid.values[v.q.values >= 0.70]] = True
Q = np.flatnonzero((n2.na.values == "") & (rf >= 8) & ~acc_r & (n2.cn.values != ""))
print(f"queries (name-only, validation, not accepted): {len(Q):,}; true matches among them {(ts[Q] >= 0).mean():.3f}", flush=True)
res = []
for c in ("US", "India"):
    P = np.flatnonzero(s1.country.values == c); q = Q[o2.country.values[Q] == c]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, max_df=0.002, dtype=np.float32)
    S = vec.fit_transform(n1.cn.values[P]); Z = vec.transform(n2.cn.values[q])
    ST = S.T.tocsr()
    for a in range(0, len(q), 2000):
        M = (Z[a:a + 2000] @ ST).tocsr()
        for i in range(M.shape[0]):
            lo, hi = M.indptr[i], M.indptr[i + 1]
            if hi == lo: continue
            dat, idx = M.data[lo:hi], M.indices[lo:hi]
            j = np.argpartition(-dat, min(1, len(dat) - 1))[:2]; j = j[np.argsort(-dat[j])]
            res.append((q[a + i], P[idx[j[0]]], dat[j[0]], dat[j[1]] if len(j) > 1 else 0.0))
    print(f"{c}: {len(q):,} queries done", flush=True)
r = pd.DataFrame(res, columns=["rid", "sid", "top", "second"])
r = r[s1f[r.sid.values] >= 8]
r["y"] = ts[r.rid.values] == r.sid.values
print(f"top-1 correct {r.y.mean():.3f} of {len(r):,}; by top-sim band:\n" +
      r.groupby(pd.cut(r.top, [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])).y.agg(["mean", "size"]).to_string(), flush=True)
def apply(sel):
    d = v.copy(); pos = pd.Series(np.arange(len(d)), index=d.rid.values).reindex(sel.rid.values).values
    inv = ~np.isnan(pos); ix = pos[inv].astype(int)
    sv, qv, yv = d.sid.values.copy(), d.q.values.copy(), d.y.values.copy()
    sv[ix] = sel.sid.values[inv]; qv[ix] = 1.0; yv[ix] = sel.y.values[inv]; d["sid"], d["q"], d["y"] = sv, qv, yv
    rr = sel.rid.values[~inv]
    new = pd.DataFrame({"rid": rr, "sid": sel.sid.values[~inv], "q": 1.0, "y": sel.y.values[~inv], "wt": np.where(ts[rr] < 0, W, 1.0)})
    return pd.concat([d, new], ignore_index=True)
base = lambda e: wscore(v.sid.values, v.q.values >= 0.70, v.y.values, v.wt.values, T, e)
grid = [(t, dl) for t in (0.5, 0.6, 0.7, 0.8, 0.9) for dl in (0.0, 0.05, 0.1, 0.2)]
out = {}
for t, dl in grid:
    sel = r[(r.top >= t) & (r.top - r.second >= dl)]; d = apply(sel)
    Fd = lambda e: wscore(d.sid.values, d.q.values >= 0.70, d.y.values, d.wt.values, T, e)
    out[(t, dl)] = {f: Fd(ents[s1f[ents] == f]) - base(ents[s1f[ents] == f]) for f in (8, 9)}
    print(f"tau {t} delta {dl}: pairs {len(sel):,} precision {sel.y.mean():.3f}  gain fold8 {out[(t, dl)][8]:+.6f} fold9 {out[(t, dl)][9]:+.6f}", flush=True)
g = []
for fit, ev in ((8, 9), (9, 8)):
    b = max(out, key=lambda k: out[k][fit]); g.append(out[b][ev]); print(f"fit {fit}: best {b} -> fold {ev} {out[b][ev]:+.6f}", flush=True)
print(f"E28 cross-fitted mean gain: {np.mean(g):+.6f}", flush=True)
