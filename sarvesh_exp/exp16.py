"""E24: where US / India validation loses its true matches (folds 8-9 entities, bge stack _v10bpw).
Each true (record, S1) pair falls in one stage: not retrieved (FAISS top-20) / retrieved but cut by the shortlist /
shortlisted but another S1 is the record's best (stage 1+2) / best S1 correct but q < 0.70 / accepted."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays
from match import normed
W = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"
s1, _, ts, s1f, rf = truth_arrays()
ev = s1f >= 8
r = np.flatnonzero((ts >= 0) & ev[np.maximum(ts, 0)]); s = ts[r]; N = len(r)
print(f"true pairs in validation folds 8-9: {N:,}", flush=True)
BIG = np.int64(len(s1) + 1); key = r.astype(np.int64) * BIG + s
def has(path):
    c = pd.read_parquet(path, columns=["rid", "sid"]); k = c.rid.values.astype(np.int64) * BIG + c.sid.values
    return np.isin(key, k)
in_cand = has(f"{W}/cand_train.parquet"); in_short = has(f"{W}/feats2_train.parquet")
v = pd.read_parquet(f"{W}/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q"]).set_index("rid")
best_sid = v.sid.reindex(r).values; best_q = v.q.reindex(r).values
cat = np.select([~in_cand, ~in_short, np.isnan(best_q), best_sid != s, best_q < 0.5, best_q < 0.70],
                ["1 not retrieved (top-20)", "2 cut by shortlist", "3 not in stage-2 rows", "4 another S1 ranked best",
                 "5 correct S1, q < 0.5", "6 correct S1, 0.5 <= q < 0.70"], "7 accepted")
t = pd.Series(cat).value_counts().sort_index()
print((t.to_frame("pairs").assign(share=(t / N).round(5))).to_string(), flush=True)
low = (cat == "5 correct S1, q < 0.5")
print("q of correct-S1 misses below 0.5:", np.histogram(best_q[low], bins=[0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])[0].tolist(), flush=True)
# is the S1 of a missed pair otherwise matched? (a lost member of a found cluster vs a whole lost cluster)
acc = v[v.q >= 0.70]; n_acc = np.bincount(acc.sid.values, minlength=len(s1))
miss = cat != "7 accepted"
print(f"missed pairs whose S1 has other accepted records: {(n_acc[s[miss]] > 0).mean():.3f}", flush=True)
# what do the missed pairs look like (examples, normalised)
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
rng = np.random.default_rng(0)
for c in ("1 not retrieved (top-20)", "4 another S1 ranked best", "5 correct S1, q < 0.5"):
    idx = np.flatnonzero(cat == c)
    if len(idx) == 0: continue
    print(f"--- examples: {c}", flush=True)
    for i in rng.choice(idx, min(8, len(idx)), replace=False):
        print(f"  S1  [{n1.nn.values[s[i]]}] [{n1.na.values[s[i]]}]\n  rec [{n2.nn.values[r[i]]}] [{n2.na.values[r[i]]}]  q={best_q[i]:.3f}", flush=True)
