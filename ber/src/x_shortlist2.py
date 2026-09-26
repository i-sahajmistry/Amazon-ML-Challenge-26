"""v9 blocking study: does the shortlist model keep the same true S1s with fewer candidates when it also sees cheap
text similarities of each retrieved pair (name / core name / address, and each pair's gap to the record's best)?
Both models fit on train records of S1 folds 4-6 and are compared on folds 7-9 at equal recall. Costs one rapidfuzz
comparison per retrieved pair (linear in records x K), so it scales like the retrieval itself.
  python x_shortlist2.py     (needs the normalisation caches built by match.features)"""
import numpy as np, pandas as pd, lightgbm as lgb
from common import WORK
from harness import truth_arrays
from match import retrieval_feats, text_feats, SHORT_F, TXT, normed

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet")
c = retrieval_feats(c[rf[c.rid.values] >= 4].sort_values(["rid", "rank"]).reset_index(drop=True))
n1 = normed("train", 1)
n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
c = text_feats(c, n1, n2)
r = c.rid.values
y = ts[r] == c.sid.values
tr, ev = np.isin(rf[r], [4, 5, 6]) & (r % 4 == 0), np.isin(rf[r], [7, 8, 9])
real = np.unique(r[ev & (ts[r] >= 0)])
n_rec, n_s1 = len(np.unique(r[ev])), (s1f >= 7).sum()
prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, verbose=-1, num_threads=32)
E = c[ev]
for label, F in (("retrieval only", SHORT_F), ("retrieval + text", SHORT_F + TXT)):
    m = lgb.train(prm, lgb.Dataset(c.loc[tr, F], y[tr]), 300)
    p = m.predict(E[F], num_threads=32)
    for tau in (0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005):
        k = p >= tau
        hit = np.isin(real, E.rid.values[k & y[ev]]).mean()
        print(f"{label:17s} tau {tau:<7} true S1 kept {hit:.4%}   per record {k.sum() / n_rec:5.3f}   per S1 {k.sum() / n_s1:6.2f}",
              flush=True)
    if F is not SHORT_F:
        m.save_model(f"{WORK}/shortlist_text.txt")
