"""Learned blocking shortlist: which of a record's top-20 retrieved S1 candidates go on to the matcher.

A small LightGBM on retrieval-only features scores every retrieved pair; a record keeps its best candidate plus
every candidate with probability >= TAU. Features come from both sides of the pair: the record's own list (score,
rank, gap to its best and to its next candidate, how many candidates are within 0.05 of its best) and the S1's
(how many retrieved records point at it, how many rank it first, this record's rank and gap among them). A clear
winner keeps one pair; near-ties (chains, shared names, France's many close candidates) keep the tied group.
Fit on train records of S1 folds 4-6 (the bi-encoder never saw them); folds 7-9 report recall vs size.
On train folds 7-9: TAU 0.001 keeps the true S1 for 99.47% of matched records at 1.54 candidates per record
(top-5: 99.02% at 5; gap 0.1: 99.42% at 1.98; all 20: 99.50%).
  python shortlist.py            -> work/shortlist_gbm.txt   (used by match.py with SHORTLIST=learned:TAU)"""
import os, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK

FEATS = ["score", "rank", "g_top1", "g_next", "top1", "top12", "n_close", "s1_rank", "s1_gap", "s1_claims", "s1_n"]
MODEL = f"{WORK}/shortlist_gbm.txt"


def feats(rid, sid, sc, rk):
    """rows of a record contiguous and in rank order (retrieve.py). Returns the FEATS frame."""
    sc = sc.astype(np.float32); rk = rk.astype(np.int16)
    nrec = int(rid.max()) + 1
    top1 = np.full(nrec, np.nan, np.float32); top1[rid[rk == 0]] = sc[rk == 0]
    top2 = np.full(nrec, np.nan, np.float32); top2[rid[rk == 1]] = sc[rk == 1]
    nxt = np.r_[sc[1:], -1.0].astype(np.float32)
    nxt[np.r_[rid[1:] != rid[:-1], True]] = -1.0
    g1 = top1[rid] - sc
    n_close = np.bincount(rid, weights=(g1 <= 0.05), minlength=nrec)[rid]
    o = np.lexsort((-sc, sid)); ss = sid[o]
    first = np.r_[True, ss[1:] != ss[:-1]]
    st = np.flatnonzero(first); gid = np.cumsum(first) - 1
    s1_rank = np.empty(len(o), np.float32); s1_rank[o] = np.arange(len(o)) - st[gid]
    s1_best = np.empty(len(o), np.float32); s1_best[o] = sc[o][st][gid]
    nsid = int(sid.max()) + 1
    return pd.DataFrame({"score": sc, "rank": rk.astype(np.float32), "g_top1": g1, "g_next": sc - nxt,
                         "top1": top1[rid], "top12": (top1 - top2)[rid], "n_close": n_close.astype(np.float32),
                         "s1_rank": s1_rank, "s1_gap": s1_best - sc,
                         "s1_claims": np.bincount(sid[rk == 0], minlength=nsid)[sid].astype(np.float32),
                         "s1_n": np.bincount(sid, minlength=nsid)[sid].astype(np.float32)})


def keep(c, tau):
    """c: all retrieved candidates of a split (rid, sid, score, rank). Boolean mask of the shortlist."""
    p = lgb.Booster(model_file=MODEL).predict(feats(c.rid.values, c.sid.values, c.score.values, c["rank"].values),
                                              num_threads=int(os.environ.get("NT", 16)))
    return (c["rank"].values == 0) | (p >= tau)


def fit():
    from harness import truth_arrays
    _, _, ts, _, rf = truth_arrays()
    c = pd.read_parquet(f"{WORK}/cand_train.parquet")
    rid, sid = c.rid.values, c.sid.values
    F = feats(rid, sid, c.score.values, c["rank"].values)
    hit = ts[rid] == sid
    tr = np.flatnonzero(np.isin(rf[rid], [4, 5, 6]))
    tr = tr[np.random.default_rng(0).random(len(tr)) < 0.35]
    va = np.flatnonzero(rf[rid] == 7)[:3_000_000]
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9,
                       num_threads=int(os.environ.get("NT", 16)), verbose=-1, seed=0),
                  lgb.Dataset(F.iloc[tr], hit[tr]), 600, valid_sets=[lgb.Dataset(F.iloc[va], hit[va])],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    m.save_model(MODEL)
    p = m.predict(F, num_threads=int(os.environ.get("NT", 16)))
    ev = np.isin(rf[rid], [7, 8, 9])
    recs = np.isin(rf, [7, 8, 9]); m_ = recs & (ts >= 0)
    for tau in (0.01, 0.005, 0.002, 0.001, 0.0005):
        k = (c["rank"].values == 0) | (p >= tau)
        found = np.bincount(rid[k & hit & ev], minlength=len(ts)) > 0
        print(f"tau {tau}: true S1 kept {found[m_].mean():.5f}, candidates/record "
              f"{np.bincount(rid[k & ev], minlength=len(ts))[recs].mean():.2f}", flush=True)


if __name__ == "__main__":
    fit()
