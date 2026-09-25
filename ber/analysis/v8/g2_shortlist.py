"""Blocking shortlist: how many of each record's top-20 retrieved S1 candidates must go to the matcher?
Recall (true S1 kept, matched records) vs candidates per record, on train records of folds 4-9 (the bi-encoder
never saw them), for:
  top-k, Sarvesh's gap:d (score >= best - d), gap:d with a cap, and a learned shortlist: a small LightGBM on
  retrieval-only features (record side AND S1 side), trained on records of folds 4-6, evaluated on 7-9.
Also prints candidates per record on test, per country, so France's near-ties are visible.
  python g2_shortlist.py      -> gen/shortlist.json, gen/shortlist_gbm.txt"""
import os, json, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, load
from harness import truth_arrays

OUT = os.environ.get("GEN_DIR", f"{WORK}/gen")
os.makedirs(OUT, exist_ok=True)
FEATS = ["score", "rank", "g_top1", "g_next", "top1", "top12", "n_close", "s1_rank", "s1_gap", "s1_claims",
         "s1_n"]


def feats(split):
    k = pd.read_parquet(f"{WORK}/cand_{split}.parquet")
    rid, sid = k.rid.values, k.sid.values
    sc, rk = k.score.values.astype(np.float32), k["rank"].values.astype(np.int16)
    del k
    nrec = rid.max() + 1
    top1 = np.full(nrec, np.nan, np.float32); top1[rid[rk == 0]] = sc[rk == 0]
    top2 = np.full(nrec, np.nan, np.float32); top2[rid[rk == 1]] = sc[rk == 1]
    # next candidate's score within the record (rows are contiguous and in rank order)
    nxt = np.r_[sc[1:], -1.0].astype(np.float32)
    nxt[np.r_[rid[1:] != rid[:-1], True]] = -1.0
    g1 = top1[rid] - sc
    n_close = np.bincount(rid, weights=(g1 <= 0.05), minlength=nrec)[rid]
    # S1 side: this record's rank among all candidate rows that point at the same S1, and the gap to that S1's best
    o = np.lexsort((-sc, sid))
    ss = sid[o]
    first = np.r_[True, ss[1:] != ss[:-1]]
    st = np.flatnonzero(first); gid = np.cumsum(first) - 1
    s1_rank = np.empty(len(o), np.float32); s1_rank[o] = np.arange(len(o)) - st[gid]
    s1_best = np.empty(len(o), np.float32); s1_best[o] = sc[o][st][gid]
    s1_n = np.bincount(sid, minlength=sid.max() + 1)[sid].astype(np.float32)
    s1_claims = np.bincount(sid[rk == 0], minlength=sid.max() + 1)[sid].astype(np.float32)
    F = pd.DataFrame({"score": sc, "rank": rk.astype(np.float32), "g_top1": g1, "g_next": sc - nxt,
                      "top1": top1[rid], "top12": (top1 - top2)[rid], "n_close": n_close.astype(np.float32),
                      "s1_rank": s1_rank, "s1_gap": s1_best - sc, "s1_claims": s1_claims, "s1_n": s1_n})
    return rid, sid, F


def curve(keep, rid, hit, recs, matched):
    """recall of the true S1 over matched records in `recs`, and kept candidates per record."""
    inr = recs[rid]
    kept = np.bincount(rid[keep & inr], minlength=len(recs))
    found = np.bincount(rid[keep & hit & inr], minlength=len(recs)) > 0
    m = recs & matched
    return float(found[m].mean()), float(kept[recs].mean())


def main():
    s1, other, ts, s1f, rf = truth_arrays()
    rid, sid, F = feats("train")
    hit = ts[rid] == sid
    matched = ts >= 0
    nrec = len(ts)
    ev = np.zeros(nrec, bool); ev[np.flatnonzero(rf >= 7)] = True        # evaluation records: folds 7-9
    res = {}
    print(f"rows {len(rid)}; eval records {ev.sum()}, matched {(ev & matched).sum()}; "
          f"true S1 anywhere in top-20: {curve(np.ones(len(rid), bool), rid, hit, ev, matched)[0]:.5f}", flush=True)
    rk, g1 = F["rank"].values, F.g_top1.values
    for k in (1, 2, 3, 4, 5, 7, 10, 20):
        res[f"top{k}"] = curve(rk < k, rid, hit, ev, matched)
    for d in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2):
        res[f"gap{d}"] = curve(g1 <= d, rid, hit, ev, matched)
        res[f"gap{d}_cap10"] = curve((g1 <= d) & (rk < 10), rid, hit, ev, matched)

    # learned shortlist: train on records of folds 4-6 (subsampled), score everything
    tr = np.isin(rf[rid], [4, 5, 6])
    rng = np.random.default_rng(0)
    tri = np.flatnonzero(tr)
    tri = tri[rng.random(len(tri)) < 0.35]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9,
                  num_threads=16, verbose=-1, seed=0)
    va = np.flatnonzero(np.isin(rf[rid], [7]))[:3_000_000]
    m = lgb.train(params, lgb.Dataset(F.iloc[tri], hit[tri]), 600,
                  valid_sets=[lgb.Dataset(F.iloc[va], hit[va])], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    m.save_model(f"{OUT}/shortlist_gbm.txt")
    print(pd.Series(m.feature_importance("gain"), index=FEATS).sort_values(ascending=False).round(0).to_string(), flush=True)
    p = m.predict(F, num_threads=16)
    for tau in (0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005):
        res[f"learned{tau}"] = curve((rk == 0) | (p >= tau), rid, hit, ev, matched)
    print(f"\n{'strategy':>18}  recall   cands/record   (train records of folds 7-9)", flush=True)
    for k, (r, n) in res.items():
        print(f"{k:>18}  {r:.5f}  {n:6.2f}", flush=True)

    # test: candidates per record by country, for the main strategies
    rid_t, sid_t, Ft = feats("test")
    ctry = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values
    pt = m.predict(Ft, num_threads=16)
    rk_t, g_t = Ft["rank"].values, Ft.g_top1.values
    strat = {"top5": rk_t < 5, "gap0.1": g_t <= 0.1, "gap0.1_cap10": (g_t <= 0.1) & (rk_t < 10)}
    for tau in (0.01, 0.005, 0.002):
        strat[f"learned{tau}"] = (rk_t == 0) | (pt >= tau)
    tst = {}
    for name, keep in strat.items():
        kept = np.bincount(rid_t[keep], minlength=len(ctry))
        has = np.bincount(rid_t, minlength=len(ctry)) > 0
        tst[name] = {c: float(kept[has & (ctry == c)].mean()) for c in sorted(set(ctry))}
        tst[name]["pairs_M"] = float(keep.sum() / 1e6)
        print(f"test {name:>16}: " + "  ".join(f"{c} {v:.2f}" for c, v in tst[name].items()), flush=True)
    json.dump({"train_folds7_9": res, "test_per_country": tst}, open(f"{OUT}/shortlist.json", "w"), indent=1)


if __name__ == "__main__":
    main()
