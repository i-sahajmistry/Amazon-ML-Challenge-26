"""Stage-2 decision judge: LightGBM (as x_nocopy.py) vs XGBoost (Apache-2.0) vs their average, on the same rows.
Rows: each record's stage-1 argmax S1 with entity context; every distractor appears once, weighted (W) to the 39%
test share, in the loss and in the validation precision. Fit on entity folds 4-7, compare on folds 8-9.
  python x_judge.py                      -> F0.5 per threshold for lgb / xgb / avg, x/judge{TAG}.json
  JUDGE=lgb|xgb|avg|best python x_judge.py test
      refit the judge on folds 4-9 (1.15 x its best iterations), score test once with stage-1 model A's and once
      with model B's probabilities (stage 2 learned on single-model probabilities) and average
      -> x/test_q{TAG}J.parquet + x/judge{TAG}J.json (judge, thr) for x_final.py (THR=auto)"""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from common import load
from harness import truth_arrays
from stage2 import context
import x_stage_multi as X

XGB = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", grow_policy="lossguide",
           max_leaves=63, max_depth=0, eta=0.05, min_child_weight=20, subsample=0.7, colsample_bytree=0.9,
           max_bin=256, nthread=X.NT, seed=0)
LGB = {**X.P2, "num_threads": X.NT}
THRS = np.round(np.arange(0.5, 0.91, 0.02), 2)
CFG = f"{X.XD}/judge{X.TAG}.json"


def wscore(sid, acc, tru, wt, T, ents):
    """macro F0.5 over entities ents; a distractor row counts wt times among the predictions."""
    n = len(T)
    tp = np.bincount(sid[acc & tru], minlength=n)[ents]
    fp = np.bincount(sid[acc & ~tru], weights=wt[acc & ~tru], minlength=n)[ents]
    t = T[ents]
    pr = np.divide(tp, tp + fp, out=np.zeros(len(t)), where=tp + fp > 0)
    rc = tp / np.maximum(t, 1)
    f = np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros(len(t)), where=tp > 0)
    return np.where(t == 0, (tp + fp == 0).astype(float), f).mean()


def record_inputs(k, pv, sel_i):
    """stage-2 record inputs under stage-1 probabilities pv (row order of k) for the rows sel_i of k:
    p of that row, p2 = best OTHER candidate of the record, rec_n20 = the record's candidates with pv >= 0.2."""
    rid = k.rid.values
    order = np.lexsort((-pv, rid))
    r_s, p_s = rid[order], pv[order]
    first = np.r_[True, r_s[1:] != r_s[:-1]]
    g = np.cumsum(first) - 1
    top1 = p_s[first]
    has2 = np.r_[~first[1:], False] & first
    top2 = np.zeros(len(top1), np.float32)
    top2[g[has2]] = p_s[np.flatnonzero(has2) + 1]
    arg1 = order[first]
    n20 = np.bincount(g, weights=(p_s >= 0.2), minlength=len(top1))
    gi = pd.Series(np.arange(len(top1)), index=r_s[first]).loc[rid[sel_i]].values
    p2 = np.where(arg1[gi] == sel_i, top2[gi], top1[gi])
    return pv[sel_i].astype(np.float32), p2.astype(np.float32), n20[gi].astype(np.float32)


def copy_free_rows():
    """stage-2 training rows without distractor copies; wt = 1 for real records, W for distractors."""
    s1_, _, ts, s1f, rf = truth_arrays()
    b = X.rows("train", pd.read_parquet(f"{X.XD}/oof5_train{X.TAG}.parquet"))
    matched = ts >= 0
    r, s = b.rid.values, b.sid.values
    keep = (rf[r] >= 4) & (s1f[s] >= 4)            # records and entities the cross-encoders never saw
    pool = np.where(~matched & (rf >= 4))[0]
    W = int(X.SHARE / (1 - X.SHARE) * matched.sum()) / len(pool)
    d = b[keep].assign(y=lambda x: ts[x.rid.values] == x.sid.values, fold=lambda x: s1f[x.sid.values])
    d = context(d.reset_index(drop=True))
    d["wt"] = np.where(matched[d.rid.values], 1.0, W)
    print(f"rows {len(d)}, distractor weight W {W:.2f}", flush=True)
    return d, np.bincount(ts[matched], minlength=len(s1_)), np.where(s1f >= 8)[0]


def fit_lgb(Xt, yt, wt, Xv=None, yv=None, wv=None, rounds=4000):
    valid = [] if Xv is None else [lgb.Dataset(Xv, yv, weight=wv)]
    cb = [lgb.log_evaluation(500)] + ([lgb.early_stopping(50)] if valid else [])
    m = lgb.train(LGB, lgb.Dataset(Xt, yt, weight=wt), rounds, valid_sets=valid, callbacks=cb)
    return (lambda Z: m.predict(Z, num_threads=X.NT)), (m.best_iteration or rounds)


def finite(Z):
    """XGBoost rejects inf (e.g. the cross-encoder gap of a record with one candidate); LightGBM treats it as the
    largest value, so map +-inf to +-1e6 to keep the same ordering."""
    return Z.replace([np.inf, -np.inf], [1e6, -1e6])


def fit_xgb(Xt, yt, wt, Xv=None, yv=None, wv=None, rounds=4000):
    dt = xgb.QuantileDMatrix(finite(Xt), yt, weight=wt, max_bin=XGB["max_bin"])
    ev = [] if Xv is None else [(xgb.QuantileDMatrix(finite(Xv), yv, weight=wv, ref=dt), "valid")]
    m = xgb.train(XGB, dt, rounds, evals=ev, early_stopping_rounds=50 if ev else None, verbose_eval=500)
    it = (m.best_iteration + 1) if ev else rounds
    return (lambda Z: m.predict(xgb.DMatrix(finite(Z)), iteration_range=(0, it))), it


FIT = {"lgb": fit_lgb, "xgb": fit_xgb}


def compare():
    d, T, ents = copy_free_rows()
    D, y, w = X.design(d), d.y.values, d.wt.values
    tr, va = d.fold.values <= 7, d.fold.values >= 8
    sid = d.sid.values[va]
    q, iters = {}, {}
    for name, fit in FIT.items():
        pred, iters[name] = fit(D[tr], y[tr], w[tr], D[va], y[va], w[va])
        q[name] = pred(D[va])
        print(f"{name}: best iteration {iters[name]}", flush=True)
    q["avg"] = (q["lgb"] + q["xgb"]) / 2
    res = {j: {float(t): wscore(sid, q[j] >= t, y[va], w[va], T, ents) for t in THRS} for j in q}
    print("thr    " + "  ".join(f"{j:>8}" for j in res), flush=True)
    for t in THRS:
        print(f"{t:.2f}  " + "  ".join(f"{res[j][float(t)]:.5f}" for j in res), flush=True)
    out = {"iters": iters, "judges": {}}
    for j, r in res.items():
        t = max(r, key=r.get)
        out["judges"][j] = {"thr": t, "best": r[t], "at_0.70": r[0.7]}
        print(f"{j}: best {r[t]:.5f} at thr {t:.2f} | at 0.70 {r[0.7]:.5f}", flush=True)
    json.dump(out, open(CFG, "w"), indent=1)


def test(judge):
    cfg = json.load(open(CFG))
    if judge == "best":
        judge = max(cfg["judges"], key=lambda j: cfg["judges"][j]["best"])
    names = ["lgb", "xgb"] if judge == "avg" else [judge]
    d, _, _ = copy_free_rows()
    D, y, w = X.design(d), d.y.values, d.wt.values
    preds = [FIT[n](D, y, w, rounds=int(cfg["iters"][n] * 1.15))[0] for n in names]
    judge_q = lambda Z: np.mean([p(Z) for p in preds], axis=0)
    k = pd.read_parquet(f"{X.XD}/p5_test{X.TAG}.parquet")
    b = X.rows("test", k)                                   # each record's argmax S1 under the mean of A and B
    t = context(b)
    idx = pd.MultiIndex.from_arrays([t.rid.values, t.sid.values])
    qs = []
    for col in ("pA", "pB"):
        bv = b.copy()
        bv["p"], bv["p2"], bv["rec_n20"] = record_inputs(k, k[col].values, bv.i.values)
        dv = context(bv)
        qs.append(pd.Series(judge_q(X.design(dv)), index=pd.MultiIndex.from_arrays([dv.rid.values, dv.sid.values]))
                  .loc[idx].values)
    t["q"] = (qs[0] + qs[1]) / 2
    t["c"] = load("test", 1).country.values[t.sid.values]
    t[["rid", "sid", "q", "c"]].to_parquet(f"{X.XD}/test_q{X.TAG}J.parquet")
    link = f"{X.XD}/p5_test{X.TAG}J.parquet"
    if not os.path.exists(link):
        os.symlink(f"p5_test{X.TAG}.parquet", link)
    json.dump({"judge": judge, "thr": cfg["judges"][judge]["thr"]}, open(f"{X.XD}/judge{X.TAG}J.json", "w"))
    print(f"wrote test_q{X.TAG}J with judge {judge}, validation thr {cfg['judges'][judge]['thr']}", flush=True)


if __name__ == "__main__":
    if "test" in sys.argv[1:]:
        test(os.environ.get("JUDGE", "best"))
    else:
        compare()
