"""Stage 2: re-score each S2/S3 record's best S1 candidate with entity context.
Stage 1 scores pairs independently; but whether a 0.5 record is real depends on its entity: with one other
confident match it is usually real (entities average 3.5), with 3+ it is usually a distractor.
Rows = one per record (its argmax S1 from stage 1). Trained on out-of-fold stage-1 probabilities with
distractors resampled to the test share (~39%), so context counts look like test.
  python stage2.py cv    -> fit on entity folds 4-7, validate on 8-9, then refit on 4-9
  python stage2.py test  -> write output/matching_results.tsv + candidate_pairs.tsv"""
import os, re, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, ROOT, load
from harness import truth_arrays, score
from match import to_sets, write_tsv
from decide import select

SHARE = float(os.environ.get("SHARE", 0.39))
OUT = os.environ.get("OUT", f"{ROOT}/output")
S1F = ["score", "rank", "gap_next", "cn_tset", "cn_wj", "cn_eq", "na_tset", "na_wj", "num_first_eq", "num_partial",
       "b_addr_empty", "b_nonascii", "src3", "a_cn_s1_count", "b_cn_s1_count", "cn_a_unshared_max",
       "cn_b_unshared_max", "na_a_unshared_max"]


def best_rows(k):
    """k: rid, sid, p (all candidate pairs, row order = feature file order). One row per record."""
    k = k.assign(i=np.arange(len(k), dtype=np.int64)).sort_values(["rid", "p"], ascending=[True, False])
    first = ~k.rid.duplicated().values
    nxt = np.r_[first[1:], True]                                   # row after this one starts a new record
    p2 = np.where(~nxt, np.r_[k.p.values[1:], 0.0], 0.0)[first]
    n20 = k.assign(t=k.p >= 0.2).groupby("rid", sort=True).t.sum().values
    b = k[first].reset_index(drop=True)
    return b.assign(p2=p2.astype(np.float32), rec_n20=n20.astype(np.float32))


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


def model_cols(k):
    """per-stage-1-model test probability columns saved by x_stage_multi.py s1 (pm0, pm1, ...)."""
    return sorted((c for c in k.columns if re.fullmatch(r"pm\d+", c)), key=lambda c: int(c[2:]))


def score_per_model(k, b, predict):
    """Stage 2 is trained on single-model out-of-fold stage-1 probabilities, but test pairs carry the mean of the
    stage-1 models. Score the test rows b (each record's argmax under the mean) once per stage-1 model, with p, p2,
    rec_n20 and the entity context recomputed from that model's probabilities, and average. predict(d) -> q."""
    d = context(b)
    cols = model_cols(k)
    if not cols:
        d["q"] = predict(d)
        return d
    idx = pd.MultiIndex.from_arrays([d.rid.values, d.sid.values])
    qs = []
    for col in cols:
        bv = b.copy()
        bv["p"], bv["p2"], bv["rec_n20"] = record_inputs(k, k[col].values, bv.i.values)
        dv = context(bv)
        qs.append(pd.Series(predict(dv), index=pd.MultiIndex.from_arrays([dv.rid.values, dv.sid.values])).loc[idx].values)
    d["q"] = np.mean(qs, axis=0)
    return d


def context(d):
    """entity-level features over all rows pointing to the same S1 (self excluded)."""
    d = d.sort_values(["sid", "p"], ascending=[True, False]).reset_index(drop=True)
    sid, p = d.sid.values, d.p.values.astype(np.float64)
    start = np.r_[True, sid[1:] != sid[:-1]]
    gid = np.cumsum(start) - 1
    bounds = np.flatnonzero(np.r_[start, True])
    size = np.diff(bounds)[gid]
    pos = np.arange(len(d)) - bounds[:-1][gid]
    top1 = p[bounds[:-1]][gid]
    top2 = np.where(size > 1, p[np.minimum(bounds[:-1] + 1, len(p) - 1)][gid], 0.0)
    gsum = lambda x: np.bincount(gid, weights=x)[gid]
    d["e_n"] = size - 1
    for t in (0.9, 0.5, 0.2):
        ind = (p >= t).astype(np.float64)
        d[f"e_ge{int(t * 100)}"] = gsum(ind) - ind
    d["e_sum"] = gsum(p) - p
    d["e_max_other"] = np.where(pos == 0, top2, top1)
    d["e_rank"] = pos
    s3 = d.src3.values.astype(np.float64); ind = (p >= 0.5).astype(np.float64)
    same = np.where(s3 == 1, gsum(ind * s3) - ind * s3, gsum(ind * (1 - s3)) - ind * (1 - s3))
    d["e_ge50_same_src"] = same
    d["e_ge50_other_src"] = d.e_ge50.values - same
    return d


FEATS = None


def design(d):
    global FEATS
    FEATS = ["p", "p2"] + [c for c in d.columns if c.startswith("e_")] + ["rec_n20"] + S1F
    X = d[FEATS].astype(np.float32).copy()
    X["margin"] = X.p - X.p2
    return X


def train_rows():
    s1, other, ts, s1f, rf = truth_arrays()
    k = pd.read_parquet(f"{WORK}/oof_train.parquet")
    b = best_rows(k)
    F = pd.read_parquet(f"{WORK}/feats2_train.parquet", columns=S1F)
    b = pd.concat([b, F.iloc[b.i.values].reset_index(drop=True)], axis=1); del F
    matched = ts >= 0
    r, s = b.rid.values, b.sid.values
    keep_m = matched[r] & (rf[r] >= 4) & (s1f[s] >= 4)            # clean OOF matched records, clean entities
    dis = ~matched[r] & (s1f[s] >= 4)
    rng = np.random.default_rng(0)
    pool = np.where(~matched)[0]
    w = np.bincount(rng.choice(pool, int(SHARE / (1 - SHARE) * matched.sum()), replace=True), minlength=len(ts))
    bd = b[dis]
    d = pd.concat([b[keep_m], bd.loc[bd.index.repeat(w[bd.rid.values])]], ignore_index=True)
    d["y"] = ts[d.rid.values] == d.sid.values
    d["fold"] = s1f[d.sid.values]
    T = np.bincount(ts[matched], minlength=len(s1))
    return context(d), T, s1f


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.9,
              bagging_fraction=0.7, bagging_freq=1, num_threads=32, verbose=-1, seed=0)


def cv():
    d, T, s1f = train_rows()
    X = design(d); y = d.y.values
    tr, va = d.fold.values <= 7, d.fold.values >= 8
    print("rows", len(d), "train", tr.sum(), "valid", va.sum(), "pos", y.mean().round(4), flush=True)
    m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), 3000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    print(pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False).round(0).head(20).to_string())
    q = m.predict(X[va], num_threads=32)
    sid, tru, p1 = d.sid.values[va], y[va], d.p.values[va]
    ents = np.where(s1f >= 8)[0]
    res = {}
    for thr in [0.4, 0.5, 0.6, 0.7]:
        print(f"  stage-1 thr {thr}: {score(sid, p1 >= thr, tru, T, ents):.5f}", flush=True)
    for thr in np.arange(0.3, 0.81, 0.05):
        res[round(float(thr), 2)] = sc = score(sid, q >= thr, tru, T, ents)
        print(f"  stage-2 thr {thr:.2f}: {sc:.5f}", flush=True)
    for lam in [1.0, 1.2]:
        print(f"  stage-2 expected-F lam {lam}: {score(sid, select(sid, q, lam), tru, T, ents):.5f}", flush=True)
    thr = max(res, key=res.get)
    json.dump({"thr": thr, "val": res[thr], "iters": m.best_iteration}, open(f"{WORK}/gbm2_cfg.json", "w"))
    # refit on all entity folds 4-9
    m2 = lgb.train(PARAMS, lgb.Dataset(X, y), int(m.best_iteration * 1.15))
    m2.save_model(f"{WORK}/gbm2.txt")
    print("best thr", thr, res[thr], flush=True)


def test():
    cfg = json.load(open(f"{WORK}/gbm2_cfg.json"))
    thr = float(os.environ.get("THR", cfg["thr"]))
    s1 = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
    k = pd.read_parquet(f"{WORK}/p1_test.parquet")
    b = best_rows(k)
    F = pd.read_parquet(f"{WORK}/feats2_test.parquet", columns=S1F)
    b = pd.concat([b, F.iloc[b.i.values].reset_index(drop=True)], axis=1); del F
    d = context(b)
    q = lgb.Booster(model_file=f"{WORK}/gbm2.txt").predict(design(d), num_threads=32)
    acc = d[q >= thr][["rid", "sid"]]
    os.makedirs(OUT, exist_ok=True)
    order = s1.entity_id.tolist()
    write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(k, s1, other), order)
    write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(acc, s1, other), order)
    ctry = s1.country.values[acc.sid.values]
    print("wrote", OUT, "thr", thr, "accepted", len(acc), pd.Series(ctry).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    {"cv": cv, "test": test}[sys.argv[1]]()
