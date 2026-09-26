"""v5 = v4 (cross-fitted stage 1 + entity-context stage 2) with cross-encoder features (x_ce.py).
The CE saw S1 folds 0-3, so every train / validation set here drops records with fold 0-3 (owned records follow
their owner's fold, distractors their hash fold).
  python x_stage.py s1     # stage 1, cross-fitted: folds 4-6 -> A, 7-9 -> B  => x/oof5_train, x/p5_test
  python x_stage.py cv     # stage 2: fit entity folds 4-7, test-like validation on 8-9, refit on 4-9
  python x_stage.py test   # write $OUT/matching_results.tsv + candidate_pairs.tsv"""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, ROOT, load
from harness import truth_arrays, score
from match import to_sets, write_tsv
from stage2 import best_rows, context, S1F, PARAMS as P2
import x_feats as XF

XD = f"{WORK}/x"
SHARE = float(os.environ.get("SHARE", 0.39))
OUT = os.environ.get("OUT", f"{ROOT}/output_v5{os.environ.get('TAG', '')}")
NOCE = bool(os.environ.get("NOCE"))   # baseline: v4 stage 1, no CE features, same clean validation set
TAG = "_noce" if NOCE else os.environ.get("TAG", "")   # suffix for stage-2 artefacts
CE_TAGS = os.environ.get("CE_TAGS", "").split(",")   # x/ce{tag}_{split}.npy files, e.g. ",_base" = small + base
CEF = [] if NOCE else [f"ce{t}{c}" for t in CE_TAGS for c in ("", "_rank", "_gap", "_s1_rank", "_s1_npos")]
EXF = [] if NOCE or os.environ.get("NOEXTRA") else [c for c in XF.COLS   # x_feats.py; EXF_DROP=a,b ablates columns
                                                      if c not in os.environ.get("EXF_DROP", "").split(",")]
# stage-1 cross-fitting over the S1 folds 4-9: S1K=2 (two models, as v5-v8), 3 or 6 (each model sees more data,
# test averages more models)
S1K = int(os.environ.get("S1K", 2))
# SELFTRAIN=1: test countries with no labelled training records (found from the data) get pseudo-labels from the
# stage-1 mean and are re-scored by models that also learn from them (see self_train)
SELFTRAIN, PL_HI, PL_LO = bool(os.environ.get("SELFTRAIN")), 0.97, 0.03
GROUPS = {2: [[4, 5, 6], [7, 8, 9]], 3: [[4, 5], [6, 7], [8, 9]], 6: [[f] for f in range(4, 10)]}[S1K]
WORDF = ["w_add_max", "w_add_sum", "w_add_n8", "w_addrate_max", "w_drop_max", "w_n_add", "w_n_drop"] \
    if os.environ.get("WORDS") else []   # words.py: distractor-word model, leave-one-country-out
LLRF = [f"{f}_{e}_{t}" for f in ("n", "a") for e in ("add", "drop") for t in ("sum", "min", "max", "cnt")]     if os.environ.get("LLR") else []   # x_llr.py word-edit log-likelihood ratios


def _other_max(g, x):
    """per row: max of x over the OTHER rows of its group g (-inf if alone)."""
    o = np.lexsort((-x, g)); gs, xs = g[o], x[o]
    first = np.r_[True, gs[1:] != gs[:-1]]; gid = np.cumsum(first) - 1; st = np.flatnonzero(first)
    size = np.diff(np.r_[st, len(xs)])
    top1 = xs[st][gid]; top2 = np.where(size > 1, xs[np.minimum(st + 1, len(xs) - 1)], -np.inf)[gid]
    out = np.empty(len(x)); out[o] = np.where(first, top2, top1)
    rank = np.empty(len(x)); rank[o] = np.arange(len(xs)) - st[gid]
    return out, rank


def ce_feats(keys, ce, t=""):
    ce = np.nan_to_num(ce.astype(np.float64), nan=-20.0)
    r, s = keys.rid.values, keys.sid.values
    om, rk = _other_max(r, ce)
    _, srk = _other_max(s, ce)
    npos = np.bincount(s, weights=(ce > 0).astype(float), minlength=s.max() + 1)[s] - (ce > 0)
    return pd.DataFrame({f"ce{t}": ce, f"ce{t}_rank": rk, f"ce{t}_gap": ce - om, f"ce{t}_s1_rank": srk,
                         f"ce{t}_s1_npos": npos}, dtype=np.float32)


def all_ce(keys, split):
    return pd.concat([ce_feats(keys, np.load(f"{XD}/ce{t}_{split}.npy"), t) for t in CE_TAGS], axis=1)


def stage1_data(split):
    d = pd.read_parquet(f"{WORK}/feats2_{split}.parquet")
    keys = d[["rid", "sid"]]
    parts = [d.drop(columns=["rid", "sid"]).reset_index(drop=True), all_ce(keys, split)]
    if EXF:
        parts.append(pd.read_parquet(f"{XD}/extra_{split}.parquet", columns=EXF))
    if LLRF:
        parts.append(pd.read_parquet(f"{XD}/llr_{split}.parquet", columns=LLRF))
    if WORDF:
        parts.append(pd.read_parquet(f"{XD}/words_{split}.parquet", columns=WORDF))
    F = pd.concat(parts, axis=1)
    return keys, F


def s1():
    _, _, ts, s1f, rf = truth_arrays()
    keys, F = stage1_data("train")
    y = ts[keys.rid.values] == keys.sid.values
    kf = rf[keys.rid.values]
    grp = [np.isin(kf, g) for g in GROUPS]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.5, bagging_freq=1, num_threads=32, verbose=-1, seed=0)
    models = []
    for i, g in enumerate(grp):   # model i learns the other groups, is early-stopped on and predicts group i
        tr = np.any([grp[j] for j in range(len(grp)) if j != i], axis=0)
        m = lgb.train(params, lgb.Dataset(F[tr], y[tr]), 4000, valid_sets=[lgb.Dataset(F[g], y[g])],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        m.save_model(f"{XD}/gbm5_k{S1K}_{i}{TAG}.txt"); models.append(m)
        print("model", i, "held-out folds", GROUPS[i], "best iter", m.best_iteration, flush=True)
    print(pd.Series(models[0].feature_importance("gain"), index=F.columns).sort_values(ascending=False)
          .round(0).head(20).to_string(), flush=True)
    P = np.column_stack([m.predict(F, num_threads=32) for m in models])
    oof = P.mean(axis=1)                  # records of folds 0-3 (no model is out of fold for them): the mean
    for i, g in enumerate(grp):
        oof[g] = P[g, i]
    keys.assign(p=oof.astype(np.float32)).to_parquet(f"{XD}/oof5_train{TAG}.parquet")
    del P
    kt, Ft = stage1_data("test")
    Pt = np.column_stack([m.predict(Ft, num_threads=32) for m in models]).astype(np.float32)
    if SELFTRAIN:
        Pt = self_train(F, y, np.any(grp, axis=0), ts, kt, Ft, Pt, params,
                        int(np.mean([m.best_iteration for m in models]) * 1.15))
    del F
    # p = mean of the stage-1 models; pm0, pm1, ... kept so stage 2 can be scored once per model (stage2.score_per_model)
    kt.assign(p=Pt.mean(axis=1), **{f"pm{i}": Pt[:, i] for i in range(Pt.shape[1])}).to_parquet(f"{XD}/p5_test{TAG}.parquet")
    print("stage 1 done", flush=True)


def self_train(F, y, tr, ts, kt, Ft, Pt, params, n_iter):
    """Self-training for test countries that have no labelled training records (an open set, found from the data).
    Pseudo-labels from the stage-1 mean: a record whose best pair has p >= PL_HI matches that pair (its other pairs
    do not); a record whose best pair has p <= PL_LO matches nothing. Two models, each fitted on every labelled
    training pair (tr) plus the pseudo-labels of one half of those records, re-score the other half, so no record is
    scored by a model that saw its own pseudo-label. Returns Pt with those records' columns replaced.
    Simulated on train with one country hidden (analysis g7_selftrain_loco.py): pseudo-labels 99.0-99.9% correct;
    F0.5 +0.00055 when the hidden country differs more (US hidden), +-0 when it is close (India hidden)."""
    other_tr = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    labelled = set(other_tr.country.values[ts >= 0])
    ct = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values
    rid = kt.rid.values
    unl = ~np.isin(ct[rid], list(labelled))
    if not unl.any():
        print("self-training: every test country has training labels", flush=True)
        return Pt
    p = Pt.mean(axis=1)
    ui = np.flatnonzero(unl)
    o = ui[np.lexsort((-p[ui], rid[ui]))]
    best = o[np.r_[True, rid[o][1:] != rid[o][:-1]]]
    pos_rec, neg_rec = rid[best][p[best] >= PL_HI], rid[best][p[best] <= PL_LO]
    is_best = np.zeros(len(rid), bool); is_best[best] = True
    plab = np.full(len(rid), -1, np.int8)
    in_pos = unl & np.isin(rid, pos_rec)
    plab[in_pos] = is_best[in_pos]
    plab[unl & np.isin(rid, neg_rec)] = 0
    print(f"self-training on {sorted(set(ct[rid[ui]]))}: {len(best)} records, pseudo-labelled {len(pos_rec)} match / "
          f"{len(neg_rec)} none; {n_iter} rounds", flush=True)
    half = (rid % 2).astype(bool)
    out = Pt.copy()
    for h in (False, True):
        pl = (plab >= 0) & (half != h)
        m = lgb.train(params, lgb.Dataset(pd.concat([F[tr], Ft[pl]], ignore_index=True),
                                          np.r_[y[tr], plab[pl] == 1]), n_iter)
        part = unl & (half == h)
        out[part] = m.predict(Ft[part], num_threads=32).astype(np.float32)[:, None]
    moved = np.abs(out[unl].mean(axis=1) - p[unl])
    print(f"self-training: mean |change| of p {moved.mean():.4f}; best pairs >= 0.5 before "
          f"{(p[best] >= 0.5).mean():.4f} after {(out[best].mean(axis=1) >= 0.5).mean():.4f}", flush=True)
    return out


def rows(split, k):
    """one row per record: its stage-1 argmax S1 + that pair's stage-1 / CE features."""
    b = best_rows(k)
    F = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=S1F)
    parts = [b, F.iloc[b.i.values].reset_index(drop=True)]
    if not NOCE:
        C = all_ce(pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"]), split)
        parts.append(C.iloc[b.i.values].reset_index(drop=True))
    if EXF:
        parts.append(pd.read_parquet(f"{XD}/extra_{split}.parquet", columns=EXF).iloc[b.i.values].reset_index(drop=True))
    if LLRF:
        parts.append(pd.read_parquet(f"{XD}/llr_{split}.parquet", columns=LLRF).iloc[b.i.values].reset_index(drop=True))
    if WORDF:
        parts.append(pd.read_parquet(f"{XD}/words_{split}.parquet", columns=WORDF).iloc[b.i.values].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def train_rows():
    s1_, _, ts, s1f, rf = truth_arrays()
    b = rows("train", pd.read_parquet(f"{WORK}/oof_train.parquet" if NOCE else f"{XD}/oof5_train{TAG}.parquet"))
    matched = ts >= 0
    r, s = b.rid.values, b.sid.values
    keep_m = matched[r] & (rf[r] >= 4) & (s1f[s] >= 4)
    dis = ~matched[r] & (rf[r] >= 4) & (s1f[s] >= 4)          # distractors the CE never saw
    rng = np.random.default_rng(0)
    pool = np.where(~matched & (rf >= 4))[0]
    w = np.bincount(rng.choice(pool, int(SHARE / (1 - SHARE) * matched.sum()), replace=True), minlength=len(ts))
    bd = b[dis]
    d = pd.concat([b[keep_m], bd.loc[bd.index.repeat(w[bd.rid.values])]], ignore_index=True)
    d["y"] = ts[d.rid.values] == d.sid.values
    d["fold"] = s1f[d.sid.values]
    return context(d), np.bincount(ts[matched], minlength=len(s1_)), s1f


def design(d):
    X = d[["p", "p2"] + [c for c in d.columns if c.startswith("e_")] + ["rec_n20"] + S1F + CEF + EXF + LLRF + WORDF].astype(np.float32)
    return X.assign(margin=X.p - X.p2)


def cv():
    d, T, s1f = train_rows()
    X, y = design(d), d.y.values
    tr, va = d.fold.values <= 7, d.fold.values >= 8
    print("rows", len(d), "train", tr.sum(), "valid", va.sum(), flush=True)
    m = lgb.train(P2, lgb.Dataset(X[tr], y[tr]), 4000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    print(pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False).round(0).head(15).to_string())
    q = m.predict(X[va], num_threads=32)
    sid, tru, ents = d.sid.values[va], y[va], np.where(s1f >= 8)[0]
    d.loc[va, ["rid", "sid", "p", "y"]].assign(q=q).to_parquet(f"{XD}/s5_val{TAG}.parquet")
    res = {}
    for thr in [0.5, 0.6]:
        print(f"  stage-1 thr {thr}: {score(sid, d.p.values[va] >= thr, tru, T, ents):.5f}", flush=True)
    for thr in np.arange(0.3, 0.91, 0.05):
        res[round(float(thr), 2)] = sc = score(sid, q >= thr, tru, T, ents)
        print(f"  stage-2 thr {thr:.2f}: {sc:.5f}", flush=True)
    thr = max(res, key=res.get)
    json.dump({"thr": thr, "val": res[thr], "iters": m.best_iteration}, open(f"{XD}/gbm5s2_cfg{TAG}.json", "w"))
    m2 = lgb.train(P2, lgb.Dataset(X, y), int(m.best_iteration * 1.15))
    m2.save_model(f"{XD}/gbm5s2{TAG}.txt")
    print("best thr", thr, res[thr], flush=True)


def test():
    thr = float(os.environ.get("THR", json.load(open(f"{XD}/gbm5s2_cfg{TAG}.json"))["thr"]))
    s1_ = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
    k = pd.read_parquet(f"{XD}/p5_test{TAG}.parquet")
    d = context(rows("test", k))
    q = lgb.Booster(model_file=f"{XD}/gbm5s2{TAG}.txt").predict(design(d), num_threads=32)
    acc = d[q >= thr][["rid", "sid"]]
    os.makedirs(OUT, exist_ok=True)
    order = s1_.entity_id.tolist()
    write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(k, s1_, other), order)
    write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(acc, s1_, other), order)
    print("wrote", OUT, "thr", thr, "accepted", len(acc),
          pd.Series(s1_.country.values[acc.sid.values]).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    {"s1": s1, "cv": cv, "test": test}[sys.argv[1]]()
