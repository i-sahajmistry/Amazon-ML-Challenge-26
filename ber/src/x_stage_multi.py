"""v5 = v4 (cross-fitted stage 1 + entity-context stage 2) with cross-encoder features (x_ce.py).
The CE saw S1 folds 0-3, so every train / validation set here drops records with fold 0-3 (owned records follow
their owner's fold, distractors their hash fold).
  python x_stage.py s1     # stage 1, cross-fitted: folds 4-6 -> A, 7-9 -> B  => x/oof5_train, x/p5_test
  python x_stage.py cv     # stage 2: fit entity folds 4-7, test-like validation on 8-9, refit on 4-9
  python x_stage.py test   # write $OUT/matching_results.tsv + candidate_pairs.tsv"""
import os, re, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, ROOT, load, HOLDOUT, HELD
from harness import truth_arrays, score
from match import to_sets, write_tsv
from stage2 import best_rows, context, S1F, PARAMS as P2

XD = f"{WORK}/x"
NT = int(os.environ.get("NT", 16))
SHARE = float(os.environ.get("SHARE", 0.39))
OUT = os.environ.get("OUT", f"{ROOT}/output_v5{os.environ.get('TAG', '')}")
NOCE = bool(os.environ.get("NOCE"))   # baseline: v4 stage 1, no CE features, same clean validation set
TAG = "_noce" if NOCE else os.environ.get("TAG", "")   # suffix for stage-2 artefacts
CE_TAGS = os.environ.get("CE_TAGS", "").split(",")   # x/ce{tag}_{split}.npy files, e.g. ",_base" = small + base
CEF = [] if NOCE else [f"ce{t}{c}" for t in CE_TAGS for c in ("", "_rank", "_gap", "_s1_rank", "_s1_npos")]
EXF = [] if NOCE or os.environ.get("NOEXTRA") else [  # x_feats.py
    "int_first_eq", "int_a0_in_b", "int_a0_mindiff", "int_jacc", "int_a_only", "int_b_only",
    "legal_added", "legal_dropped", "legal_a", "legal_b", "legal_xor"]
WORDF = ["w_add_max", "w_add_sum", "w_add_n8", "w_addrate_max", "w_drop_max", "w_n_add", "w_n_drop"] \
    if os.environ.get("WORDS") else []   # words.py: country-agnostic distractor-word features
LLRF = [f"{f}_{e}_{t}" for f in ("n", "a") for e in ("add", "drop") for t in ("sum", "min", "max", "cnt")]     if os.environ.get("LLR") else []   # x_llr.py word-edit log-likelihood ratios
PSEUDO_W = float(os.environ.get("PSEUDO_W", 1.0))   # weight of pseudo-labelled stage-1 rows (pseudo.py)
# llm_judge.py: an offline LLM's log-odds that the record and its best S1 are the same business, the best score among
# the record's other judged candidates, and the gap; NaN where the record was not judged (confident records)
LLMF = ["llm", "llm_alt", "llm_gap"] if os.environ.get("LLM") else []
# DROP=<regex>: leave the matching features out of both stages (loco_unseen.pbs: do judges that do not lean on the
# learned readers transfer better to a country no model saw?), e.g. "^ce" = no cross-encoder features
DROP = os.environ.get("DROP", "")


def _drop(F):
    return F.drop(columns=[c for c in F.columns if re.search(DROP, c)]) if DROP else F


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
    F = _drop(pd.concat(parts, axis=1))
    return keys, F


def _pseudo(keys, f):
    """positions in keys of the pairs pseudo.py labelled in file f, and their pseudo-labels."""
    P = pd.read_parquet(f)
    idx = pd.MultiIndex.from_arrays([keys.rid.values, keys.sid.values]).get_indexer(
        pd.MultiIndex.from_arrays([P.rid.values, P.sid.values]))
    assert (idx >= 0).all(), "pseudo-labelled pairs missing from the stage-1 rows"
    return idx, P.y.values.astype(bool)


def s1():
    _, _, ts, s1f, rf = truth_arrays()
    keys, F = stage1_data("train")
    y = ts[keys.rid.values] == keys.sid.values
    kf = rf[keys.rid.values]
    gA, gB = np.isin(kf, [4, 5, 6]), np.isin(kf, [7, 8, 9])
    # self-training (pseudo.py): pseudo-labelled pairs of a country without labels join BOTH judges' training rows.
    # PSEUDO: pairs of the train split (a held-out country, whose true labels are never used); PSEUDO_TEST: of the test split
    trA, trB, w, extra = gA.copy(), gB.copy(), None, None
    if os.environ.get("PSEUDO"):
        idx, yp = _pseudo(keys, os.environ["PSEUDO"])
        y = y.copy(); y[idx] = yp
        trA[idx] = trB[idx] = True
        w = np.ones(len(y)); w[idx] = PSEUDO_W
        print(f"pseudo-labelled train pairs: {len(idx):,} ({yp.mean():.1%} positive), weight {PSEUDO_W}", flush=True)
    if os.environ.get("PSEUDO_TEST"):
        kt0, Ft0 = stage1_data("test")
        idx, yp = _pseudo(kt0, os.environ["PSEUDO_TEST"])
        extra = (Ft0.iloc[idx].reset_index(drop=True), yp, np.full(len(idx), PSEUDO_W))
        del kt0, Ft0
        print(f"pseudo-labelled test pairs: {len(idx):,} ({yp.mean():.1%} positive), weight {PSEUDO_W}", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.5, bagging_freq=1, num_threads=NT, verbose=-1, seed=0)
    models = {}
    for name, tr, va in [("A", trA, gB), ("B", trB, gA)]:
        Xt, yt = F[tr], y[tr]
        wt = None if w is None and extra is None else (np.ones(tr.sum()) if w is None else w[tr])
        if extra is not None:
            Xt, yt, wt = pd.concat([Xt, extra[0]], ignore_index=True), np.r_[yt, extra[1]], np.r_[wt, extra[2]]
        m = lgb.train(params, lgb.Dataset(Xt, yt, weight=wt), 4000, valid_sets=[lgb.Dataset(F[va], y[va])],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        del Xt
        m.save_model(f"{XD}/gbm5_{name}{TAG}.txt"); models[name] = m
        print(name, "best iter", m.best_iteration, flush=True)
    print(pd.Series(models["A"].feature_importance("gain"), index=F.columns).sort_values(ascending=False)
          .round(0).head(20).to_string(), flush=True)
    pA, pB = models["A"].predict(F, num_threads=NT), models["B"].predict(F, num_threads=NT)
    oof = keys.assign(p=np.where(gB, pA, np.where(gA, pB, (pA + pB) / 2)).astype(np.float32))
    if HOLDOUT:   # the held-out country is scored like test: both models, stage 2 once per model (loco_eval.py)
        oof = oof.assign(pA=pA.astype(np.float32), pB=pB.astype(np.float32))
    oof.to_parquet(f"{XD}/oof5_train{TAG}.parquet")
    del F
    if os.environ.get("SKIP_TEST"):
        print("stage 1 done (train only)", flush=True)
        return
    kt, Ft = stage1_data("test")
    pA_t, pB_t = models["A"].predict(Ft, num_threads=NT), models["B"].predict(Ft, num_threads=NT)
    # pA / pB kept so stage 2 can be scored once per stage-1 model (it was trained on single-model probabilities)
    kt.assign(p=((pA_t + pB_t) / 2).astype(np.float32), pA=pA_t.astype(np.float32),
              pB=pB_t.astype(np.float32)).to_parquet(f"{XD}/p5_test{TAG}.parquet")
    print("stage 1 done", flush=True)


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
    if LLMF:
        parts.append(llm_feats(split, b))
    return pd.concat(parts, axis=1)


def llm_feats(split, b):
    """LLMF for each record's row in b (its best S1): the judge's score of that pair, the best judged alternative."""
    L = pd.read_parquet(f"{XD}/llm_{split}.parquet", columns=["rid", "sid", "llm"])
    own = pd.MultiIndex.from_arrays([b.rid.values, b.sid.values])
    s = pd.Series(L.llm.values, index=pd.MultiIndex.from_arrays([L.rid.values, L.sid.values]))
    llm = s.reindex(own).values
    best_sid = pd.Series(b.sid.values, index=b.rid.values)
    alt = L[L.sid.values != best_sid.reindex(L.rid.values).values].groupby("rid").llm.max()
    llm_alt = alt.reindex(b.rid.values).values
    return pd.DataFrame({"llm": llm, "llm_alt": llm_alt, "llm_gap": llm - llm_alt}, dtype=np.float32)


def train_rows():
    s1_, _, ts, s1f, rf = truth_arrays()
    b = rows("train", pd.read_parquet(f"{WORK}/oof_train.parquet" if NOCE else f"{XD}/oof5_train{TAG}.parquet"))
    matched = ts >= 0
    r, s = b.rid.values, b.sid.values
    ok_r, ok_s = (rf[r] >= 4) & (rf[r] < HELD), (s1f[s] >= 4) & (s1f[s] < HELD)   # HELD: held-out country
    keep_m = matched[r] & ok_r & ok_s
    dis = ~matched[r] & ok_r & ok_s          # distractors the CE never saw
    rng = np.random.default_rng(0)
    pool = np.where(~matched & (rf >= 4) & (rf < HELD))[0]
    w = np.bincount(rng.choice(pool, int(SHARE / (1 - SHARE) * matched.sum()), replace=True), minlength=len(ts))
    bd = b[dis]
    d = pd.concat([b[keep_m], bd.loc[bd.index.repeat(w[bd.rid.values])]], ignore_index=True)
    d["y"] = ts[d.rid.values] == d.sid.values
    d["fold"] = s1f[d.sid.values]
    return context(d), np.bincount(ts[matched], minlength=len(s1_)), s1f


def design(d):
    X = d[["p", "p2"] + [c for c in d.columns if c.startswith("e_")] + ["rec_n20"] + S1F + CEF + EXF + LLRF + WORDF
          + LLMF].astype(np.float32)
    return _drop(X).assign(margin=X.p - X.p2)


def cv():
    d, T, s1f = train_rows()
    X, y = design(d), d.y.values
    tr, va = d.fold.values <= 7, d.fold.values >= 8
    print("rows", len(d), "train", tr.sum(), "valid", va.sum(), flush=True)
    m = lgb.train(P2, lgb.Dataset(X[tr], y[tr]), 4000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    print(pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False).round(0).head(15).to_string())
    q = m.predict(X[va], num_threads=NT)
    sid, tru, ents = d.sid.values[va], y[va], np.where((s1f >= 8) & (s1f < HELD))[0]
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
    q = lgb.Booster(model_file=f"{XD}/gbm5s2{TAG}.txt").predict(design(d), num_threads=NT)
    acc = d[q >= thr][["rid", "sid"]]
    os.makedirs(OUT, exist_ok=True)
    order = s1_.entity_id.tolist()
    write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(k, s1_, other), order)
    write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(acc, s1_, other), order)
    print("wrote", OUT, "thr", thr, "accepted", len(acc),
          pd.Series(s1_.country.values[acc.sid.values]).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    {"s1": s1, "cv": cv, "test": test}[sys.argv[1]]()
