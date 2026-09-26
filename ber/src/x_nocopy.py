"""Stage 2 without distractor copies (the issue Sarvesh and Mohanish found in v4; x_stage* inherit it).
train_rows reaches the 39% test distractor share by repeating distractors. Copies of one record point at the same
S1, so the entity-context features see identical twins that test never has, and validation on copies reads high.
Here each distractor appears once with weight W (its average copy count), in the loss and in the validation
precision. Fits entity folds 4-7 with copies (as v5) and without (weighted), and scores both on the same copy-free
folds 8-9: the view Sarvesh's 0.98784 (gap 0.1 + DISTRACTORS=weight) is measured on.
  python x_nocopy.py        (TAG / CE_TAGS env as x_stage_multi: v5 = unset, v6 = _base and ",_base")
  python x_nocopy.py test   # weighted model only: refit on folds 4-9, write x/test_q{TAG}w.parquet for x_final.py
  env PEERS=1 adds house-number peer context (outputs tagged {TAG}p); BAG=n averages n seeds; P2X='{"num_leaves": 127}' overrides stage-2 LightGBM parameters"""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from common import load
from harness import truth_arrays, score, wscore
from stage2 import context
import x_stage_multi as X


s1_, _, ts, s1f, rf = truth_arrays()
b = X.rows("train", pd.read_parquet(f"{X.XD}/oof5_train{X.TAG}.parquet"))
matched = ts >= 0
r, s = b.rid.values, b.sid.values
keep_m = matched[r] & (rf[r] >= 4) & (s1f[s] >= 4)
dis = ~matched[r] & (rf[r] >= 4) & (s1f[s] >= 4)
pool = np.where(~matched & (rf >= 4))[0]
n_draw = int(X.SHARE / (1 - X.SHARE) * matched.sum())
W = n_draw / len(pool)
w = np.bincount(np.random.default_rng(0).choice(pool, n_draw, replace=True), minlength=len(ts))   # as train_rows
bd = b[dis]
T = np.bincount(ts[matched], minlength=len(s1_))
ents = np.where(s1f >= 8)[0]


def ctx(d):
    return X.peers(context(d)) if X.PEERS else context(d)


def prep(d):
    d = d.assign(y=ts[d.rid.values] == d.sid.values, fold=s1f[d.sid.values])
    return ctx(d)


TEST = "test" in sys.argv[1:]   # extra args only rename the run.sh log
dup = None if TEST else prep(pd.concat([b[keep_m], bd.loc[bd.index.repeat(w[bd.rid.values])]], ignore_index=True))
nc = prep(pd.concat([b[keep_m], bd], ignore_index=True))
nc["wt"] = np.where(matched[nc.rid.values], 1.0, W)
print(f"distractor weight W {W:.2f}; rows without copies {len(nc)}", flush=True)
va = nc.fold.values >= 8
Xv, yv, wv = X.design(nc[va]), nc.y.values[va], nc.wt.values[va]
sid_v = nc.sid.values[va]
P = {**X.P2, **json.loads(os.environ.get("P2X", "{}"))}   # stage-2 LightGBM params, optional overrides
BAG = int(os.environ.get("BAG", 1))
models = {}
for name, d, wt in ((("copies", dup, None),) if not TEST else ()) + (("weighted", nc, nc.wt.values),):
    tr = d.fold.values <= 7
    models[name] = lgb.train(P, lgb.Dataset(X.design(d[tr]), d.y.values[tr], weight=None if wt is None else wt[tr]),
                             4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv)],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
q = {k: m.predict(Xv, num_threads=32) for k, m in models.items()}
if TEST:
    # BAG > 1: average that many weighted models with different seeds (bagging / feature sampling differ)
    bag = [models["weighted"]] + [lgb.train({**P, "seed": i, "bagging_seed": i, "feature_fraction_seed": i},
                                            lgb.Dataset(X.design(nc[nc.fold.values <= 7]), nc.y.values[nc.fold.values <= 7],
                                                        weight=nc.wt.values[nc.fold.values <= 7]),
                                            4000, valid_sets=[lgb.Dataset(Xv, yv, weight=wv)],
                                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                                  for i in range(1, BAG)]
    q["weighted"] = np.mean([m.predict(Xv, num_threads=32) for m in bag], axis=0)
    for thr in np.arange(0.5, 0.91, 0.05):
        print(f"{thr:.2f}  weighted-trained, copy-free {wscore(sid_v, q['weighted'] >= thr, yv, wv, T, ents):.5f}", flush=True)
    # validation predictions for error analysis (x_verr.py)
    nc.loc[va, ["rid", "sid", "p", "y", "wt"]].assign(q=q["weighted"]).to_parquet(f"{X.XD}/val_q{X.QTAG}w.parquet")
    full = [lgb.train({**P, "seed": i, "bagging_seed": i, "feature_fraction_seed": i} if i else P,
                      lgb.Dataset(X.design(nc), nc.y.values, weight=nc.wt.values), int(m.best_iteration * 1.15))
            for i, m in enumerate(bag)]
    full[0].save_model(f"{X.XD}/gbm5s2{X.QTAG}w.txt")
    k = pd.read_parquet(f"{X.XD}/p5_test{X.TAG}.parquet")
    d = ctx(X.rows("test", k))
    d["q"] = np.mean([m.predict(X.design(d), num_threads=32) for m in full], axis=0)
    d["c"] = load("test", 1).country.values[d.sid.values]
    d[["rid", "sid", "q", "c"]].to_parquet(f"{X.XD}/test_q{X.QTAG}w.parquet")
    if not os.path.exists(f"{X.XD}/p5_test{X.QTAG}w.parquet"):   # x_final.py reads candidates by the same tag
        os.symlink(f"p5_test{X.TAG}.parquet", f"{X.XD}/p5_test{X.QTAG}w.parquet")
    print("wrote", f"{X.XD}/test_q{X.QTAG}w.parquet", flush=True)
    sys.exit()
vd = dup.fold.values >= 8
q_dup = models["copies"].predict(X.design(dup[vd]), num_threads=32)
print("thr    copies-trained: on copies (as x_stage cv) | copy-free     weighted-trained: copy-free", flush=True)
for thr in np.arange(0.5, 0.91, 0.05):
    a = score(dup.sid.values[vd], q_dup >= thr, dup.y.values[vd], T, ents)
    b1 = wscore(sid_v, q["copies"] >= thr, yv, wv, T, ents)
    b2 = wscore(sid_v, q["weighted"] >= thr, yv, wv, T, ents)
    print(f"{thr:.2f}   {a:.5f} | {b1:.5f}     {b2:.5f}", flush=True)
