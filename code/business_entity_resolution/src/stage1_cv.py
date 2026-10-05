"""Stage 1, cross-fitted: two pairwise LightGBM models on complementary S1 folds give out-of-fold (OOF)
probabilities for every train pair (needed to train stage 2 without leakage); test = mean of both models.
  folds 4-6 -> model A, folds 7-9 -> model B; distractors follow their hash fold; folds 0-3 (embedder) -> mean."""
import numpy as np, pandas as pd, lightgbm as lgb
from common import WORK
from match import features
from harness import truth_arrays

s1, other, ts, s1f, rf = truth_arrays()
keys, F, _, _ = features("train")
y = ts[keys.rid.values] == keys.sid.values
kf = rf[keys.rid.values]
gA, gB = np.isin(kf, [4, 5, 6]), np.isin(kf, [7, 8, 9])
print("pairs", len(keys), "A", gA.sum(), "B", gB.sum(), "features", F.shape[1], flush=True)
params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.5, bagging_freq=1, num_threads=32, verbose=-1, seed=0)
models = {}
for name, tr, va in [("A", gA, gB), ("B", gB, gA)]:
    dtr = lgb.Dataset(F[tr], y[tr]); dva = lgb.Dataset(F[va], y[va], reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(f"{WORK}/gbm_{name}.txt"); models[name] = m
    print(name, "best iter", m.best_iteration, flush=True)
imp = pd.Series(models["A"].feature_importance("gain"), index=F.columns).sort_values(ascending=False)
print(imp.round(0).head(25).to_string(), flush=True)
pA = models["A"].predict(F, num_threads=32); pB = models["B"].predict(F, num_threads=32)
p = np.where(gB, pA, np.where(gA, pB, (pA + pB) / 2)).astype(np.float32)
keys.assign(p=p).to_parquet(f"{WORK}/oof_train.parquet")
del F
kt, Ft, _, _ = features("test")
pt = ((models["A"].predict(Ft, num_threads=32) + models["B"].predict(Ft, num_threads=32)) / 2).astype(np.float32)
kt.assign(p=pt).to_parquet(f"{WORK}/p1_test.parquet")
print("done", flush=True)

# quick check on the test-like harness: stage-1 OOF with a global threshold, entities in folds 8-9
from harness import build, score
k = pd.read_parquet(f"{WORK}/oof_train.parquet")
sid, pp, tru, T, ents = build(k, ts, s1f, rf, eval_folds=(8, 9), dis_folds=tuple(range(10)))
for thr in [0.4, 0.5, 0.6, 0.7]:
    print(f"stage-1 OOF, folds 8-9, thr {thr}: {score(sid, pp >= thr, tru, T, ents):.5f}", flush=True)
