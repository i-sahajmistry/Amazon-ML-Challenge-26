"""Unbiased validation: every fold-9 S1 entity faces ALL records that retrieve it, not just fold-9 records.
Uses only records the matcher never trained on: fold-9 records and distractors with hash fold 0-3/9.
Distractors are re-weighted (duplicated as new ids) to the full-population share, and to the test share (~39%)."""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
from common import load, s1_fold, f05, WORK
from match import features, assign
P = f"{WORK}/pred_all_train.parquet"
s1 = load("train", 1); other = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
if not os.path.exists(P):
    keys, F, _, _ = features("train")
    m = lgb.Booster(model_file=f"{WORK}/gbm.txt")
    keys = keys.assign(p=m.predict(F, num_threads=32).astype(np.float32)); del F
    keys.to_parquet(P)
k = pd.read_parquet(P)
gt = load("train", "ground_truth")
pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m").query("m != \"\"")
rid_of = pd.Series(np.arange(len(other)), index=other.entity_id)
true_sid = np.full(len(other), -1, np.int64)
true_sid[rid_of.loc[pairs.m].values] = pd.Series(np.arange(len(s1)), index=s1.entity_id).loc[pairs.source1_entity_id].values
s1f = s1_fold(s1.entity_id.tolist()); rf = s1_fold(other.entity_id.tolist()); rf[true_sid >= 0] = s1f[true_sid[true_sid >= 0]]
matched = true_sid >= 0
n_match, n_dis = matched.sum(), (~matched).sum()
print(f"train records: matched {n_match}, distractors {n_dis} ({n_dis/len(other):.1%})")
f9 = np.where(s1f == 9)[0]; f9set = set(f9.tolist())
truth = {s1.entity_id.values[i]: set() for i in f9}
for s, r in zip(pairs.source1_entity_id.values, pairs.m.values):
    if s in truth: truth[s].add(r)
s1c = dict(zip(s1.entity_id, s1.country))
clean_dis = (~matched) & np.isin(rf, [0, 1, 2, 3, 9])     # never in GBM training (hash folds 4-8 were)
val_match = matched & (rf == 9)
rng = np.random.default_rng(0)
def evaluate(dis_per_match, thrs):
    """dis_per_match: distractors per matched record overall; the fold-9 S1s get val_match records +
    clean distractors resampled to the full population size implied by that ratio."""
    want = int(dis_per_match * n_match)                   # distractors in the full population
    pool = np.where(clean_dis)[0]
    draw = rng.choice(pool, want, replace=want > len(pool))
    copies = pd.Series(draw).value_counts()
    kk = k[np.isin(k.sid.values, f9)]                     # only pairs that could land on fold-9 S1s ...
    base = kk[val_match[kk.rid.values]]                   # ... from fold-9 matched records (their argmax is over all their cands)
    dis_rows = []
    # a records decision depends only on its own candidates - each distractors decision once
    dk = k[np.isin(k.rid.values, copies.index.values)]
    da = assign(dk[["rid", "sid"]], dk.p.values, 0.0)     # best candidate + its p for every distractor
    da = da[np.isin(da.sid.values, f9)].assign(w=lambda d: copies.loc[d.rid.values].values)
    va = k[val_match[k.rid.values]]
    vb = assign(va[["rid", "sid"]], va.p.values, 0.0)     # best candidate of each fold-9 matched record
    out = []
    for thr in thrs:
        pred = {s: set() for s in truth}
        ids1 = s1.entity_id.values; ids2 = other.entity_id.values
        v = vb[vb.p >= thr]
        for s, r in zip(v.sid.values, v.rid.values):
            if s in f9set: pred[ids1[s]].add(ids2[r])
        d = da[da.p >= thr]
        for s, r, w in zip(d.sid.values, d.rid.values, d.w.values):
            for c in range(w): pred[ids1[s]].add(f"D{r}_{c}")
        tot = f05(pred, truth)
        byc = {c: round(f05({x: pred[x] for x in truth if s1c[x] == c}, {x: truth[x] for x in truth if s1c[x] == c}), 4) for c in ["US", "India"]}
        out.append((thr, tot, byc, int(d.w.sum()), len(v)))
        print(f"  thr {thr:.2f}  F0.5 {tot:.5f}  {byc}  distractor FPs on fold-9 S1s {int(d.w.sum())}", flush=True)
    return out
thrs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
print("\n== train-like (26% distractors), full population"); evaluate(n_dis / n_match, thrs)
print("\n== test-like (39% distractors)"); evaluate(0.39 / 0.61, thrs)
