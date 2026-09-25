"""Where does stage 2 lose F0.5 on the test-like harness? Scores oracle fixes per error type and prints examples.
  python x_errors.py            # v4: refits the stage-2 CV model (entity folds 4-7 -> validate on 8-9)
  python x_errors.py v5[_tag]   # v5: reads x/s5_val{tag}.parquet written by x_stage.py cv"""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, load
from harness import score, truth_arrays
import stage2 as S

XD = f"{WORK}/x"; os.makedirs(XD, exist_ok=True)
V5 = sys.argv[1] if len(sys.argv) > 1 else ""
TG = V5[2:]
VQ = f"{XD}/s5_val{TG}.parquet" if V5 else f"{XD}/s2_val.parquet"
THR = json.load(open(f"{XD}/gbm5s2_cfg{TG}.json" if V5 else f"{WORK}/gbm2_cfg.json"))["thr"]

if not os.path.exists(VQ):
    d, _, _ = S.train_rows()
    X = S.design(d); y = d.y.values; va = d.fold.values >= 8
    m = lgb.train(S.PARAMS, lgb.Dataset(X[~va], y[~va]), 3000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
    cols = ["rid", "sid", "p", "y", "b_addr_empty", "b_nonascii", "src3"]
    d.loc[va, cols].assign(q=m.predict(X[va], num_threads=32)).to_parquet(VQ)

s1, other, ts, s1f, rf = truth_arrays()
v = pd.read_parquet(VQ)
if V5:   # s5_val lacks the record flags; take them from the feature file via the record id
    o = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    v["b_addr_empty"] = (o.business_address.str.strip() == "").values[v.rid.values]
    v["b_nonascii"] = (~o.business_name.map(str.isascii)).values[v.rid.values]
    v["src3"] = o.entity_id.str.startswith("S3").values[v.rid.values]
matched = ts >= 0
rid, sid, q, tru = v.rid.values, v.sid.values, v.q.values, v.y.values
T = np.bincount(ts[matched], minlength=len(s1))
ents = np.where(s1f >= 8)[0]
D = ~matched[rid]                       # distractor: owns nothing
W = matched[rid] & ~tru                 # real record, wrong entity chosen
C = tru                                 # real record, right entity chosen
acc = q >= THR
sc = lambda a, s=sid, t=tru: score(s, a, t, T, ents)
base = sc(acc)
print(f"rows {len(v)}  distractor {D.sum()}  wrong-entity {W.sum()}  right-entity {C.sum()}  thr {THR}")
print(f"current                          {base:.5f}")
for name, a in [("reject every distractor", acc & ~D), ("reject every wrong-entity row", acc & ~W),
                ("accept every right-entity row", acc | C), ("perfect accept/reject (argmax fixed)", C)]:
    print(f"{name:36s} {sc(a):.5f}  (+{sc(a) - base:.5f})")
# perfect argmax too: every real record whose true entity is among its top-5 candidates is matched correctly
k = pd.read_parquet(f"{XD}/oof5_train.parquet" if V5 else f"{WORK}/oof_train.parquet", columns=["rid", "sid"])
hit = matched[k.rid.values] & (ts[k.rid.values] == k.sid.values)
r = k.rid.values[hit]; r = r[np.isin(s1f[ts[r]], [8, 9])]
one = np.ones(len(r), bool)
print(f"{'candidate ceiling (top-5)':36s} {score(ts[r], one, one, T, ents):.5f}")
print("\naccepted-error counts:", {"distractor FP": int((acc & D).sum()), "wrong-entity FP": int((acc & W).sum()),
                                   "missed right-entity": int((~acc & C).sum())})
for nm, msk in [("distractor FP", acc & D), ("missed right-entity", ~acc & C)]:
    e = v[msk]
    print(f"{nm}: addr empty {e.b_addr_empty.mean():.1%}, non-latin name {e.b_nonascii.mean():.1%}, "
          f"S3 {e.src3.mean():.1%}  (all rows: addr empty {v.b_addr_empty.mean():.1%})")
    print("  q deciles:", np.round(np.quantile(e.q, np.linspace(0.1, 0.9, 9)), 3).tolist())

txt = lambda df, i: f"{df.business_name.values[i]} | {df.business_address.values[i]}"
rng = np.random.default_rng(0)
for nm, msk in [("DISTRACTOR ACCEPTED", acc & D), ("WRONG ENTITY ACCEPTED", acc & W), ("RIGHT ENTITY MISSED", ~acc & C)]:
    idx = np.unique(rid[msk], return_index=True)[1]
    idx = rng.choice(np.flatnonzero(msk)[idx], min(15, len(idx)), replace=False)
    print(f"\n=== {nm}")
    for i in idx:
        print(f"q={q[i]:.2f} p1={v.p.values[i]:.2f} [{s1.country.values[sid[i]]}]\n  S1 : {txt(s1, sid[i])}\n  rec: {txt(other, rid[i])}")
        if W[i]:
            print(f"  true S1: {txt(s1, ts[rid[i]])}")
