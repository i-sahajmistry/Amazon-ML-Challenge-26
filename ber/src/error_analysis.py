import numpy as np, pandas as pd, json
from common import load, WORK
from match import assign
s1 = load("train", 1); other = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
v = pd.read_parquet(f"{WORK}/val_pred.parquet"); thr = json.load(open(f"{WORK}/gbm_thr.json"))["thr"]
a = assign(v[["rid","sid"]], v.p.values, thr).merge(v[["rid","sid","y"]], on=["rid","sid"])
pos_r = set(v.rid[v.y])
fp = a[~a.y]; tp = a[a.y]
fn_rids = pos_r - set(tp.rid)
in_cand_best_wrong = len(set(fp.rid) & pos_r)
print(f"thr {thr} accepted {len(a)} TP {len(tp)} FP {len(fp)} (of which true record assigned to wrong S1: {in_cand_best_wrong}, record truly unmatched: {len(fp)-in_cand_best_wrong}) FN {len(fn_rids)}")
fnv = v[v.rid.isin(fn_rids) & v.y]
print("FN below threshold with true S1 in candidates:", len(fnv), " p quantiles", fnv.p.quantile([.25,.5,.75]).round(3).tolist())
def show(r, s):
    b, x = other.iloc[r], s1.iloc[s]
    print(f"   S1: {x.business_name} | {x.business_address} | {x.country}\n   R : {b.business_name} | {b.business_address}")
print("\n== False positives"); 
for _, r in fp.sample(8, random_state=0).iterrows(): print(f"  p={r.p:.2f}"); show(r.rid, r.sid)
print("\n== False negatives (true pair scored low)")
for _, r in fnv.sample(8, random_state=0).iterrows(): print(f"  p={r.p:.3f}"); show(r.rid, r.sid)
