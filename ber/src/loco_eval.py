"""Leave-one-country-out: how does v8 do on a train country that no model ever saw? A labelled stand-in for France.

Run after the v8 steps with HOLDOUT=<country>: common.s1_fold gives that country's train records and S1s fold HELD,
so the bi-encoder, shortlist, cross-encoders and stages 1-2 never train on it (dictionaries and word statistics are
label-free and built for every country, as for France). Run on a normal v8 run (no HOLDOUT) it gives the in-country
baseline: the same numbers per country on entity folds 8-9.

Stage 2 is the copy-free weighted LightGBM of x_nocopy.py, fitted on entity folds 4-7 of the trained countries with
early stopping on 8-9. Trained countries are scored on entity folds 8-9 with their out-of-fold stage-1 probabilities
(the validation view). The held-out country is scored on all its S1s exactly like test: each record's argmax S1 under
the mean of stage-1 models A and B, entity context over its own rows, stage 2 once with A's and once with B's
probabilities, averaged. Distractors are weighted to the 39% test share in precision, per group.
Per group: F0.5 by threshold; loss by error kind at the production threshold and at the group's best; where every
true record ended up (accepted / rejected by threshold / out-ranked by another S1 / cut by the shortlist / not in the
retrieved top 20).
  TAG=_v8 CE_TAGS=,_raw WORDS=1 [HOLDOUT=India] python loco_eval.py"""
import os
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

import x_stage_multi as X
from common import HELD, HOLDOUT, WORK
from harness import truth_arrays
from stage2 import context
from x_judge import record_inputs, wscore

THR = float(os.environ.get("THR", 0.70))
THRS = [round(float(t), 2) for t in np.arange(0.30, 0.96, 0.05)]
pd.set_option("display.width", 220)

s1_, other, ts, s1f, rf = truth_arrays()
matched = ts >= 0
T = np.bincount(ts[matched], minlength=len(s1_))
NS = len(s1_)
c_s1 = s1_.country.values
raw_f = s1f.astype(np.int64)
if HOLDOUT:   # crc fold of the held-out S1s, to compare its folds 8-9 with the in-country validation of the same S1s
    h = s1f == HELD
    raw_f[h] = [zlib.crc32(x.encode()) % 10 for x in s1_.entity_id.values[h]]
k = pd.read_parquet(f"{X.XD}/oof5_train{X.TAG}.parquet")
short_keys = k.rid.values.astype(np.int64) * NS + k.sid.values
b = X.rows("train", k)   # each record's argmax S1 (out-of-fold p; mean of A and B for the held-out country)
r, s = b.rid.values, b.sid.values

# ---------------------------------------------------------------- stage 2 as x_nocopy.py (weighted, copy-free)
ok_r, ok_s = (rf[r] >= 4) & (rf[r] < HELD), (s1f[s] >= 4) & (s1f[s] < HELD)
pool = np.where(~matched & (rf >= 4) & (rf < HELD))[0]
W = int(X.SHARE / (1 - X.SHARE) * (matched & (rf < HELD)).sum()) / len(pool)
nc = pd.concat([b[matched[r] & ok_r & ok_s], b[~matched[r] & ok_r & ok_s]], ignore_index=True)
nc = context(nc.assign(y=ts[nc.rid.values] == nc.sid.values, fold=s1f[nc.sid.values]))
nc["wt"] = np.where(matched[nc.rid.values], 1.0, W)
tr, va = nc.fold.values <= 7, nc.fold.values >= 8
m = lgb.train({**X.P2, "num_threads": X.NT},
              lgb.Dataset(X.design(nc[tr]), nc.y.values[tr], weight=nc.wt.values[tr]), 4000,
              valid_sets=[lgb.Dataset(X.design(nc[va]), nc.y.values[va], weight=nc.wt.values[va])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
print(f"stage 2 fitted on entity folds 4-7 of {sorted(set(c_s1[nc.sid.values[tr]]))}, distractor weight {W:.2f}, "
      f"best iteration {m.best_iteration}", flush=True)


def outcomes(ents, rid, sid, acc):
    """where each true record of the entities ents ended up."""
    tr_r = np.flatnonzero(matched & np.isin(ts, ents))
    tkey = tr_r.astype(np.int64) * NS + ts[tr_r]
    row_sid = np.full(len(ts), -1, np.int64); row_acc = np.zeros(len(ts), bool)
    row_sid[rid], row_acc[rid] = sid, acc
    c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid"])
    isr = np.zeros(len(ts), bool); isr[tr_r] = True
    c = c[isr[c.rid.values]]
    in_cand = np.isin(tkey, c.rid.values.astype(np.int64) * NS + c.sid.values)
    in_short = np.isin(tkey, short_keys)
    own = row_sid[tr_r] == ts[tr_r]
    st = np.select([own & row_acc[tr_r], own, in_short, in_cand],
                   ["accepted", "rejected by threshold", "out-ranked by another S1", "cut by the shortlist"],
                   "not retrieved (top 20)")
    return pd.Series(st).value_counts(normalize=True).reindex(
        ["accepted", "rejected by threshold", "out-ranked by another S1", "cut by the shortlist",
         "not retrieved (top 20)"]).fillna(0)


def kinds(ents, sid, acc, y, wt, isd):
    tp = np.bincount(sid[acc & y], minlength=NS)[ents]
    fpd = np.bincount(sid[acc & isd], weights=wt[acc & isd], minlength=NS)[ents]
    fpw = np.bincount(sid[acc & ~y & ~isd], minlength=NS)[ents]
    t = T[ents]
    pr = np.divide(tp, tp + fpd + fpw, out=np.zeros(len(t)), where=tp + fpd + fpw > 0)
    rc = tp / np.maximum(t, 1)
    f = np.where(t == 0, (tp + fpd + fpw == 0).astype(float),
                 np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros(len(t)), where=tp > 0))
    loss = 1 - f
    ks = {"singleton given a record": (t == 0) & (tp + fpd + fpw > 0),
          "distractor merged": (t > 0) & (fpd > 0),
          "wrong-S1 record merged (no distractor)": (t > 0) & (fpd == 0) & (fpw > 0),
          "missed matches only": (t > 0) & (fpd == 0) & (fpw == 0) & (tp < t)}
    return pd.DataFrame({"entities": {k_: v.mean() for k_, v in ks.items()},
                         "F0.5 lost": {k_: loss[v].sum() / len(t) for k_, v in ks.items()}}), f.mean()


def report(name, ents, rows, q):
    sid, rid, y = rows.sid.values, rows.rid.values, rows.y.values
    wt, isd = rows.wt.values, ~matched[rows.rid.values]
    curve = {t: wscore(sid, q >= t, y, wt, T, ents) for t in THRS}
    best = max(curve, key=curve.get)
    print(f"\n==================== {name}: {len(ents):,} S1, {len(rows):,} records "
          f"({isd.mean():.3f} distractors, weighted to 39%)")
    print(f"F0.5 @ {THR}: {wscore(sid, q >= THR, y, wt, T, ents):.5f}    best F0.5 {curve[best]:.5f} @ thr {best}")
    print("F0.5 by threshold: " + "  ".join(f"{t:.2f}:{v:.5f}" for t, v in curve.items()))
    for t_ in sorted({THR, best}):
        kt, _ = kinds(ents, sid, q >= t_, y, wt, isd)
        print(f"loss by error kind @ {t_}:\n" + kt.round(5).to_string())
    print(f"true records by outcome @ {THR}:\n" + outcomes(ents, rid, sid, q >= THR).round(5).to_string(), flush=True)
    return {"group": name, "S1": len(ents), f"F0.5@{THR}": curve.get(THR, wscore(sid, q >= THR, y, wt, T, ents)),
            "best F0.5": curve[best], "best thr": best}


summary = []
# ---------------------------------------------------------------- trained countries, entity folds 8-9
vq = m.predict(X.design(nc[va]), num_threads=X.NT)
vr = nc[va].reset_index(drop=True)
for c in sorted(set(c_s1[vr.sid.values])):
    sel = c_s1[vr.sid.values] == c
    ents = np.where((s1f >= 8) & (s1f < HELD) & (c_s1 == c))[0]
    summary.append(report(f"{c}, trained on (entity folds 8-9)", ents, vr[sel], vq[sel]))

# ---------------------------------------------------------------- held-out country, scored like test
if HOLDOUT:
    hb = b[rf[r] == HELD].reset_index(drop=True)
    hm = matched[hb.rid.values]
    Wh = X.SHARE / (1 - X.SHARE) * (matched & (rf == HELD)).sum() / (~matched & (rf == HELD)).sum()
    hd = context(hb.assign(y=ts[hb.rid.values] == hb.sid.values))
    idx = pd.MultiIndex.from_arrays([hd.rid.values, hd.sid.values])
    qs = []
    for col in ("pA", "pB"):   # stage 2 learned on single-model probabilities: score once per stage-1 model
        bv = hb.copy()
        bv["p"], bv["p2"], bv["rec_n20"] = record_inputs(k, k[col].values, bv.i.values)
        dv = context(bv)
        qs.append(pd.Series(m.predict(X.design(dv), num_threads=X.NT),
                            index=pd.MultiIndex.from_arrays([dv.rid.values, dv.sid.values])).loc[idx].values)
    hq = (qs[0] + qs[1]) / 2
    hd["wt"] = np.where(matched[hd.rid.values], 1.0, Wh)
    # the held-out country's rows and scores, for loco_blend.py (mixing two variants' scores)
    hd[["rid", "sid", "y", "wt"]].assign(q=hq.astype(np.float32)).to_parquet(f"{X.XD}/loco_q{X.TAG}.parquet")
    print(f"\nheld-out {HOLDOUT}: distractor weight {Wh:.2f}", flush=True)
    summary.append(report(f"{HOLDOUT}, never trained on (all S1)", np.where(s1f == HELD)[0], hd, hq))
    ents89 = np.where((s1f == HELD) & (raw_f >= 8))[0]
    sel = np.isin(hd.sid.values, ents89)
    summary.append(report(f"{HOLDOUT}, never trained on (crc folds 8-9)", ents89, hd[sel], hq[sel]))
    # calibration: share of the held-out country's rows the model is sure about vs the trained countries'
    for nm, qq in (("trained, folds 8-9", vq), (f"{HOLDOUT} held out", hq)):
        print(f"stage-2 q bands, {nm}: " + "  ".join(
            f"[{a},{b_}) {((qq >= a) & (qq < b_)).mean():.4f}" for a, b_ in
            ((0, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 0.95), (0.95, 1.01))), flush=True)

print("\nSUMMARY\n" + pd.DataFrame(summary).round(5).to_string(index=False), flush=True)
