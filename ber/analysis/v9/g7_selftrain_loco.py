"""Does self-training help an unseen country? Simulated on train, where labels exist: one train country plays France.

For target country C (India, then US) with the other country as the only labelled source:
  M0   stage 1 fitted on the source country's records (S1 folds 4-9); C is unseen.
  PL   pseudo-labels on C's records (folds 4-9, labels hidden): a record whose best pair has p >= HI is a match
       for that pair (its other pairs negative); a record whose best pair has p <= LO matches nothing.
  M1   source labels + C's pseudo-labels (weight PL_W), cross-fitted over two halves of C's records, so every C
       record is scored by a model that did not see its pseudo-label.
  SUP  upper bound: the source country plus C's own labels of S1 folds 4-7.
Metric: macro F0.5 over C's entities of S1 folds 8-9, each record at its argmax, distractors weighted to the 39%
test share, at the best threshold (and at 0.5): stage 1 only, with v8 features (work/x TAG _v8).
  TAG=_v8 CE_TAGS=,_raw WORDS=1 python g7_selftrain_loco.py"""
import os, json, numpy as np, pandas as pd, lightgbm as lgb
from common import load
from harness import truth_arrays
from x_judge import wscore
import x_stage_multi as X

HI, LO, PL_W = float(os.environ.get("PL_HI", 0.97)), float(os.environ.get("PL_LO", 0.03)), float(os.environ.get("PL_W", 1.0))
NT = int(os.environ.get("NT", 8))
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.5, bagging_freq=1, num_threads=NT, verbose=-1, seed=0)
THRS = np.round(np.arange(0.3, 0.91, 0.05), 2)


def fit(F, y, w, tr):
    return lgb.train(PARAMS, lgb.Dataset(F[tr], y[tr], weight=w[tr]), 400)


def argmax_rows(rid, p):
    """index of each record's best pair."""
    o = np.lexsort((-p, rid))
    return o[np.r_[True, rid[o][1:] != rid[o][:-1]]]


def evaluate(p, rid, sid, y, rec, T, ents, W, matched):
    """F0.5 over entities ents; each record in rec at its argmax pair; distractor rows weighted W."""
    sel = np.flatnonzero(rec[rid])
    b = sel[argmax_rows(rid[sel], p[sel])]
    wt = np.where(matched[rid[b]], 1.0, W)
    res = {float(t): wscore(sid[b], p[b] >= t, y[b], wt, T, ents) for t in THRS}
    t = max(res, key=res.get)
    return {"best": res[t], "thr": t, "at_0.5": res[0.5]}


def main():
    s1, other, ts, s1f, rf = truth_arrays()
    keys, F = X.stage1_data("train")
    rid, sid = keys.rid.values, keys.sid.values
    y = ts[rid] == sid
    c_rec = other.country.values
    matched = ts >= 0
    pool = np.where(~matched & (rf >= 4))[0]
    W = int(0.39 / 0.61 * matched.sum()) / len(pool)
    T = np.bincount(ts[matched], minlength=len(s1))
    out = {}
    for tgt in ("India", "US"):
        src_rec = (c_rec != tgt) & (rf >= 4)
        tgt_rec = (c_rec == tgt) & (rf >= 4)
        ents = np.flatnonzero((s1.country.values == tgt) & (s1f >= 8))
        val_rec = tgt_rec & (np.isin(rf, [8, 9]))
        src, tgt_rows = src_rec[rid], tgt_rec[rid]
        ones = np.ones(len(rid), np.float32)
        r = {}
        m0 = fit(F, y, ones, src)
        p0 = m0.predict(F, num_threads=NT)
        r["M0 (source only)"] = evaluate(p0, rid, sid, y, val_rec, T, ents, W, matched)
        # pseudo-labels on the target records from M0 (no target labels used)
        ti = np.flatnonzero(tgt_rows)
        best = ti[argmax_rows(rid[ti], p0[ti])]
        rec_best_p = pd.Series(p0[best], index=rid[best])
        pos_rec = rec_best_p.index.values[rec_best_p.values >= HI]
        neg_rec = rec_best_p.index.values[rec_best_p.values <= LO]
        is_best = np.zeros(len(rid), bool); is_best[best] = True
        plab = np.full(len(rid), -1, np.int8)                      # -1 = no pseudo-label
        in_pos = np.isin(rid, pos_rec) & tgt_rows
        plab[in_pos] = is_best[in_pos].astype(np.int8)             # best pair 1, the record's other pairs 0
        plab[np.isin(rid, neg_rec) & tgt_rows] = 0
        lab = plab >= 0
        prec_pos = y[lab & (plab == 1)].mean(); prec_neg = 1 - y[lab & (plab == 0)].mean()
        print(f"{tgt}: pseudo-labelled records {len(pos_rec)} match / {len(neg_rec)} none of {len(best)}; "
              f"pair labels correct: positives {prec_pos:.4f}, negatives {prec_neg:.4f}", flush=True)
        # M1: cross-fitted over two halves of the target records
        half = (rid % 2).astype(bool)
        yp = np.where(lab, plab == 1, y)                           # target rows use pseudo-labels only
        wp = np.where(tgt_rows, PL_W, 1.0).astype(np.float32)
        p1 = p0.copy()
        for h in (False, True):
            tr = src | (tgt_rows & lab & (half != h))
            m = fit(F, yp, wp, tr)
            part = tgt_rows & (half == h)
            p1[part] = m.predict(F[part], num_threads=NT)
        r["M1 (+ pseudo-labels)"] = evaluate(p1, rid, sid, y, val_rec, T, ents, W, matched)
        sup = src | (tgt_rows & np.isin(rf[rid], [4, 5, 6, 7]))
        ps = fit(F, y, ones, sup).predict(F, num_threads=NT)
        r["SUP (target labels, folds 4-7)"] = evaluate(ps, rid, sid, y, val_rec, T, ents, W, matched)
        for k, v in r.items():
            print(f"  {tgt} {k:32s} best {v['best']:.5f} at {v['thr']:.2f} | at 0.50 {v['at_0.5']:.5f}", flush=True)
        out[tgt] = {**r, "pseudo": {"pos_records": int(len(pos_rec)), "neg_records": int(len(neg_rec)),
                                    "pos_pair_precision": float(prec_pos), "neg_pair_precision": float(prec_neg)}}
    json.dump(out, open(f"{X.XD}/selftrain_loco{X.TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
