"""Test-like evaluation harness (fold-9 S1 entities, clean records, distractors resampled to ~39%).
build(k): k = DataFrame(rid, sid, p) for train candidate pairs. Returns per-row arrays (sid, p, is_true) of every
record copy whose best candidate is a fold-9 entity, and T (true match count) per S1."""
import numpy as np, pandas as pd
from common import load, s1_fold
from match import assign

def truth_arrays():
    s1 = load("train", 1); other = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    gt = load("train", "ground_truth")
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m").query("m != \"\"")
    rid_of = pd.Series(np.arange(len(other)), index=other.entity_id)
    ts = np.full(len(other), -1, np.int64)
    ts[rid_of.loc[pairs.m].values] = pd.Series(np.arange(len(s1)), index=s1.entity_id).loc[pairs.source1_entity_id].values
    s1f = s1_fold(s1.entity_id.tolist()); rf = s1_fold(other.entity_id.tolist()); rf[ts >= 0] = s1f[ts[ts >= 0]]
    return s1, other, ts, s1f, rf

def build(k, ts, s1f, rf, eval_folds=(9,), dis_folds=(0, 1, 2, 3, 9), share=0.39, seed=0):
    matched = ts >= 0
    best = assign(k[["rid", "sid"]], k.p.values, 0.0)
    rng = np.random.default_rng(seed)
    pool = np.where(~matched & np.isin(rf, dis_folds))[0]
    w = np.bincount(rng.choice(pool, int(share / (1 - share) * matched.sum()), replace=True), minlength=len(ts))
    ev = np.isin(s1f, eval_folds)
    b = best[ev[best.sid.values]]
    r = b.rid.values
    keep_m = matched[r] & np.isin(rf[r], eval_folds)            # clean matched records (never trained on)
    bm = b[keep_m]; bd = b[~matched[r]]
    reps = w[bd.rid.values]
    sid = np.r_[bm.sid.values, np.repeat(bd.sid.values, reps)]
    p = np.r_[bm.p.values, np.repeat(bd.p.values, reps)]
    tru = np.r_[ts[bm.rid.values] == bm.sid.values, np.zeros(reps.sum(), bool)]
    T = np.bincount(ts[matched], minlength=len(s1f))
    return sid, p, tru, T, np.where(ev)[0]

def score(sid, acc, tru, T, ents):
    n = len(T)
    TP = np.bincount(sid[acc & tru], minlength=n)[ents]; P = np.bincount(sid[acc], minlength=n)[ents]; Tt = T[ents]
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(P > 0, TP / np.maximum(P, 1), 0); rc = TP / np.maximum(Tt, 1)
        f = np.where(Tt == 0, (P == 0).astype(float), np.where(TP > 0, 1.25 * pr * rc / (0.25 * pr + rc), 0))
    return f.mean()
