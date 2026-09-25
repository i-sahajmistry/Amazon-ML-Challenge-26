"""Stage 2 over each record's TOP-2 stage-1 candidates, so a record can move to its runner-up entity.
Entity context comes from every record's top-1 claim (no double counting); a slot-1 row sees its entity's claims
as "others". Decision per record: the row with the higher q, accepted if q >= thr. Uses x_stage_multi artefacts
(TAG / CE_TAGS env as there).
  python x_top2.py cv     # fit entity folds 4-7, validate 8-9: re-ranking vs fixed argmax with the same model"""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb
from harness import truth_arrays, score
import x_stage_multi as X

P2 = dict(X.P2)


def top2(k):
    """k: rid, sid, p (feature-file order). Rows = each record's two best candidates, with slot / p_other / rec_n20."""
    k = k.assign(i=np.arange(len(k), dtype=np.int64)).sort_values(["rid", "p"], ascending=[True, False])
    r = k.rid.values
    first = np.r_[True, r[1:] != r[:-1]]
    pos = np.arange(len(k)) - np.flatnonzero(first)[np.cumsum(first) - 1]
    n20 = k.assign(t=k.p >= 0.2).groupby("rid", sort=True).t.transform("sum").values
    t = k[pos < 2].assign(slot=pos[pos < 2], rec_n20=n20[pos < 2]).reset_index(drop=True)
    rr, p = t.rid.values, t.p.values
    nxt = np.r_[rr[1:] == rr[:-1], False]; prv = np.r_[False, rr[1:] == rr[:-1]]
    t["p_other"] = np.where(t.slot.values == 0, np.where(nxt, np.r_[p[1:], 0], 0), np.where(prv, np.r_[0, p[:-1]], 0))
    return t


def ent_ctx(t, src3):
    """entity features for every row from slot-0 claims of all record copies (self excluded)."""
    top = t.slot.values == 0
    s, p, own = t.sid.values, t.p.values.astype(np.float64), top.astype(np.float64)
    n = s.max() + 1
    agg = lambda w: np.bincount(s[top], weights=w[top], minlength=n)[s]
    one = np.ones(len(t))
    out = {"e_n": agg(one) - own, "e_sum": agg(p) - p * own}
    for th in (0.9, 0.5, 0.2):
        ind = (p >= th).astype(np.float64)
        out[f"e_ge{int(th * 100)}"] = agg(ind) - ind * own
    ind = (p >= 0.5).astype(np.float64); s3 = src3.astype(np.float64)
    same = np.where(s3 == 1, agg(ind * s3) - ind * s3 * own, agg(ind * (1 - s3)) - ind * (1 - s3) * own)
    out["e_ge50_same_src"], out["e_ge50_other_src"] = same, out["e_ge50"] - same
    # rank among slot-0 claims of the entity (claims with higher p) and max p of the other claims
    key = np.sort(s[top] + 0.5 * (1 - p[top]))
    lo = np.searchsorted(key, s.astype(np.float64), "left")
    out["e_rank"] = np.searchsorted(key, s + 0.5 * (1 - p), "left") - lo
    mx = np.full(n, 0.0); np.maximum.at(mx, s[top], p[top])
    o = np.lexsort((-p[top], s[top])); st, pt = s[top][o], p[top][o]
    f0 = np.r_[True, st[1:] != st[:-1]]
    sec = np.zeros(n); idx = np.flatnonzero(f0) + 1
    ok = idx < len(st); ok[ok] &= st[idx[ok]] == st[idx[ok] - 1]
    sec[st[np.flatnonzero(f0)[ok]]] = pt[idx[ok]]
    self_max = top & (p >= mx[s])
    out["e_max_other"] = np.where(self_max, sec[s], mx[s])
    return pd.DataFrame(out, index=t.index)


def features(split, t):
    F = pd.read_parquet(f"{X.WORK}/feats2_{split}.parquet", columns=X.S1F).iloc[t.i.values].reset_index(drop=True)
    parts = [F, X.all_ce(pd.read_parquet(f"{X.WORK}/feats2_{split}.parquet", columns=["rid", "sid"]), split)
             .iloc[t.i.values].reset_index(drop=True)]
    if X.EXF:
        parts.append(pd.read_parquet(f"{X.XD}/extra_{split}.parquet", columns=X.EXF).iloc[t.i.values].reset_index(drop=True))
    return pd.concat(parts, axis=1)


def design(t, src3):
    base = t[["p", "p_other", "slot", "rec_n20"]].reset_index(drop=True)
    return pd.concat([base, ent_ctx(t, src3).reset_index(drop=True), t[X.S1F + X.CEF + X.EXF].reset_index(drop=True)],
                     axis=1).astype(np.float32).assign(margin=lambda x: x.p - x.p_other)


def cv():
    s1_, other, ts, s1f, rf = truth_arrays()
    t = top2(pd.read_parquet(f"{X.XD}/oof5_train{X.TAG}.parquet"))
    t = pd.concat([t.reset_index(drop=True), features("train", t)], axis=1)
    matched = ts >= 0
    head = t[t.slot.values == 0]
    f1 = pd.Series(s1f[head.sid.values], index=head.rid.values)          # record fold = its slot-0 entity's fold
    fr = f1.reindex(t.rid.values).values
    r = t.rid.values
    keep = (rf[r] >= 4) & (fr >= 4)
    rng = np.random.default_rng(0)
    pool = np.where(~matched & (rf >= 4))[0]
    w = np.bincount(rng.choice(pool, int(X.SHARE / (1 - X.SHARE) * matched.sum()), replace=True), minlength=len(ts))
    rep = np.where(matched[r], 1, w[r]) * keep
    t = t.loc[t.index.repeat(rep)].reset_index(drop=True)
    # one id per record copy (copies of a distractor are separate claimants); slot 0 and 1 of copy c share it
    c = t.groupby(["rid", "slot"]).cumcount().values
    t["uid"] = t.groupby([t.rid.values, c]).ngroup().values
    t["y"] = ts[t.rid.values] == t.sid.values
    t["fold"] = f1.reindex(t.rid.values).values
    src3 = other.entity_id.str.startswith("S3").values[t.rid.values]
    Xd, y = design(t, src3), t.y.values
    tr, va = t.fold.values <= 7, t.fold.values >= 8
    print("rows", len(t), "train", tr.sum(), "valid", va.sum(), "slot-1 share", round((t.slot == 1).mean(), 3), flush=True)
    m = lgb.train(P2, lgb.Dataset(Xd[tr], y[tr]), 4000, valid_sets=[lgb.Dataset(Xd[va], y[va])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)])
    q = m.predict(Xd[va], num_threads=32)
    v = t.loc[va, ["uid", "sid", "slot", "y"]].assign(q=q)
    T = np.bincount(ts[matched], minlength=len(s1_)); ents = np.where(s1f >= 8)[0]
    best = v.sort_values(["uid", "q"], ascending=[True, False]).drop_duplicates("uid")
    fixed = v[v.slot == 0]
    print("switched to runner-up:", int((best.slot == 1).sum()), "of", len(best), flush=True)
    for thr in np.arange(0.5, 0.86, 0.05):
        a = score(best.sid.values, best.q.values >= thr, best.y.values, T, ents)
        b = score(fixed.sid.values, fixed.q.values >= thr, fixed.y.values, T, ents)
        print(f"  thr {thr:.2f}  top-2 re-rank {a:.5f}   fixed argmax {b:.5f}", flush=True)


if __name__ == "__main__":
    {"cv": cv}[sys.argv[1]]()
