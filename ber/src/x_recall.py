"""Two recall fixes on top of the final decisions (variant_frB = variant_v10seed_dd + these two lists).
  restore_empty  every S1 with no accepted record takes its best candidate if q >= 0.40 (all countries). An S1 left
                 empty scores 0 as soon as it has one true match; on US / India validation (v10 + LLM judge) the rule
                 gains +0.00009 at 0.40 (`python x_recall.py val`).
  restore_nafr   countries without training labels (common.unlabelled): rejected records with no address whose
                 normalised name, or core name, equals their S1's and no other S1's of the country, and that the LLM
                 judge accepts (margin > 0). On US / India validation such records are 96-98% (name) / 76-78% (core
                 name) true and the models accept 95-98% / 68-70% of them; France accepts 84% / 50%. French names reuse
                 a small vocabulary, so a name-only match looks weak to models trained on US / India. Where labels exist
                 the rejected ones are mostly wrong (53-62% / 29-32% true), so, like the same-address fixes, this is
                 applied to the countries without labels only.
Both lists leave out the pairs the same-address fixes reject.
EMPTY_C=unlabelled (or a country list) limits the empty-S1 rule to those countries (variant_frD: US / India already
carry Sarvesh's own empty-S1 rescue).
  python x_recall.py COMB LLM RESTORE REJECT   e.g. _varfr _v10p restore_fr_dd reject_fr_dd
      COMB: x/test_q{COMB}, the stage-2 q after the self-training vetoes (x_final.py SAVE_Q); LLM: x/llm_test{LLM},
      the judge's margins on the unsure rows; RESTORE / REJECT: the same-address lists (x_ddfix.py)
      -> x/restore_empty.parquet, x/restore_nafr.parquet for x_final.py RESTORE=...,restore_empty,restore_nafr
  python x_recall.py val [VQ]                  the empty-S1 rule on US / India validation (x/llm_val{VQ}, default
                                               _v10p: v10's validation rows with the judge's margins), with labels"""
import os, sys, numpy as np, pandas as pd
from common import WORK, load, unlabelled, countries
from match import normed

XD = f"{WORK}/x"
T, TE = 0.70, 0.40


def lists(comb, llm, restore, reject):
    d = pd.read_parquet(f"{XD}/test_q{comb}.parquet")
    key = pd.MultiIndex.from_arrays([d.rid.values, d.sid.values])
    rj = pd.MultiIndex.from_frame(pd.read_parquet(f"{XD}/{reject}.parquet")[["rid", "sid"]])
    for f, val in ((restore, 1.0), (reject, 0.0)):
        r = pd.read_parquet(f"{XD}/{f}.parquet")
        d.loc[key.isin(pd.MultiIndex.from_frame(r[["rid", "sid"]])), "q"] = val
    acc = d.q.values >= T
    ct1 = load("test", 1).country.values
    has = np.bincount(d.sid.values[acc], minlength=len(ct1)) > 0
    b = d[~has[d.sid.values] & (d.q.values >= TE) & ~key.isin(rj)].sort_values("q", ascending=False).drop_duplicates("sid")
    if os.environ.get("EMPTY_C"):
        b = b[b.c.isin(countries(os.environ["EMPTY_C"]))]
    b[["rid", "sid"]].to_parquet(f"{XD}/restore_empty.parquet")
    print("restore_empty", len(b), b.c.value_counts().to_dict(), flush=True)
    t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
    noaddr = t2.na.values[d.rid.values] == ""
    nn = pd.DataFrame({"c": ct1, "nn": t1.nn.values}).value_counts()
    twin = nn.reindex(pd.MultiIndex.from_arrays([d.c.values, t2.nn.values[d.rid.values]])).fillna(0).values >= 2
    same = (t2.nn.values[d.rid.values] == t1.nn.values[d.sid.values]) | (t2.cn.values[d.rid.values] == t1.cn.values[d.sid.values])
    L = pd.read_parquet(f"{XD}/llm_test{llm}.parquet", columns=["rid", "sid", "llm"])
    m = pd.Series(L.llm.values, index=pd.MultiIndex.from_frame(L[["rid", "sid"]])).reindex(key).values
    sel = d.c.isin(unlabelled()).values & ~acc & noaddr & ~twin & same & (m > 0) & ~key.isin(rj)
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_nafr.parquet")
    print("restore_nafr", int(sel.sum()), d[sel].c.value_counts().to_dict(), flush=True)


def val(vq):
    """the empty-S1 rule on US / India validation: v10 + the judge's blend (cross-fitted over two halves of the S1s,
    as x_llm.stack_check), then every S1 with no accepted row takes its best row if q >= t"""
    import os
    os.environ["DFOLD"] = "1"
    from sklearn.linear_model import LogisticRegression
    from harness import truth_arrays
    s1, other, ts, s1f, rf = truth_arrays()
    Tn = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]
    cty = load("train", 1).country.values
    v = pd.read_parquet(f"{XD}/llm_val{vq}.parquet")
    X = lambda d: np.c_[np.log(np.clip(d.q.values, 1e-6, 1 - 1e-6) / np.clip(1 - d.q.values, 1e-6, 1)), d.llm.values]
    band = np.isfinite(v.llm.values); q = v.q.values.copy()
    for h in (0, 1):
        tr, te = band & (v.sid.values % 2 != h), band & (v.sid.values % 2 == h)
        q[te] = LogisticRegression(C=1.0).fit(X(v[tr]), v.y.values[tr], sample_weight=v.wt.values[tr]).predict_proba(X(v[te]))[:, 1]
    sid, y, wt = v.sid.values, v.y.values, v.wt.values

    def f05(acc, e):
        n = len(Tn)
        tp = np.bincount(sid[acc & y], minlength=n)[e]
        fp = np.bincount(sid[acc & ~y], weights=wt[acc & ~y], minlength=n)[e]
        t = Tn[e]
        pr = np.divide(tp, tp + fp, out=np.zeros(len(t)), where=tp + fp > 0)
        rc = tp / np.maximum(t, 1)
        f = np.divide(1.25 * pr * rc, 0.25 * pr + rc, out=np.zeros(len(t)), where=tp > 0)
        return np.where(t == 0, (tp + fp == 0).astype(float), f).mean()

    acc = q >= T
    has = np.bincount(sid[acc], minlength=len(Tn)) > 0
    best = pd.DataFrame({"i": np.arange(len(v)), "sid": sid, "q": q})[~has[sid]].sort_values("q", ascending=False).drop_duplicates("sid")
    for c in ("US", "India", "all"):
        e = ents if c == "all" else ents[cty[ents] == c]
        f0 = f05(acc, e)
        out = []
        for t in (0.2, 0.3, 0.4, 0.5, 0.6):
            k = best[(best.q.values >= t) & np.isin(best.sid.values, e)].i.values
            a = acc.copy(); a[k] = True
            out.append(f"t {t}: +{len(k)} rows (true {y[k].mean() if len(k) else 0:.3f}) {f05(a, e) - f0:+.5f}")
        print(f"{c}: F0.5 {f0:.5f}; " + "; ".join(out), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "val":
        val(sys.argv[2] if len(sys.argv) > 2 else "_v10p")
    else:
        lists(*sys.argv[1:5])
