"""Recall fixes on the final scores (after the vetoes and the same-address fixes), from Mohanish's variant_frB.
  restore_empty     an S1 with no accepted record takes its best candidate if q >= 0.40: an S1 left empty scores 0 as
                    soon as it has one true match (US / India validation on the bge + LLM stack: +0.00007, `val`). For
                    the countries without training labels the LLM judge must not say no: the candidates it rejects
                    there are mostly a type word swapped at the S1's own address (the same-building decoy the
                    same-address fixes reject).
  restore_nafr      countries without training labels: a rejected record with no address whose normalised name equals
                    its S1's and no other S1's of the country, and that the judge accepts. US / India models accept
                    93-97% of these records, France 84%: its self-training vetoes learned "no address = reject".
  restore_samename  countries without training labels: a rejected record with its S1's exact name at another house
                    number, the only record of the S1 at that number, outside the decoy shift (1-13 above the S1's
                    number: 76-83% of the decoy-cluster records sit there), that the judge accepts. The data generator
                    moves a true match to another address in every country (US / India 0.05 such records per S1,
                    France 0.034). On US / India validation they belong to their S1 97-98% of the time when no other S1
                    has the name, 93% / 87-89% / 68-78% with 1 / 2 / 3+ namesakes (the rest are distractors; almost
                    never another namesake's record), and the models accept 47-97%; France accepts 5-62%. France's
                    rejected ones get the judge's yes 69-81% of the time, US / India's rejected ones (1-6% true)
                    17-44%: by the judge's validation calibration ~70-85% of them are true, ~92% of those it accepts.
  restore_nacore    countries without training labels: a rejected record with no address whose core name (legal form
                    dropped) equals its S1's while the full name differs ("X SAS" for "X SARL"), no other S1 of the
                    country has that core name, and the judge accepts it. Without the namesake condition these were
                    ~60% true by hand review, mostly an "X SARL" record whose own S1 "X SAS" sits elsewhere.
  restore_llmveto   countries without training labels (Sarvesh's E49, France-safe part): a record the main stack +
                    judge accepts (x/test_q{VBASE}) but the self-training vetoes remove, restored when the judge's
                    margin is >= 4 (US / India validation: vetoed rows at that margin are 95% true, +0.00019) and
                    its house number fits: the record has none, or it sits at the S1's number, adds a word and no
                    decoy-like one (x_ddfix's census). Records at another number are left out (the decoy cluster; on
                    US / India the rejected same-name ones are 0.2-2% true, E41 / E42), as are unedited or
                    word-dropping same-address records (their nD spread is decoy-like).
Records the judge has not scored (stage-2 q outside its band) go to x/sn_rows.parquet: `ROWS=sn_rows python x_llm.py
rows` scores them, then rerun this script.
All lists leave out the pairs the same-address fixes reject.
  python x_recall.py COMB LLM RESTORE REJECT [VBASE]   e.g. _fin _v10p restore_dd reject_dd _v10plw
      COMB: x/test_q{COMB} (x_final.py SAVE_Q); LLM: x/llm_test{LLM}; RESTORE / REJECT: x_ddfix.py's lists
      -> x/restore_empty.parquet, x/restore_nafr.parquet, x/restore_samename.parquet, x/restore_nacore.parquet,
         x/restore_llmveto.parquet (with VBASE: the unlabelled countries' main stack + judge, before the vetoes)
         for x_final.py RESTORE=...
  python x_recall.py val [VQ]                  the empty-S1 rule on US / India validation (x/llm_val{VQ}), with labels"""
import glob, sys, numpy as np, pandas as pd
from common import WORK, load, unlabelled
from match import normed
from x_anatomy import first

XD = f"{WORK}/x"
T, TE = 0.70, 0.40
SHIFT = (1, 13)   # the decoys' house-number shift above the S1's number
LV = 4            # judge margin that overrules a self-training veto


def lists(comb, llm, restore, reject, vbase=None):
    d = pd.read_parquet(f"{XD}/test_q{comb}.parquet")
    key = pd.MultiIndex.from_arrays([d.rid.values, d.sid.values])
    rj = pd.MultiIndex.from_frame(pd.read_parquet(f"{XD}/{reject}.parquet")[["rid", "sid"]])
    for f, val in ((restore, 1.0), (reject, 0.0)):
        r = pd.read_parquet(f"{XD}/{f}.parquet")
        d.loc[key.isin(pd.MultiIndex.from_frame(r[["rid", "sid"]])), "q"] = val
    L = pd.read_parquet(f"{XD}/llm_test{llm}.parquet", columns=["rid", "sid", "llm"])
    m = pd.Series(L.llm.values, index=pd.MultiIndex.from_frame(L[["rid", "sid"]])).reindex(key).values
    unl = d.c.isin(unlabelled()).values
    acc = d.q.values >= T
    ct1 = load("test", 1).country.values
    has = np.bincount(d.sid.values[acc], minlength=len(ct1)) > 0
    ok = ~has[d.sid.values] & (d.q.values >= TE) & ~key.isin(rj) & ~(unl & (m <= 0))
    b = d[ok].sort_values("q", ascending=False).drop_duplicates("sid")
    b[["rid", "sid"]].to_parquet(f"{XD}/restore_empty.parquet")
    print("restore_empty", len(b), b.c.value_counts().to_dict(), flush=True)
    t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
    noaddr = t2.na.values[d.rid.values] == ""
    nn = pd.DataFrame({"c": ct1, "nn": t1.nn.values}).value_counts()
    twin = nn.reindex(pd.MultiIndex.from_arrays([d.c.values, t2.nn.values[d.rid.values]])).fillna(0).values >= 2
    same = t2.nn.values[d.rid.values] == t1.nn.values[d.sid.values]
    sel = unl & ~acc & noaddr & ~twin & same & (m > 0) & ~key.isin(rj)
    nafr = sel
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_nafr.parquet")
    print("restore_nafr", int(sel.sum()), d[sel].c.value_counts().to_dict(), flush=True)
    # restore_samename: the exact name at another house number of the record's own (namesake S1s allowed: an
    # address-bearing record's best S1 is its own when it is a true match)
    cand = unl & ~acc & ~noaddr & same & ~key.isin(rj)
    ss = np.flatnonzero(np.isin(d.sid.values, d.sid.values[cand]))          # every row of those S1s
    a = np.array([first(v) for v in t1.num.values[d.sid.values[ss]]])
    n = np.array([first(v) for v in t2.num.values[d.rid.values[ss]]])
    at = pd.Series(n).groupby([d.sid.values[ss], n]).transform("size").values
    sh = n - a
    sn = np.zeros(len(d), bool)
    sn[ss[(a >= 0) & (n >= 0) & (sh != 0) & (at == 1) & ~((sh >= SHIFT[0]) & (sh <= SHIFT[1]))]] = True
    sn &= cand
    # restore_nacore: no address, the same core name as its S1 (full name differs), no other S1 with that core name
    cnc = pd.DataFrame({"c": ct1, "cn": t1.cn.values}).value_counts()
    ctwin = cnc.reindex(pd.MultiIndex.from_arrays([d.c.values, t2.cn.values[d.rid.values]])).fillna(0).values >= 2
    core = t2.cn.values[d.rid.values] == t1.cn.values[d.sid.values]
    nc = unl & ~acc & noaddr & ~same & core & ~ctwin & ~key.isin(rj)
    ms = m.copy()
    for f in sorted(glob.glob(f"{XD}/llm_sn_rows*.parquet")):
        e = pd.read_parquet(f)
        e2 = pd.Series(e.llm.values, index=pd.MultiIndex.from_frame(e[["rid", "sid"]])).reindex(key).values
        ms = np.where(np.isfinite(ms), ms, e2)
    need = (sn | nc) & ~np.isfinite(ms)
    d[need][["rid", "sid"]].reset_index(drop=True).to_parquet(f"{XD}/sn_rows.parquet")
    sel = sn & (ms > 0)
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_samename.parquet")
    print(f"restore_samename {int(sel.sum())} {d[sel].c.value_counts().to_dict()} (candidates {int(sn.sum())}, judge "
          f"yes {int(sel.sum())} / no {int((sn & (ms <= 0)).sum())}, not scored {int((sn & ~np.isfinite(ms)).sum())}; "
          f"{int(need.sum())} rows to score -> x/sn_rows.parquet)",
          flush=True)
    sel = nc & (ms > 0)
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_nacore.parquet")
    print(f"restore_nacore {int(sel.sum())} {d[sel].c.value_counts().to_dict()} (candidates {int(nc.sum())}, judge "
          f"yes {int(sel.sum())} / no {int((nc & (ms <= 0)).sum())}, not scored {int((nc & ~np.isfinite(ms)).sum())})",
          flush=True)
    if vbase:
        llmveto(d, key, unl, acc, rj, m, t1, t2, key.isin(pd.MultiIndex.from_frame(b[["rid", "sid"]])) | nafr, vbase)


def llmveto(d, key, unl, acc, rj, m, t1, t2, placed, vbase):
    from x_anatomy import positions
    from x_ddfix import census, words
    b0 = pd.read_parquet(f"{XD}/test_q{vbase}.parquet", columns=["rid", "sid", "q"])
    qb = pd.Series(b0.q.values, index=pd.MultiIndex.from_frame(b0[["rid", "sid"]])).reindex(key).values
    vet = unl & ~acc & (qb >= T) & ~key.isin(rj) & ~placed & (m >= LV)
    keep = np.zeros(len(d), bool)
    for c in sorted(set(d.c.values[vet])):
        x = census(positions(d[d.c.values == c].assign(i=np.flatnonzero(d.c.values == c)), t1, t2))
        w = words(x, 300)
        dec = set(w[w.tv >= 0.20].index)
        aw = x["add"].str.split().apply(set).values
        ok = (x.b.values < 0) | ((x.pos.values == "SAME") & (x["add"].values != "") & np.array([not (a & dec) for a in aw]))
        keep[x.i.values[ok]] = True
    sel = vet & keep
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_llmveto.parquet")
    print(f"restore_llmveto {int(sel.sum())} {d[sel].c.value_counts().to_dict()} (vetoed with judge margin >= {LV}: "
          f"{int(vet.sum())})", flush=True)


def val(vq):
    """the empty-S1 rule on US / India validation: stage 2 + the judge's blend (cross-fitted over two halves of the
    S1s, as x_llm.stack_check), then every S1 with no accepted row takes its best row if q >= t"""
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
    for c in sorted(set(cty[ents])) + ["all"]:
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
        lists(*sys.argv[1:6])
