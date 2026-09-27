"""Recall fixes on the final scores (after the vetoes and the same-address fixes), from Mohanish's variant_frB and
E16fr3-sn and Sarvesh's E49.
  restore_empty     an S1 with no accepted record takes its best candidate if q >= 0.40: an S1 left empty scores 0 as
                    soon as it has one true match (US / India validation on the bge + LLM stack: +0.00007, `val`). For
                    the countries without training labels the LLM judge must not say no: the candidates it rejects
                    there are mostly a type word swapped at the S1's own address (the same-building decoy the
                    same-address fixes reject).
The other lists are for the countries without training labels, whose self-training vetoes learned "another address or
no address = reject" (US / India models accept 93-100% of each kind below, where it is 89-100% true):
  restore_nafr      a rejected record with no address whose normalised name equals its S1's and no other S1's of the
                    country, and that the judge accepts.
  restore_nacore    the same with the core name (legal form dropped; the full name differs).
  restore_samename  a record with its S1's exact name at another house number (the only record of the S1 there,
                    outside the decoy shift 1-13 above the S1's number) that the main stack accepts and a self-trained
                    stack vetoes, that the judge accepts, on the S1's street or with no other S1 of that name in the
                    country, and not anchored: none of its candidate S1s sits on its street 1-13 numbers below it. An
                    anchored record is that S1's decoy with a type word swapped that happens to spell another S1's name
                    (US / India validation: 0% true; others 89-91%; with 3+ namesakes on other streets 54-64%).
  restore_vetona    a record the main stack accepts but a self-trained stack vetoes, whose address has no house number,
                    that the judge accepts with a margin >= 2, and whose S1's street words all appear in its address
                    or whose name no other S1 has.
  restore_exact     a record at its S1's exact address (house number and street), the only S1 of the country
                    there, that the main stack accepts and a self-trained stack vetoes, whose name adds no word the
                    S1's name lacks (dropped words, initials, a website): not a type-word swap and not an acronym with a
                    letter changed, the same-building decoys. US / India validation: 98-100% true.
Street words: the first comma part of an address with a digit, minus the words in more than 1% of the country's S1
addresses (street types, articles, city names); "on the street" means every street word matches up to a typo. On US /
India validation the model's own rejections of these kinds are 75-99% wrong records, so restore_samename,
restore_vetona and restore_exact only undo a self-trained veto. All lists leave out the pairs the same-address fixes
reject.
  python x_recall.py COMB LLM RESTORE REJECT [MAIN]   e.g. _fin _v10p restore_dd reject_dd _v10plw
      COMB: x/test_q{COMB} (x_final.py SAVE_Q); LLM: x/llm_test{LLM}; RESTORE / REJECT: x_ddfix.py's lists; MAIN: the
      main stack before the vetoes (samename, vetona, exact); P5 (env, default _v10): candidate pairs x/p5_test{P5}
      -> x/restore_{empty,nafr,nacore,samename,vetona,exact}.parquet for x_final.py RESTORE=...,
         x/reject_decword.parquet for x_final.py REJECT=...
  reject_decword    (a reject list) a record of a country without labels that adds a decoy-like word of that country
                    (x_ddfix's census: TV >= 0.20, the same-building type-word swap) at any position: the same-address
                    fixes reject these only at the S1's house number; the recall lists above and the base stack still
                    accept ~160 elsewhere (no house number, another number, "Internes Club" <- "Internes Union").
Records the judge has not scored (stage-2 q outside its band) go to x/sn_rows.parquet: `ROWS=sn_rows python x_llm.py
rows` scores them, then rerun this script.
  python x_recall.py val [VQ]                  the empty-S1 rule on US / India validation (x/llm_val{VQ}), with labels"""
import glob, os, re, sys, numpy as np, pandas as pd
from collections import Counter
from anyascii import anyascii
from rapidfuzz import fuzz
from common import WORK, load, unlabelled
from match import normed
from x_anatomy import first

XD = f"{WORK}/x"
T, TE = 0.70, 0.40
SHIFT = (1, 13)   # the decoys' house-number shift above their S1's number


def part(a):
    """words of an address's first comma part with a digit (the street)"""
    for p in anyascii(a).lower().split(","):
        if re.search(r"\d", p):
            return re.findall(r"[a-z]{3,}", p)
    return []


def on(u, w):
    """every street word of u appears in w, up to a typo"""
    return bool(u) and all(any(fuzz.ratio(x, y) >= 80 for y in w) for x in u)


def lists(comb, llm, restore, reject, main=None):
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
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_nafr.parquet")
    print("restore_nafr", int(sel.sum()), d[sel].c.value_counts().to_dict(), flush=True)
    decword(d, unl, t1, t2)
    free = unl & ~acc & ~key.isin(rj)
    # accepted by the main stack before the vetoes (MAIN): the kinds below are restored only where the model trained on
    # labels said yes and only a self-trained veto said no (on US / India the model's own rejections of them are 75-99%
    # wrong records)
    mq = np.zeros(len(d))
    if main:
        e = pd.read_parquet(f"{XD}/test_q{main}.parquet", columns=["rid", "sid", "q"]).set_index("rid").reindex(d.rid.values)
        mq = np.where(e.sid.values == d.sid.values, e.q.values, 0.0)
    base = free & (mq >= T)
    # street words and first house numbers
    s1, o = load("test", 1), pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
    common = {}
    for c in unlabelled():
        a1 = s1.business_address.values[ct1 == c]
        df = Counter(w for a in a1 for w in set(re.findall(r"[a-z]{3,}", anyascii(a).lower())))
        common[c] = {w for w, k in df.items() if k > 0.01 * len(a1)}
    c1 = dict(zip(np.arange(len(s1)), ct1))
    sw1 = lambda s: frozenset(w for w in part(s1.business_address.values[s]) if w not in common.get(c1[s], ()))
    sw2 = lambda r, c: frozenset(w for w in part(o.business_address.values[r]) if w not in common.get(c, ()))
    n1 = lambda s: first(t1.num.values[s])
    n2 = lambda r: first(t2.num.values[r])
    # restore_samename: the exact name at another house number, on the S1's street or with no namesake, not anchored
    cand = base & ~noaddr & same
    ss = np.flatnonzero(np.isin(d.sid.values, d.sid.values[cand]))          # every row of those S1s
    a = np.array([n1(s) for s in d.sid.values[ss]])
    n = np.array([n2(r) for r in d.rid.values[ss]])
    at = pd.Series(n).groupby([d.sid.values[ss], n]).transform("size").values
    sh = n - a
    sn = np.zeros(len(d), bool)
    sn[ss[(a >= 0) & (n >= 0) & (sh != 0) & (at == 1) & ~((sh >= SHIFT[0]) & (sh <= SHIFT[1]))]] = True
    sn &= cand
    p5 = pd.read_parquet(f"{XD}/p5_test{os.environ.get('P5', '_v10')}.parquet", columns=["rid", "sid"])
    idx = np.flatnonzero(sn)
    cands = p5[p5.rid.isin(d.rid.values[idx])].groupby("rid").sid.apply(list).to_dict()
    for i in idx:
        r, s, c = d.rid.values[i], d.sid.values[i], d.c.values[i]
        w, k = sw2(r, c), n2(r)
        anchored = any(z != s and on(sw1(z), w) and SHIFT[0] <= k - n1(z) <= SHIFT[1] for z in cands.get(r, ()))
        sn[i] = not anchored and (on(sw1(s), w) or not twin[i])
    # restore_nacore: no address, the same core name as its S1 (full name differs), no other S1 with that core name
    cnc = pd.DataFrame({"c": ct1, "cn": t1.cn.values}).value_counts()
    ctwin = cnc.reindex(pd.MultiIndex.from_arrays([d.c.values, t2.cn.values[d.rid.values]])).fillna(0).values >= 2
    core = t2.cn.values[d.rid.values] == t1.cn.values[d.sid.values]
    nc = free & noaddr & ~same & core & ~ctwin
    ms = m.copy()
    for f in sorted(glob.glob(f"{XD}/llm_sn_rows*.parquet")):
        e = pd.read_parquet(f)
        e2 = pd.Series(e.llm.values, index=pd.MultiIndex.from_frame(e[["rid", "sid"]])).reindex(key).values
        ms = np.where(np.isfinite(ms), ms, e2)
    need = (sn | nc) & ~np.isfinite(ms)
    d[need][["rid", "sid"]].reset_index(drop=True).to_parquet(f"{XD}/sn_rows.parquet")
    for name, k in (("restore_samename", sn), ("restore_nacore", nc)):
        sel = k & (ms > 0)
        d[sel][["rid", "sid"]].to_parquet(f"{XD}/{name}.parquet")
        print(f"{name} {int(sel.sum())} {d[sel].c.value_counts().to_dict()} (candidates {int(k.sum())}, judge no "
              f"{int((k & (ms <= 0)).sum())}, not scored {int((k & ~np.isfinite(ms)).sum())})", flush=True)
    print(f"{int(need.sum())} rows for the judge -> x/sn_rows.parquet", flush=True)
    # restore_vetona: vetoed, no house number, the judge's margin >= 2, on the S1's street or with no namesake
    sel = np.zeros(len(d), bool)
    for i in np.flatnonzero(base & ~noaddr & (m >= 2)):
        r, s = d.rid.values[i], d.sid.values[i]
        if n2(r) < 0:
            sel[i] = on(sw1(s), set(re.findall(r"[a-z]{3,}", anyascii(o.business_address.values[r]).lower()))) or not twin[i]
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_vetona.parquet")
    print("restore_vetona", int(sel.sum()), d[sel].c.value_counts().to_dict(), flush=True)
    # restore_exact: vetoed at the S1's exact address, the only S1 there, the name adds no word the S1's name lacks
    here = Counter((c1[s], n1(s), sw1(s)) for s in np.flatnonzero(np.isin(ct1, unlabelled())))
    sel = np.zeros(len(d), bool)
    for i in np.flatnonzero(free & ~noaddr):
        r, s, c = d.rid.values[i], d.sid.values[i], d.c.values[i]
        k, w = n2(r), sw2(r, c)
        if k < 0 or k != n1(s) or here[(c, k, sw1(s))] != 1 or not (on(sw1(s), w) and on(w, sw1(s))):
            continue
        R, S = t2.cn.values[r].split(), t1.cn.values[s].split()
        ini = len(R) == 1 and 2 <= len(R[0]) <= 5 and any("".join(x[0] for x in S if len(x) >= j).startswith(R[0]) for j in (1, 4))
        nm = anyascii(o.business_name.values[r]).lower().strip()
        web = bool(re.search(r"\.[a-z]{2,4}$|^[#@]", nm)) and fuzz.partial_ratio("".join(S), re.sub(r"[^a-z]", "", nm.split(".")[0])) >= 80
        sel[i] = ini or base[i] and (set(R) <= set(S) or web)
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_exact.parquet")
    print("restore_exact", int(sel.sum()), d[sel].c.value_counts().to_dict(), flush=True)
    # restore_alias: at the S1's exact address, the only S1 there, a made-up name sharing nothing with the S1's (the
    # generator's unrelated-DBA noise), restored when the main stack's q >= 0.80 (vetoed; US / India validation 97-99.8%
    # true), or q >= 0.10 with at most 2 made-up names at the S1 (~90%; below 0.10 they are 0% true)
    cnt = Counter(w for sp in ("train", "test") for x in normed(sp, 1).nn.values for w in set(x.split()))
    gen = {w for w, k in cnt.items() if k >= 200}
    mk = np.zeros(len(d), bool)
    for i in np.flatnonzero(unl & ~noaddr):
        r, s, c = d.rid.values[i], d.sid.values[i], d.c.values[i]
        R, S = t2.nn.values[r].split(), [x for x in t1.nn.values[s].split() if len(x) >= 2]
        new = [x for x in R if len(x) >= 4 and x.isalpha() and x not in cnt]
        if not new or not S or not all(x in new or x in gen or len(x) <= 2 for x in R):
            continue
        k, w = n2(r), sw2(r, c)
        if k < 0 or k != n1(s) or here[(c, k, sw1(s))] != 1 or not (on(sw1(s), w) and on(w, sw1(s))):
            continue
        nm = anyascii(o.business_name.values[r]).lower().strip()
        mk[i] = not (re.search(r"\.(com|fr|in|net|org)\b|^[@#]", nm) or any(fuzz.ratio(x, y) >= 70 for x in R for y in S)
                     or any(len(y) >= 4 and y in "".join(R) for y in S))
    na = pd.Series(mk).groupby(d.sid.values).transform("sum").values
    sel = mk & free & ((mq >= 0.80) | ((mq >= 0.10) & (na <= 2)))
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/restore_alias.parquet")
    print("restore_alias", int(sel.sum()), d[sel].c.value_counts().to_dict(), flush=True)


def decword(d, unl, t1, t2):
    """reject_decword: records of the countries without labels that add a decoy-like word (x_ddfix's census)"""
    from x_anatomy import positions
    from x_ddfix import census, words
    sel = np.zeros(len(d), bool)
    for c in sorted(set(d.c.values[unl])):
        i = np.flatnonzero(d.c.values == c)
        x = census(positions(d.iloc[i], t1, t2))
        w = words(x, int(os.environ.get("DEC_MIN", 100)))
        dec = set(w[w.tv >= 0.20].index)
        sel[i[np.array([bool(set(a.split()) & dec) for a in x["add"].values])]] = True
    d[sel][["rid", "sid"]].to_parquet(f"{XD}/reject_decword{os.environ.get("DEC_TAG", "")}.parquet")
    print("reject_decword", int(sel.sum()), d[sel].c.value_counts().to_dict(), f"(accepted now: {int((sel & (d.q.values >= T)).sum())})",
          flush=True)


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
