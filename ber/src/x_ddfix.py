"""Same-address fixes for the countries without training labels, built directly from label-free statistics of their
own test records: no hand-written words and no country names (the lists of v10_fr3_llm_dd, leaderboard 0.990282).

Why. Each S1 gets a roughly fixed number of decoys, normally at a shifted house number. On labelled US / India
validation, edited records at the S1's own number are 39% / 31% decoys when the S1 has no record at another number
(nD = 0) and 1-2% / 6-16% otherwise: a decoy placed at the S1's address leaves the shifted cluster short. The models'
word priors come from US / India text and do not carry over to a country without labels (a suffix that marks a decoy
in the US is true-match noise there), so for such a country the same-address decisions follow this statistic. Where
labels exist the models are already right and the same fixes would hurt (`python x_ddfix.py check`), so they are only
applied to the countries without labels.

For each test country without a labelled training record (common.unlabelled):
  1. rows: every record's best S1 and its current decision (x/test_q{COMB}: the stage-2 q with the self-training
     vetoes, accepted at q >= 0.70)
  2. x_anatomy.positions: SAME (the S1's first house number), D (the S1's shifted decoy cluster), OTHER, NONE, and the
     words a record adds to its S1's core name; nD = the S1's records at D or OTHER, capped at 4
  3. per word added at SAME in >= MIN records: TV = total variation distance between the nD spread of those records
     and that of the unedited SAME records. true-like: TV <= TV_TRUE and SAME / D >= 0.1 (not a shifted-decoy word);
     decoy-like: TV >= TV_DEC. The words fall in separated groups (true-match noise <= 0.10, pure shifted-decoy words
     0.15-0.19 with SAME / D < 0.05, swapped type words >= 0.22), so the lists hardly move with the thresholds.
  4. restore: rejected records that the structural rule accepts (rule_fr.py -> x/test_q_rule{BASE}) and that add
     only true-like words (or none); reject: accepted SAME records that add a decoy-like word
-> x/restore_dd.parquet, x/reject_dd.parquet for x_final.py RESTORE=restore_dd REJECT=reject_dd
  python x_ddfix.py BASE COMB [TV_TRUE TV_DEC MIN]    e.g. _v10plw _v10fr3l (defaults 0.10 0.20 300)
  python x_ddfix.py check [VQ]                        the same fixes on US / India validation (x/val_q{VQ}, default
                                                      _v10pw) with labels: what they would do where labels exist"""
import sys, numpy as np, pandas as pd
from common import WORK, load, unlabelled
from match import normed
from x_anatomy import positions

XD = f"{WORK}/x"
T = 0.70


def census(d):
    """nD: the S1's records at another house number (D or OTHER), capped at 4"""
    nD = d.pos.isin(["D", "OTHER"]).groupby(d.sid).sum()
    return d.assign(nD=nD.reindex(d.sid.values).values.clip(max=4), acc=d.q.values >= T)


def words(d, mn):
    """per word added at SAME in >= mn records: records at SAME and at D, and the TV of its nD spread to the unedited
    SAME records"""
    s = d[d.pos == "SAME"]
    ref = s[s["add"] == ""].nD.value_counts(normalize=True).reindex(range(5), fill_value=0).values
    e = s[s["add"] != ""].assign(w=lambda x: x["add"].str.split()).explode("w")
    dw = d[d.pos == "D"].assign(w=lambda x: x["add"].str.split()).explode("w").w.value_counts()
    rows = []
    for w, g in e.groupby("w"):
        if len(g) >= mn:
            dist = g.nD.value_counts(normalize=True).reindex(range(5), fill_value=0).values
            rows.append(dict(w=w, same=len(g), D=int(dw.get(w, 0)), tv=float(np.abs(dist - ref).sum() / 2)))
    t = pd.DataFrame(rows).set_index("w")
    t["ratio"] = t.same / t.D.clip(lower=1)
    return t


def fixes(t, cand, tv_true=0.10, tv_dec=0.20, mn=300):
    """t: one country's rows after positions + census; cand: structural-rule mask. -> restore, reject masks, lists"""
    fw = words(t, mn)
    tru = set(fw[(fw.tv <= tv_true) & (fw.ratio >= 0.1)].index)
    dec = set(fw[fw.tv >= tv_dec].index)
    aw = t["add"].str.split().apply(set).values
    restore = cand & ~t.acc.values & np.array([a <= tru for a in aw])
    reject = (t.pos.values == "SAME") & t.acc.values & np.array([bool(a & dec) for a in aw])
    return restore, reject, tru, dec


def check(vq):
    """the same fixes on each labelled country's validation rows, lists from its own records: labelled F0.5 before /
    after, and the true share of the rows each fix would change"""
    from harness import truth_arrays, wscore
    from rule_fr import rule
    s1, other, ts, s1f, rf = truth_arrays()
    Tn = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]
    n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
    cty = load("train", 1).country.values
    v = pd.read_parquet(f"{XD}/val_q{vq}.parquet")
    v = v.assign(c=cty[v.sid.values])
    base = v.q.values >= T
    for c in sorted(set(v.c)):
        m = np.flatnonzero(v.c.values == c)
        t = census(positions(v.iloc[m].reset_index(drop=True), n1, n2))
        res, rej, tru, dec = fixes(t, rule("train", t[["rid", "sid", "c"]]))
        e = ents[cty[ents] == c]
        f0 = wscore(v.sid.values, base, v.y.values, v.wt.values, Tn, e)
        print(f"\n== {c} validation: true-like {len(tru)}, decoy-like {len(dec)} words", flush=True)
        for nm, k, val in (("restore", res, True), ("reject", rej, False)):
            a = base.copy(); a[m[k]] = val
            f1 = wscore(v.sid.values, a, v.y.values, v.wt.values, Tn, e)
            print(f"  {nm}: {int(k.sum())} rows, true share {t.y.values[k].mean() if k.any() else float('nan'):.4f}, "
                  f"F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else "_v10pw")
        sys.exit()
    BASE, COMB = sys.argv[1], sys.argv[2]
    tv_true, tv_dec = (float(x) for x in (sys.argv[3:5] if len(sys.argv) > 4 else (0.10, 0.20)))
    mn = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
    comb = pd.read_parquet(f"{XD}/test_q{COMB}.parquet")
    r = pd.read_parquet(f"{XD}/test_q_rule{BASE}.parquet")
    ok = pd.MultiIndex.from_frame(r[r.q.values > 0][["rid", "sid"]])
    R, J = [], []
    for c in unlabelled():
        t = census(positions(comb[comb.c.values == c].reset_index(drop=True), t1, t2))
        cand = pd.MultiIndex.from_arrays([t.rid.values, t.sid.values]).isin(ok)
        res, rej, tru, dec = fixes(t, cand, tv_true, tv_dec, mn)
        print(f"{c}: rows {len(t)}, accepted {int(t.acc.sum())}; structural-rule records not accepted {int((cand & ~t.acc.values).sum())}\n"
              f"  true-like ({len(tru)}): {sorted(tru)}\n  decoy-like ({len(dec)}): {sorted(dec)}\n"
              f"  restore {int(res.sum())}, reject {int(rej.sum())}", flush=True)
        R.append(t[res][["rid", "sid"]]); J.append(t[rej][["rid", "sid"]])
    pd.concat(R, ignore_index=True).to_parquet(f"{XD}/restore_dd.parquet")
    pd.concat(J, ignore_index=True).to_parquet(f"{XD}/reject_dd.parquet")
    print("wrote", f"{XD}/restore_dd.parquet", f"{XD}/reject_dd.parquet", flush=True)
