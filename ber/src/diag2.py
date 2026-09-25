import numpy as np, pandas as pd, json
from scipy.optimize import nnls
from common import load, s1_fold, f05, WORK
from match import assign
s1 = load("train", 1); other = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
gt = load("train", "ground_truth")
pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m").query("m != \"\"")
rid_of = pd.Series(np.arange(len(other)), index=other.entity_id)
true_sid = np.full(len(other), -1, np.int64)
true_sid[rid_of.loc[pairs.m].values] = pd.Series(np.arange(len(s1)), index=s1.entity_id).loc[pairs.source1_entity_id].values
s1f = s1_fold(s1.entity_id.tolist()); matched = true_sid >= 0
rf = s1_fold(other.entity_id.tolist()); rf[matched] = s1f[true_sid[matched]]
oc = other.country.values

# ---------- 1. what are the extra test distractors? 3-component mixture of best-cosine histograms
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "score", "rank"])
top = c[c["rank"] == 0].set_index("rid").score
is_true = true_sid[c.rid.values] == c.sid.values
alt = c[~is_true].groupby("rid").score.max()          # best score excluding the true S1 = "if my S1 were removed"
ct = pd.read_parquet(f"{WORK}/cand_test.parquet", columns=["rid", "score", "rank"])
ttop = ct[ct["rank"] == 0].set_index("rid").score
tother_c = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values
bins = np.linspace(0.3, 1.0, 71)
def h(x): v, _ = np.histogram(np.clip(x, 0.3, 0.999), bins); return v / v.sum()
print("== mixture fit of test best-cosine: matched / natural distractor / removed-entity record")
for cn in ["US", "India"]:
    clean_m = matched & (rf >= 4) & (oc == cn)
    hm = h(top.loc[np.where(clean_m)[0]].values)
    hd = h(top.loc[np.where(~matched & (oc == cn))[0]].values)
    hr = h(alt.reindex(np.where(clean_m)[0]).dropna().values)
    ht = h(ttop.loc[np.where(tother_c == cn)[0]].values)
    for name, A in [("2-comp m+d", np.c_[hm, hd]), ("2-comp m+r", np.c_[hm, hr]), ("3-comp", np.c_[hm, hd, hr])]:
        w, res = nnls(A, ht); w = w / w.sum()
        print(f"  {cn:6s} {name:11s} weights {np.round(w, 3)}  residual {res:.4f}")
    # sanity: train itself
    htr = h(top.loc[np.where((oc == cn) & (~matched | (rf >= 4)))[0]].values)
    w, res = nnls(np.c_[hm, hd, hr], htr); print(f"  {cn:6s} train-check 3-comp {np.round(w / w.sum(), 3)} residual {res:.4f}")

# ---------- 2. loss decomposition, test-like, thr 0.5 (clean records only, as in eval_full)
k = pd.read_parquet(f"{WORK}/pred_all_train.parquet")
best = assign(k[["rid", "sid"]], k.p.values, 0.0)
best["ts"] = true_sid[best.rid.values]
f9 = s1f == 9
truth_n = np.bincount(true_sid[matched], minlength=len(s1))
rng = np.random.default_rng(0)
n_match = matched.sum(); pool = np.where(~matched & np.isin(rf, [0, 1, 2, 3, 9]))[0]
draw = rng.choice(pool, int(0.39 / 0.61 * n_match), replace=True); w = np.bincount(draw, minlength=len(other))
thr = 0.5
acc = best[best.p >= thr]
vm = acc[(acc.ts >= 0) & (rf[acc.rid.values] == 9)]            # fold-9 matched records
tp = np.bincount(vm.sid[vm.ts == vm.sid], minlength=len(s1))
fpw = np.bincount(vm.sid[vm.ts != vm.sid], minlength=len(s1))
ad = acc[acc.ts < 0]; fpd = np.bincount(ad.sid, weights=w[ad.rid.values], minlength=len(s1))
idx = np.where(f9)[0]
T, TP, FPD, FPW = truth_n[idx], tp[idx], fpd[idx], fpw[idx]
P = TP + FPD + FPW
with np.errstate(divide="ignore", invalid="ignore"):
    pr = np.where(P > 0, TP / P, 0); rc = np.where(T > 0, TP / T, 0)
    f = np.where(T == 0, (P == 0).astype(float), np.where(TP > 0, 1.25 * pr * rc / (0.25 * pr + rc), 0))
loss = 1 - f
print(f"\n== loss decomposition (test-like, thr {thr}): F0.5 {f.mean():.5f}, total loss {loss.mean():.5f}")
cats = {"singleton + any FP": (T == 0) & (P > 0), "has distractor FP": (T > 0) & (FPD > 0),
        "has wrong-S1 FP (no distr.)": (T > 0) & (FPD == 0) & (FPW > 0), "FN only": (T > 0) & (FPD == 0) & (FPW == 0) & (TP < T)}
for n, m in cats.items(): print(f"  {n:28s} entities {m.mean():6.2%}  loss {loss[m].sum() / len(f):.5f}")
# distractor FPs: clustered on the same S1?
nd = np.bincount(ad.sid, minlength=len(s1))[idx]
print("  distractors accepted per fold-9 S1 (unweighted, clean pool):", dict(pd.Series(nd).value_counts().sort_index().head(6)))
# ---------- 3. France vs others on test (v3 output)
t1 = load("test", 1); tc = dict(zip(t1.entity_id, t1.country))
for tag, f_ in [("v3 thr0.5", f"{WORK}/../output_v3/matching_results.tsv")]:
    mr = pd.read_csv(f_, sep="\t", dtype=str, keep_default_na=False)
    n = mr.matched_entity_ids.map(lambda s: 0 if s == "" else s.count(",") + 1); cc = mr.source1_entity_id.map(tc)
    nrec = pd.Series(tother_c).value_counts()
    g = pd.DataFrame({"n": n, "c": cc}).groupby("c").n.agg(entities="size", empty=lambda x: (x == 0).mean(), per_s1="mean", assigned="sum")
    g["records"] = nrec; g["assigned_share"] = g.assigned / g.records
    print(f"\n== TEST {tag}\n", g.round(4).to_string())
tt = ct[ct["rank"] == 0]; print("test best-cosine median by country:", pd.Series(tt.score.values).groupby(tother_c[tt.rid.values]).median().round(3).to_dict())
# train reference at thr 0.5 (fold-9 matched + natural distractors, unweighted)
