"""E38: richer judge blend on US / India validation. Base = x_llmstack's blend: for unsure rows (0.01 < q < 0.99) with
an LLM score, a logistic regression of the label on [logit q, LLM margin] (weighted) replaces q. Here: + record-type
features (no address, #S1s sharing the record's core name, #S1s sharing the S1's, invented name, exact core-name match,
first house numbers differ), as LR and as a small LightGBM. Cross-fitted over S1 halves (sid % 2), as x_recall.val.
F0.5 at 0.70, plain and with the empty-S1 rule (t 0.4). Also: France name-only records whose name matches no S1
exactly (E37 'typo' type): examples with their best S1, for hand review."""
import os, numpy as np, pandas as pd, lightgbm as lgb
os.environ.setdefault("DFOLD", "1")
from sklearn.linear_model import LogisticRegression
from common import load
from harness import truth_arrays, wscore
from match import normed
W_ = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
c2 = pd.concat([load("train", 2), load("train", 3)], ignore_index=True).country.values
cnt = pd.Series(n1.cn.values + "|" + s1.country.values).value_counts()
amb_r = cnt.reindex(n2.cn.values + "|" + c2).fillna(0).values; amb_s = cnt.reindex(n1.cn.values + "|" + s1.country.values).values
vocab = set(" ".join(n1.cn.values).split())
inv = np.array([bool(x) and all(t not in vocab for t in x.split()) for x in n2.cn.values])
first = lambda a: np.array([x.split()[0] if x else "" for x in a])
f2, f1 = first(n2.num.values), first(n1.num.values)
v = pd.read_parquet(f"{W_}/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"])
L = pd.read_parquet(f"{W_}/llm_val_v10p.parquet", columns=["rid", "sid", "llm"]); L = L[np.isfinite(L.llm.values)]
v = v.merge(L, on=["rid", "sid"], how="left")
v.loc[(v.q.values <= 0.01) | (v.q.values >= 0.99), "llm"] = np.nan
r, s = v.rid.values, v.sid.values
X = pd.DataFrame({"lq": np.log(np.clip(v.q.values, 1e-6, 1 - 1e-6) / np.clip(1 - v.q.values, 1e-6, 1)), "llm": v.llm.values,
                  "na_empty": (n2.na.values[r] == "").astype(float), "amb_r": np.log1p(amb_r[r]), "amb_s": np.log1p(amb_s[s]),
                  "inv": inv[r].astype(float), "cn_eq": (n2.cn.values[r] == n1.cn.values[s]).astype(float),
                  "num_diff": ((f2[r] != "") & (f1[s] != "") & (f2[r] != f1[s])).astype(float)})
y, wt, band = v.y.values.astype(bool), v.wt.values, np.isfinite(v.llm.values)
print(f"validation rows {len(v):,}; unsure with LLM {band.sum():,}", flush=True)
def fit_pred(cols, kind):
    q = v.q.values.copy()
    for h in (0, 1):
        tr, te = band & (s % 2 != h), band & (s % 2 == h)
        if kind == "lr":
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X[cols].values[tr], y[tr], sample_weight=wt[tr]); q[te] = m.predict_proba(X[cols].values[te])[:, 1]
        else:
            m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, verbose=-1, seed=0),
                          lgb.Dataset(X[cols].values[tr], y[tr], weight=wt[tr]), 300)
            q[te] = m.predict(X[cols].values[te])
    return q
top_cache = {}
def F(q, e, rescue):
    a = q >= 0.70
    if rescue:
        na = np.bincount(s[a], minlength=n)
        b = pd.DataFrame({"i": np.arange(len(q)), "sid": s, "q": q})[na[s] == 0].sort_values("q", ascending=False).drop_duplicates("sid")
        a = a.copy(); a[b.i.values[b.q.values >= 0.40]] = True
    return wscore(s, a, y, wt, T, e)
extra = ["na_empty", "amb_r", "amb_s", "inv", "cn_eq", "num_diff"]
qs = {"stage 2 only": v.q.values, "LR [lq, llm] (current)": fit_pred(["lq", "llm"], "lr"),
      "LR + record-type": fit_pred(["lq", "llm"] + extra, "lr"), "LGB [lq, llm]": fit_pred(["lq", "llm"], "lgb"),
      "LGB + record-type": fit_pred(["lq", "llm"] + extra, "lgb")}
b0 = qs["LR [lq, llm] (current)"]
for k, q in qs.items():
    print(f"{k:26s} F0.5 {F(q, ents, False):.5f} (vs current {F(q, ents, False) - F(b0, ents, False):+.6f})   "
          f"+ empty-S1 rule {F(q, ents, True):.5f} (vs current {F(q, ents, True) - F(b0, ents, True):+.6f})  "
          f"fold8 {F(q, ents[s1f[ents] == 8], True) - F(b0, ents[s1f[ents] == 8], True):+.6f} fold9 {F(q, ents[s1f[ents] == 9], True) - F(b0, ents[s1f[ents] == 9], True):+.6f}", flush=True)
# France 'typo' name-only examples (test, E16fr3-like chain q from the E10 sandbox)
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
tc2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
tcnt = pd.Series(m1.cn.values + "|" + t1.country.values).value_counts()
ta = tcnt.reindex(m2.cn.values + "|" + tc2.country.values).fillna(0).values
fr = pd.read_parquet("/scratch/scai/mtech/aib262045/amlc_e10_0.005/work/x/test_q_frchain.parquet"); fr = fr[fr.c == "France"]
k = (m2.na.values[fr.rid.values] == "") & (ta[fr.rid.values] == 0)
f = fr[k]
print(f"France name-only records with no exact-name S1: {len(f):,}; best-S1 q histogram (0,.1,.3,.5,.7,1]: "
      f"{np.histogram(f.q.values, [0, .1, .3, .5, .7, 1.01])[0].tolist()}", flush=True)
for band_lo, band_hi in ((0.3, 0.7), (0.1, 0.3)):
    g = f[(f.q >= band_lo) & (f.q < band_hi)]
    print(f"--- France typo name-only, {band_lo} <= q < {band_hi}: {len(g):,}", flush=True)
    for i in np.random.default_rng(0).choice(len(g), min(12, len(g)), replace=False):
        rr, ss = g.rid.values[i], g.sid.values[i]
        print(f"  rec [{tc2.business_name.values[rr]}]  ->  S1 [{t1.business_name.values[ss]}] [{t1.business_address.values[ss]}]  q={g.q.values[i]:.3f}", flush=True)
