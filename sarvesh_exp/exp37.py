"""E50: E49 + house-number compatibility. Restore a vetoed record only if its first house number equals the S1's
(integer-aware) or the record has no number. (1) US / India validation: vetoed rows split by number compatibility and
LLM margin (true rate, F0.5 gain vs chain). (2) France: filter the E49 add lists -> add_llmveto{m}n."""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from sklearn.linear_model import LogisticRegression
from common import load
from harness import truth_arrays, wscore
from match import normed
OUT = sys.argv[1]
def first_int(a):
    out = np.full(len(a), -1, np.int64)
    for i, x in enumerate(a):
        t = x.split()[0] if x else ""
        if t.isdigit() and len(t) < 12: out[i] = int(t)
    return out
def compat(M1, M2, r, s):
    a, b = first_int(M2.num.values[r]), first_int(M1.num.values[s])
    return (a < 0) | (a == b)
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
SX = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"; V = "/scratch/scai/mtech/aib262144/amlc_variant/work/x"
v = pd.read_parquet(f"{SX}/val_q_v10pw.parquet", columns=["rid", "sid", "q", "y", "wt"])
L = pd.read_parquet(f"{SX}/llm_val_v10p.parquet", columns=["rid", "sid", "llm"]); L = L[np.isfinite(L.llm.values)]
v = v.merge(L, on=["rid", "sid"], how="left")
sid, y, wt, q0 = v.sid.values, v.y.values.astype(bool), v.wt.values, v.q.values
llm = v.llm.values.copy(); llm[(q0 <= 0.01) | (q0 >= 0.99)] = np.nan; band = np.isfinite(llm)
X = np.c_[np.log(np.clip(q0, 1e-6, 1 - 1e-6) / np.clip(1 - q0, 1e-6, 1)), np.nan_to_num(llm)]
qm = q0.copy()
for h in (0, 1):
    tr, te = band & (sid % 2 != h), band & (sid % 2 == h)
    qm[te] = LogisticRegression(C=1.0).fit(X[tr], y[tr], sample_weight=wt[tr]).predict_proba(X[te])[:, 1]
qa = []
for t in ("a1", "a2", "a3"):
    e = pd.read_parquet(f"{V}/val_q_{t}pw.parquet", columns=["rid", "sid", "q"])
    m = v[["rid", "sid"]].merge(e, on="rid", how="left", suffixes=("", "_b"))
    qa.append(np.where(m.sid.values == m.sid_b.values, m.q.values, 0.0))
qc = np.minimum(qm, np.array(qa).min(axis=0)); am, ac = qm >= 0.70, qc >= 0.70; vet = am & ~ac
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
cp = compat(n1, n2, v.rid.values, sid)
F = lambda a, e: wscore(sid, a, y, wt, T, e)
print(f"US/India vetoed {vet.sum():,}: number-compatible {(vet & cp).sum():,} (true {y[vet & cp].mean():.3f}), conflicting {(vet & ~cp).sum():,} (true {y[vet & ~cp].mean():.3f})", flush=True)
for m_ in (0, 2, 4, 6):
    for name, r in ((f"llm>={m_}", vet & band & (llm >= m_)), (f"llm>={m_} & number-compatible", vet & band & (llm >= m_) & cp)):
        a = ac | r
        print(f"  restore {name:30s} rows {r.sum():>5,} true {y[r].mean() if r.any() else 0:.3f} gain {F(a, ents) - F(ac, ents):+.6f} "
              f"(fold8 {F(a, ents[s1f[ents] == 8]) - F(ac, ents[s1f[ents] == 8]):+.6f}, fold9 {F(a, ents[s1f[ents] == 9]) - F(ac, ents[s1f[ents] == 9]):+.6f})", flush=True)
t1 = load("test", 1); o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
sid_of = pd.Series(np.arange(len(t1)), index=t1.entity_id.values); rid_of = pd.Series(np.arange(len(o)), index=o.entity_id.values)
for m_ in (2, 4, 6):
    a = pd.read_parquet(f"{OUT}/add_llmveto{m_}_free.parquet")
    k = compat(m1, m2, rid_of.reindex(a.rec_id).values, sid_of.reindex(a.s1_id).values)
    a[k].to_parquet(f"{OUT}/add_llmveto{m_}n.parquet"); print(f"France llm>={m_}: {len(a):,} -> number-compatible {k.sum():,}", flush=True)
