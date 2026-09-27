"""E49: LLM arbitration of veto disagreements. The France chain = min(main+judge, a1, a2, a3) (same S1). On US / India
validation (labels): main+judge q = x_llmstack blend of val_q_v10pw with llm_val_v10p (cross-fitted halves, pipeline
weights); chain = min with val_q_a1/a2/a3 (variant). Rows the main accepts but the chain vetoes: restore those whose LLM
margin >= m (or: whose self-trained stacks are split, 2-of-3 / 1-of-3 accept). F0.5 (share 0.49 as the pipeline and
0.40 test-like), cross-fitted m. Then France test: how many vetoed France records each rule restores (-> add lists)."""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from sklearn.linear_model import LogisticRegression
from common import load
from harness import truth_arrays, wscore
OUT = sys.argv[1]
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
SX = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"; V = "/scratch/scai/mtech/aib262144/amlc_variant/work/x"
v = pd.read_parquet(f"{SX}/val_q_v10pw.parquet", columns=["rid", "sid", "q", "y", "wt"])
L = pd.read_parquet(f"{SX}/llm_val_v10p.parquet", columns=["rid", "sid", "llm"]); L = L[np.isfinite(L.llm.values)]
v = v.merge(L, on=["rid", "sid"], how="left")
sid, y, wt0, q0 = v.sid.values, v.y.values.astype(bool), v.wt.values, v.q.values
llm = v.llm.values.copy(); llm[(q0 <= 0.01) | (q0 >= 0.99)] = np.nan; band = np.isfinite(llm)
X = np.c_[np.log(np.clip(q0, 1e-6, 1 - 1e-6) / np.clip(1 - q0, 1e-6, 1)), np.nan_to_num(llm)]
qm = q0.copy()
for h in (0, 1):
    tr, te = band & (sid % 2 != h), band & (sid % 2 == h)
    qm[te] = LogisticRegression(C=1.0).fit(X[tr], y[tr], sample_weight=wt0[tr]).predict_proba(X[te])[:, 1]
qa = []
for t in ("a1", "a2", "a3"):
    e = pd.read_parquet(f"{V}/val_q_{t}pw.parquet", columns=["rid", "sid", "q"])
    m = v[["rid", "sid"]].merge(e, on="rid", how="left", suffixes=("", "_b"))
    qa.append(np.where(m.sid.values == m.sid_b.values, m.q.values, 0.0))
qa = np.array(qa); qc = np.minimum(qm, qa.min(axis=0))
am, ac = qm >= 0.70, qc >= 0.70
vet = am & ~ac; nacc = (qa >= 0.70).sum(axis=0)
isd = ts[v.rid.values] < 0
print(f"US/India val: main+judge accepts {am.sum():,}, chain accepts {ac.sum():,}, vetoed {vet.sum():,} (true {y[vet].mean():.3f}); "
      f"vetoed with an LLM score {(vet & band).sum():,} (true {y[vet & band].mean():.3f})", flush=True)
rules = {f"llm>={m_}": vet & band & (llm >= m_) for m_ in (-1, 0, 1, 2, 3, 4, 5, 6)}
rules.update({f"{k}of3 self-trained accept": vet & (nacc >= k) for k in (1, 2)})
rules.update({f"llm>={m_} & 1of3": vet & band & (llm >= m_) & (nacc >= 1) for m_ in (0, 2, 4)})
for share in (0.49, 0.40):
    W = share / (1 - share) * (~isd).sum() / isd.sum(); wt = np.where(isd, W, 1.0)
    F = lambda a, e: wscore(sid, a, y, wt, T, e)
    print(f"== metric share {share}: main {F(am, ents):.5f}  chain {F(ac, ents):.5f}", flush=True)
    res = {}
    for k, r in rules.items():
        a = ac | r; res[k] = {f: F(a, ents[s1f[ents] == f]) - F(ac, ents[s1f[ents] == f]) for f in (8, 9)}
        print(f"   restore {k:28s}: rows {r.sum():>6,} true {y[r].mean() if r.any() else 0:.3f}  gain vs chain {F(a, ents) - F(ac, ents):+.6f}", flush=True)
    g = []
    for fit, ev in ((8, 9), (9, 8)):
        b = max(res, key=lambda k: res[k][fit]); g.append(res[b][ev]); print(f"   fit {fit}: best '{b}' -> fold {ev} {res[b][ev]:+.6f}", flush=True)
    print(f"   cross-fitted gain vs chain: {np.mean(g):+.6f}", flush=True)
# France test
t1 = load("test", 1); o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
d = pd.read_parquet(f"{V}/test_q_v10plw.parquet"); d = d[d.c == "France"].reset_index(drop=True)
Lt = pd.read_parquet(f"{SX}/llm_test_v10p.parquet", columns=["rid", "sid", "llm"]); Lt = Lt[np.isfinite(Lt.llm.values)]
d = d.merge(Lt, on=["rid", "sid"], how="left")
qa_t = []
for t in ("a1", "a2", "a3"):
    e = pd.read_parquet(f"{V}/test_q_{t}pw.parquet", columns=["rid", "sid", "q"])
    m = d[["rid", "sid"]].merge(e, on="rid", how="left", suffixes=("", "_b"))
    qa_t.append(np.where(m.sid.values == m.sid_b.values, m.q.values, 0.0))
qa_t = np.array(qa_t); qct = np.minimum(d.q.values, qa_t.min(axis=0)); nacc_t = (qa_t >= 0.70).sum(axis=0)
rj = pd.read_parquet(f"{V}/reject_fr_dd.parquet"); rjk = set(zip(rj.rid.values, rj.sid.values))
notrj = np.array([(a, b) not in rjk for a, b in zip(d.rid.values, d.sid.values)])
vt = (d.q.values >= 0.70) & (qct < 0.70) & notrj; lt = d.llm.values
print(f"France test: main+judge accepts {(d.q.values >= 0.7).sum():,}, vetoed (not in reject_fr_dd) {vt.sum():,}, with LLM score {(vt & np.isfinite(lt)).sum():,}", flush=True)
for m_ in (0, 2, 4, 6):
    sel = vt & np.isfinite(lt) & (lt >= m_)
    pd.DataFrame({"s1_id": t1.entity_id.values[d.sid.values[sel]], "rec_id": o.entity_id.values[d.rid.values[sel]]}).to_parquet(f"{OUT}/add_llmveto{m_}.parquet")
    print(f"   France restore llm>={m_}: {sel.sum():,}", flush=True)
for k in (1, 2):
    sel = vt & (nacc_t >= k)
    pd.DataFrame({"s1_id": t1.entity_id.values[d.sid.values[sel]], "rec_id": o.entity_id.values[d.rid.values[sel]]}).to_parquet(f"{OUT}/add_vote{k}of3.parquet")
    print(f"   France restore {k}of3 self-trained accept: {sel.sum():,}", flush=True)
