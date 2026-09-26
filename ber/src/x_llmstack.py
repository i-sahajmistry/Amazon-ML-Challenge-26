"""Blend the LLM judge (x_llm.py) into stage 2 for the unsure records: a logistic regression of the label on
[logit q, LLM margin], fitted on the validation rows that have an LLM score (weighted like the validation), replaces q
of every unsure test row that has one. The LLM scored the unsure rows of stage-2 model QT (x/llm_val{QT},
x/llm_test{QT}); another stage-2 model QB reuses those scores for the same (record, S1) pairs. Writes
x/test_q{QB}lw.parquet (+ candidate link) for x_final.py, after a cross-fitted validation check.
  python x_llmstack.py [QT] [QB]      (default _v9p _v9p)"""
import os, sys, numpy as np, pandas as pd
from x_llm import fit_stack, apply_stack, stack_check, XD, LO, HI
from harness import truth_arrays

QT = sys.argv[1] if len(sys.argv) > 1 else "_v9p"
QB = sys.argv[2] if len(sys.argv) > 2 else QT
LT = os.environ.get("LLM_TAG", "")              # LLM variant (x_llm.py LLM_TAG), e.g. _fr: continued on France
OUT = os.environ.get("OUT", f"{QB}l{LT}")       # -> x/test_q{OUT}w.parquet


def attach(base, src):
    """LLM margin of the same (record, S1) pair where the LLM scored it, for the base model's unsure rows."""
    s = src[np.isfinite(src.llm.values)][["rid", "sid", "llm"]]
    d = base.merge(s, on=["rid", "sid"], how="left")
    d.loc[(d.q.values <= LO) | (d.q.values >= HI), "llm"] = np.nan
    return d


v = attach(pd.read_parquet(f"{XD}/val_q{QB}w.parquet"), pd.read_parquet(f"{XD}/llm_val{QT}{LT}.parquet"))
band = (v.q.values > LO) & (v.q.values < HI)
print(f"validation unsure rows {band.sum()}, with an LLM score {np.isfinite(v.llm.values).sum()}", flush=True)
s1, other, ts, s1f, rf = truth_arrays()
stack_check(v, ts, s1f)
lr = fit_stack(v[np.isfinite(v.llm.values)])
print("stack coefficients [logit q, llm]", lr.coef_.round(3), "intercept", lr.intercept_.round(3), flush=True)
d = attach(pd.read_parquet(f"{XD}/test_q{QB}w.parquet"), pd.read_parquet(f"{XD}/llm_test{QT}{LT}.parquet"))
m = np.isfinite(d.llm.values)
q = d.q.values.copy()
q[m] = apply_stack(lr, d[m])
for c in sorted(d.c.unique()):
    k = d.c.values == c
    u = k & (d.q.values > LO) & (d.q.values < HI)
    print(f"{c}: unsure rows {u.sum()}, re-scored {(m & k).sum()}; accepted at 0.70 before {(d.q.values[k] >= 0.7).sum()}, "
          f"after {(q[k] >= 0.7).sum()}", flush=True)
out = f"{XD}/test_q{OUT}w.parquet"
d.assign(q=q)[["rid", "sid", "q", "c"]].to_parquet(out)
link = f"{XD}/p5_test{OUT}w.parquet"
if not os.path.exists(link):
    os.symlink(os.path.basename(os.path.realpath(f"{XD}/p5_test{QB}w.parquet")), link)
print("wrote", out, flush=True)
