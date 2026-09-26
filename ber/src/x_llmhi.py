"""LLM judge on the rows stage 2 never sends to it (v12b). x_llm.py scores rows with 0.01 < q < 0.99 and x_llmstack.py
blends them into stage 2. For a country without training labels stage 2 is not calibrated: on France, 94% of the rows
v10_fr3_llm accepts sit at q >= 0.99, where US / India validation rows are 99.99% true matches, so the judge never sees
them. This script gives those rows the judge's answer; x_final.py LLMVETO=M rejects the rows whose margin
logit(yes) - logit(no) is below M (0: the judge answers "no").
  python x_llmhi.py rows QBASE QCOMB [QV]   # rows of countries without training labels accepted (q >= 0.70) in
                                           # x/test_q{QCOMB} whose stage-2 q in x/test_q{QBASE} is >= 0.99
                                           # -> x/llmhi_test_rows.parquet; with QV also US / India validation rows of
                                           # x/val_q{QV} at q >= 0.99 (40k true + all others) -> x/llmhi_val_rows.parquet
  SHARD=i/n python x_llmhi.py test|val     # score one shard of the rows (one GPU each) -> x/llmhi_{split}_{i}of{n}.parquet
  python x_llmhi.py check [M]              # share of rows each cut rejects (validation true / not true, test) and
                                           # sample test rows the cut M (default 0) rejects
env LLM (base model dir) as x_llm.py; the LoRA adapter is x/llm_lora (x_llm.py train)."""
import glob, os, sys, numpy as np, pandas as pd
from common import WORK, load

XD = f"{WORK}/x"
T, HI = 0.70, 0.99
CUTS = (-3, -2, -1, -0.5, 0, 0.5)


def unlabelled():
    """test countries without a single labelled training record (France here), found from the data"""
    return sorted(set(load("test", 1).country.unique()) - set(load("train", 1).country.unique()))


def rows(qbase, qcomb, qv=None):
    c = pd.read_parquet(f"{XD}/test_q{qcomb}.parquet")
    base = pd.read_parquet(f"{XD}/test_q{qbase}.parquet", columns=["rid", "q"]).set_index("rid").q
    acc = c.c.isin(unlabelled()).values & (c.q.values >= T)
    keep = acc & (base.reindex(c.rid).values >= HI)
    c[keep][["rid", "sid"]].reset_index(drop=True).to_parquet(f"{XD}/llmhi_test_rows.parquet")
    print(f"{unlabelled()}: {int(acc.sum())} accepted rows, {int(keep.sum())} of them at stage-2 q >= {HI} -> "
          f"{XD}/llmhi_test_rows.parquet", flush=True)
    if qv:
        v = pd.read_parquet(f"{XD}/val_q{qv}.parquet")
        v = v[v.q.values >= HI]
        rng = np.random.default_rng(0)
        pick = np.r_[rng.permutation(np.flatnonzero(v.y.values))[:40000], np.flatnonzero(~v.y.values)]
        v.iloc[np.sort(pick)][["rid", "sid", "y", "wt", "q"]].reset_index(drop=True).to_parquet(f"{XD}/llmhi_val_rows.parquet")
        print(f"validation rows at q >= {HI}: {len(v)} ({int((~v.y.values).sum())} not true), {len(pick)} kept", flush=True)


def score(split):
    from x_llm import texts, prompts, model, score as llm_score
    i, n = map(int, os.environ.get("SHARD", "0/1").split("/"))
    r = pd.read_parquet(f"{XD}/llmhi_{split}_rows.parquet").iloc[i::n].copy()
    T1, T2 = texts("test" if split == "test" else "train")      # validation rows index the train split
    r["llm"] = llm_score(model("load"), prompts(T1, T2, r.sid.values, r.rid.values))
    r.to_parquet(f"{XD}/llmhi_{split}_{i}of{n}.parquet")
    print("wrote", f"{XD}/llmhi_{split}_{i}of{n}.parquet", len(r), flush=True)


def scored(split):
    """every scored shard of a split (any shard count), checked against the row list"""
    h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{XD}/llmhi_{split}_*of*.parquet"))], ignore_index=True)
    h = h.drop_duplicates(["rid", "sid"])
    need = pd.read_parquet(f"{XD}/llmhi_{split}_rows.parquet") if split == "test" else h
    assert len(h) == len(need), f"x_llmhi.py {split}: a shard is missing ({len(h)} of {len(need)} rows scored)"
    return h


def check(m0=0.0):
    v, h = scored("val"), scored("test")
    tr, ot = v.llm.values[v.y.values], v.llm.values[~v.y.values]
    for nm, x in (("validation true", tr), ("validation not true", ot), ("test", h.llm.values)):
        print(f"{nm:20s} n {len(x):7d}  margin quantiles 1/5/10/50%: {np.percentile(x, [1, 5, 10, 50]).round(2)}", flush=True)
    t = pd.DataFrame([{"cut": m, "val_true": (tr < m).mean(), "val_not_true": (ot < m).mean(), "test": (h.llm.values < m).mean(),
                       "test_rows": int((h.llm.values < m).sum())} for m in CUTS])
    t["max_true_share"] = t.val_true / t.test   # if the test country's true matches looked like US / India's to the judge
    print(t.round(5).to_string(index=False), flush=True)
    raw2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True); raw1 = load("test", 1)
    k = np.flatnonzero(h.llm.values < m0)
    print(f"\n{len(k)} test rows below {m0}; 30 random (record || its S1):", flush=True)
    for i in np.random.default_rng(1).choice(k, min(30, len(k)), replace=False):
        r, s = h.rid.values[i], h.sid.values[i]
        print(f"{h.llm.values[i]:+.2f} | {raw2.business_name[r]} ; {raw2.business_address[r]}  ||  "
              f"{raw1.business_name[s]} ; {raw1.business_address[s]}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "rows":
        rows(*a[1:4])
    elif a[0] == "check":
        check(float(a[1]) if len(a) > 1 else 0.0)
    else:
        score(a[0])
