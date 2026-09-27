"""Experiment (not part of the pipeline): how small can the candidate list get? Recomputes the text shortlist
probability P for every retrieved pair (as match.keep_mask with SHORTLIST=text) and, per cut-off tau, reports:
test pairs kept (total / per S1, per country), accepted matches of a reference submission that would be cut, and on
train S1 folds 7-9 (unseen by the shortlist model) the share of matched records whose true S1 stays in the list."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, load
from match import normed, retrieval_feats, text_feats, SHORT_F, TXT
from harness import truth_arrays

TAUS = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05]
REF = os.environ.get("REF", "/scratch/scai/mtech/aib262045/amlc_v9x/output_v9x_0/matching_results.tsv")
m = lgb.Booster(model_file=f"{WORK}/shortlist_text.txt")


def probs(split, rid_keep=None):
    c = pd.read_parquet(f"{WORK}/cand_{split}.parquet")
    if rid_keep is not None:
        c = c[rid_keep[c.rid.values]].reset_index(drop=True)
    c = retrieval_feats(c)
    n1 = normed(split, 1); n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c = text_feats(c, n1, n2)
    c["P"] = m.predict(c[SHORT_F + TXT], num_threads=int(os.environ.get("NT", 8)))
    return c[["rid", "sid", "P"]]


# ---------------- test
c = probs("test")
s1 = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
ctry = s1.country.values[c.sid.values]
n_s1 = s1.country.value_counts()
print(f"test retrieved pairs {len(c):,}; kept at 0.001: {(c.P.values >= 0.001).sum():,} (pipeline: 12,760,925)", flush=True)
sid_of = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
rid_of = pd.Series(np.arange(len(other)), index=other.entity_id.values)
ref = pd.read_csv(REF, sep="\t", dtype=str).fillna("")
acc = [(sid_of[a], rid_of[b]) for a, lst in zip(ref.source1_entity_id, ref.matched_entity_ids) if lst for b in lst.split(",")]
acc = pd.DataFrame(acc, columns=["sid", "rid"])
key = c.rid.values.astype(np.int64) * len(s1) + c.sid.values
akey = acc.rid.values.astype(np.int64) * len(s1) + acc.sid.values
Pacc = pd.Series(c.P.values, index=key).reindex(akey).values
actry = s1.country.values[acc.sid.values]
print(f"reference accepted pairs {len(acc):,}; not among retrieved pairs: {np.isnan(Pacc).sum():,}", flush=True)
rows = []
for t in TAUS:
    k = c.P.values >= t
    r = {"tau": t, "pairs": int(k.sum())}
    for cc in n_s1.index:
        sel = ctry == cc
        r[f"{cc} per S1"] = round(k[sel].sum() / n_s1[cc], 2)
        r[f"{cc} accepted cut"] = int(((Pacc < t) & (actry == cc)).sum())
    rows.append(r)
print("\nTEST\n" + pd.DataFrame(rows).to_string(index=False), flush=True)

# ---------------- train, S1 folds 7-9 (the shortlist model never saw them)
_, _, ts, s1f, rf = truth_arrays()
keep = np.isin(rf, [7, 8, 9])
c = probs("train", keep)
key = c.rid.values.astype(np.int64) * (s1f.shape[0]) + c.sid.values
matched = np.flatnonzero(keep & (ts >= 0))
tkey = matched.astype(np.int64) * s1f.shape[0] + ts[matched]
Pt = pd.Series(c.P.values, index=key).reindex(tkey).fillna(-1).values
cnt = load("train", 1).country.values[ts[matched]]
rows = []
for t in TAUS:
    r = {"tau": t, "pairs per record": round((c.P.values >= t).sum() / keep.sum(), 3)}
    for cc in np.unique(cnt):
        r[f"{cc} true S1 kept %"] = round(100 * (Pt[cnt == cc] >= t).mean(), 3)
    rows.append(r)
print("\nTRAIN folds 7-9\n" + pd.DataFrame(rows).to_string(index=False), flush=True)
