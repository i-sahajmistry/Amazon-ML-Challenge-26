"""The decoy-group signal: a record whose first house number differs from its S1's while another record claiming the
same S1 has that same number. Train (labels; claimed S1 = top-1 retrieved): how often such records are distractors.
Test (v9p stage-2 best S1): how many records it flags per country, how many of those v9p accepts, and samples.
  python x_numrule.py"""
import numpy as np, pandas as pd
from common import WORK, load
from harness import truth_arrays
from match import normed

first = lambda num: np.array([int(x.split()[0][:15]) if x else -1 for x in num])


MAXD = int(__import__("os").environ.get("MAXD", 10 ** 15))   # only numbers within MAXD of the S1's


def flag(rid, sid, kr, ks):
    d = pd.DataFrame({"s": sid, "k": kr})
    peer = d.groupby(["s", "k"]).s.transform("size").values > 1
    return (kr >= 0) & (ks >= 0) & (kr != ks) & (np.abs(kr - ks) <= MAXD) & peer


import os, sys
N_SAMPLE = int(os.environ.get("N_SAMPLE", 8))
ONLY = os.environ.get("ONLY", "")             # e.g. ONLY=France: skip train, sample that country only
s1, other, ts, s1f, rf = truth_arrays() if not ONLY else (None,) * 5
c = None if ONLY else pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
if not ONLY:
    c = c[c["rank"].values == 0]
    kr = first(pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True).num.values)
    ks = first(normed("train", 1).num.values)
    r, s = c.rid.values, c.sid.values
    f = flag(r, s, kr[r], ks[s])
    diff = (kr[r] >= 0) & (ks[s] >= 0) & (kr[r] != ks[s])
    for ctry in ("US", "India"):
        m = s1.country.values[s] == ctry
        print(f"train {ctry}: number differs {diff[m].mean():.3%} of records, of which distractors {(ts[r] != s)[m & diff].mean():.3f}; "
              f"flagged (differs + same-number peer) {f[m].mean():.3%}, of which distractors {(ts[r] != s)[m & f].mean():.4f}", flush=True)


q = pd.read_parquet(f"{WORK}/x/test_q_v9pw.parquet")
t1 = load("test", 1); t2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
tkr = first(pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True).num.values)
tks = first(normed("test", 1).num.values)
q["f"] = flag(q.rid.values, q.sid.values, tkr[q.rid.values], tks[q.sid.values])
q["diff"] = (tkr[q.rid.values] >= 0) & (tks[q.sid.values] >= 0) & (tkr[q.rid.values] != tks[q.sid.values])
txt = lambda df, i: f"{df.business_name.values[i]} | {df.business_address.values[i]}"
rng = np.random.default_rng(0)
for ctry in ([ONLY] if ONLY else ("US", "India", "France")):
    g = q[q.c == ctry]
    a = g.q.values >= 0.7
    print(f"test {ctry}: number differs {g['diff'].mean():.3%} (accepted {(a & g['diff'].values).sum()}), flagged "
          f"{g.f.mean():.3%} of records, accepted and flagged {(a & g.f.values).sum()} ({(a & g.f.values).sum() / a.sum():.3%} of accepted)",
          flush=True)
    for i in rng.choice(np.flatnonzero(a & g.f.values), N_SAMPLE, replace=False):
        print(f"   {g.q.values[i]:.2f}  S1 {txt(t1, g.sid.values[i])}\n         REC {txt(t2, g.rid.values[i])}", flush=True)
