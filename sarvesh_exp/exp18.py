"""E26: name-only records (empty normalised address). (1) Validation: recall and miss stage for empty- vs non-empty-
address records. (2) Rule: an empty-address record whose core name (cn, then skeleton sk) equals the core name of
exactly one S1 (among all S1s of the split) is matched to that S1. Scored on validation (folds 8-9, distractors
weighted) by adding / moving rows, cross-checked per fold. (3) Test: share of records with empty address and share
accepted, per country (label-free: is the same gap there?)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load, WORK
from harness import truth_arrays, wscore
from match import normed
W_ = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
empty = (n2.na.values == "")
c1 = s1.country.values
v = pd.read_parquet(f"{W_}/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"])
# (1) recall by address-empty, validation true pairs
r = np.flatnonzero((ts >= 0) & (s1f[np.maximum(ts, 0)] >= 8)); s = ts[r]
vi = v.set_index("rid"); bs = vi.sid.reindex(r).values; bq = vi.q.reindex(r).values
acc_true = (bs == s) & (bq >= 0.70)
for name, m in (("empty address", empty[r]), ("with address", ~empty[r])):
    print(f"VAL true pairs, {name}: {m.sum():,} ({m.mean():.4f} of true pairs)  recall {acc_true[m].mean():.4f}  "
          f"missed {(~acc_true[m]).sum():,} = {(~acc_true & m).sum() / len(r):.5f} of all true pairs", flush=True)
print(f"all records with empty address (train): {empty.mean():.4f}; among distractors {empty[ts < 0].mean():.4f}", flush=True)
# (2) name rule
W = v.wt.values[~v.y.values.astype(bool) & (ts[v.rid.values] < 0)].mean()
for key in ("cn", "sk"):
    k1 = n1[key].values; cnt = pd.Series(k1).value_counts()
    uniq = pd.Series(np.arange(n), index=k1)[~pd.Index(k1).duplicated(keep=False)]      # key -> the only S1 having it
    rr = np.flatnonzero(empty & (rf >= 8) & (n2[key].values != ""))
    tgt = uniq.reindex(n2[key].values[rr]).values; ok = ~np.isnan(tgt)
    rr, tgt = rr[ok], tgt[ok].astype(int)
    keep = s1f[tgt] >= 8; rr, tgt = rr[keep], tgt[keep]
    tru = ts[rr] == tgt
    print(f"RULE {key}: proposals {len(rr):,} (validation, empty address, unique S1 key)  true {tru.mean():.4f}  "
          f"distractor {(ts[rr] < 0).mean():.4f}  other-S1 match {((ts[rr] >= 0) & ~tru).mean():.4f}", flush=True)
    # apply: records already in v -> move their row to tgt and accept; others -> new rows
    d = v.copy(); pos = pd.Series(np.arange(len(d)), index=d.rid.values)
    have = pos.reindex(rr).values; inv = ~np.isnan(have)
    ix = have[inv].astype(int)
    already = (d.sid.values[ix] == tgt[inv]) & (d.q.values[ix] >= 0.70)
    d.loc[d.index[ix], "sid"] = tgt[inv]; d.loc[d.index[ix], "q"] = 1.0
    d.loc[d.index[ix], "y"] = ts[rr[inv]] == tgt[inv]
    new = pd.DataFrame({"rid": rr[~inv], "sid": tgt[~inv], "q": 1.0, "y": ts[rr[~inv]] == tgt[~inv],
                        "wt": np.where(ts[rr[~inv]] < 0, W, 1.0)})
    d = pd.concat([d, new], ignore_index=True)
    print(f"  moved/accepted {inv.sum():,} existing rows ({already.sum():,} already accepted there), added {len(new):,} new rows", flush=True)
    base = v.q.values >= 0.70
    Fb = lambda e: wscore(v.sid.values, base, v.y.values.astype(bool), v.wt.values, T, e)
    Fr = lambda e: wscore(d.sid.values, d.q.values >= 0.70, d.y.values.astype(bool), d.wt.values, T, e)
    for f in (8, 9):
        e = ents[s1f[ents] == f]; print(f"  fold {f}: base {Fb(e):.5f}  rule {Fr(e):.5f}  gain {Fr(e) - Fb(e):+.6f}", flush=True)
    print(f"  ALL: gain {Fr(ents) - Fb(ents):+.6f}", flush=True)
    # by S1 name multiplicity when not unique: how many empty-address true pairs have a non-unique key
    rt = np.flatnonzero(empty & (ts >= 0)); kk = n2[key].values[rt]
    print(f"  empty-address true pairs whose record {key} equals the S1 {key}: {(kk == k1[ts[rt]]).mean():.4f}; "
          f"of those, key unique among S1s: {cnt.reindex(kk[kk == k1[ts[rt]]]).eq(1).mean():.4f}", flush=True)
# (3) test
t1 = load("test", 1); t2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True); te = m2.na.values == ""
q17 = pd.read_parquet("/scratch/scai/mtech/aib262045/amlc_e10_0.005/work/x/test_q_resc17.parquet", columns=["rid", "q", "c"])
acc = np.zeros(len(m2), bool); acc[q17.rid.values[(q17.q.values >= 0.70) & (q17.c.values != "France")]] = True
for c in ("US", "India"):
    k = t2.country.values == c
    print(f"TEST {c}: empty-address records {te[k].mean():.4f}; accepted: empty {acc[k & te].mean():.4f} vs with address {acc[k & ~te].mean():.4f}", flush=True)
k = t2.country.values == "France"; print(f"TEST France: empty-address records {te[k].mean():.4f}", flush=True)
tr2 = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
for c in ("US", "India"):
    k = tr2.country.values == c
    print(f"TRAIN {c}: empty-address records {empty[k].mean():.4f}; truly matched: empty {(ts[k & empty] >= 0).mean():.4f} vs with address {(ts[k & ~empty] >= 0).mean():.4f}", flush=True)
