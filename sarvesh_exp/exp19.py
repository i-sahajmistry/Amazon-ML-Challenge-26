"""E27 test files: E17 + the name-only rule. A test record with an empty normalised address whose core name equals the
core name of exactly one test S1 (same country) is matched to that S1 (its row moved there, q = 1). The rule's pairs
join the candidate set as a second blocking channel (exact name key), so matched IDs stay within candidate_pairs.
COUNTRIES = which countries get the rule (US,India = validated; +France = leaderboard bet). France keeps the
variant's chain (x/test_q_frchain = FROM applied) + restore / reject before the rule."""
import os, sys, numpy as np, pandas as pd
from common import load, WORK
from match import normed
OUT_TAG = sys.argv[1]; CTRY = sys.argv[2].split(","); KEYS = (sys.argv[3] if len(sys.argv) > 3 else "cn").split(",")
X = f"{WORK}/x"
ui = pd.read_parquet(f"{X}/test_q_resc17.parquet"); ui = ui[ui.c != "France"]
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
d = pd.concat([ui, fr], ignore_index=True)[["rid", "sid", "q", "c"]].astype({"rid": np.int64, "sid": np.int64, "q": np.float64})
assert d.rid.is_unique
t1 = load("test", 1); t2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
empty = m2.na.values == ""
props = []
done = np.zeros(len(m2), bool)
for key in KEYS:
    k1 = m1[key].values + "|" + t1.country.values
    uniq = pd.Series(np.arange(len(t1)), index=k1)[~pd.Index(k1).duplicated(keep=False)]
    rr = np.flatnonzero(empty & (m2[key].values != "") & ~done & np.isin(t2.country.values, CTRY))
    tgt = uniq.reindex(m2[key].values[rr] + "|" + t2.country.values[rr]).values; ok = ~np.isnan(tgt)
    rr, tgt = rr[ok], tgt[ok].astype(np.int64); done[rr] = True
    props.append(pd.DataFrame({"rid": rr, "sid": tgt}))
    print(f"rule {key}: {len(rr):,} pairs " + str(pd.Series(t1.country.values[tgt]).value_counts().to_dict()), flush=True)
p = pd.concat(props, ignore_index=True)
pos = pd.Series(np.arange(len(d)), index=d.rid.values).reindex(p.rid.values).values; inv = ~np.isnan(pos); ix = pos[inv].astype(int)
was = (d.sid.values[ix] == p.sid.values[inv]) & (d.q.values[ix] >= 0.70)
moved = (d.sid.values[ix] != p.sid.values[inv]) & (d.q.values[ix] >= 0.70)
sv, qv = d.sid.values.copy(), d.q.values.copy(); sv[ix] = p.sid.values[inv]; qv[ix] = 1.0; d["sid"], d["q"] = sv, qv
new = pd.DataFrame({"rid": p.rid.values[~inv], "sid": p.sid.values[~inv], "q": 1.0, "c": t1.country.values[p.sid.values[~inv]]})
d = pd.concat([d, new], ignore_index=True)
print(f"rule pairs {len(p):,}: already accepted there {was.sum():,}; newly accepted {len(p) - was.sum():,} "
      f"(moved from another accepted S1 {moved.sum():,}, new rows {len(new):,})", flush=True)
d.to_parquet(f"{X}/test_q{OUT_TAG}.parquet")
cand = pd.read_parquet(f"{X}/p5_test_v10b.parquet", columns=["rid", "sid"])
BIG = np.int64(len(t1) + 1); have = cand.rid.values.astype(np.int64) * BIG + cand.sid.values
add = p[~np.isin(p.rid.values.astype(np.int64) * BIG + p.sid.values, have)]
pd.concat([cand, add[["rid", "sid"]]], ignore_index=True).to_parquet(f"{X}/p5_test{OUT_TAG}.parquet")
print(f"candidate pairs: {len(cand):,} + {len(add):,} name-channel pairs", flush=True)
