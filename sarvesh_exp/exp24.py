"""E33: invented-name records. A record whose core-name tokens appear in no S1 name (split-wide vocabulary) has an
invented brand name. Measure their share / recall / truth rate on validation, then a rule: such a record is matched
to the only S1 with the same address key (exact normalised address; or the sorted token set of the address), row
moved / added with q = 1, only if not already accepted. Cross-fitted over the key choice; test: counts per country."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load
from harness import truth_arrays, wscore
from match import normed
W_ = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
T = np.bincount(ts[ts >= 0], minlength=n); ents = np.where(s1f >= 8)[0]
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
o2c = pd.concat([load("train", 2), load("train", 3)], ignore_index=True).country.values
def invented(names, vocab):
    return np.array([bool(x) and all(t not in vocab for t in x.split()) for x in names])
vocab = set(" ".join(n1.cn.values).split())
inv = invented(n2.cn.values, vocab)
v = pd.read_parquet(f"{W_}/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"]).astype(
    {"sid": np.int64, "q": np.float64, "y": bool, "wt": np.float64})
W = v.wt.values[(ts[v.rid.values] < 0)].mean()
acc_r = np.zeros(len(n2), bool); acc_r[v.rid.values[v.q.values >= 0.70]] = True
print(f"invented-name records: {inv.mean():.4f} of all; truly matched {(ts[inv] >= 0).mean():.4f} "
      f"(others {(ts[~inv] >= 0).mean():.4f}); with empty address {(n2.na.values[inv] == '').mean():.3f}", flush=True)
r = np.flatnonzero((ts >= 0) & (s1f[np.maximum(ts, 0)] >= 8)); s = ts[r]
vi = v.set_index("rid"); ok = (vi.sid.reindex(r).values == s) & (vi.q.reindex(r).values >= 0.70)
print(f"VAL true pairs with invented name: {inv[r].sum():,} ({inv[r].mean():.4f})  recall {ok[inv[r]].mean():.4f}  "
      f"missed = {(~ok & inv[r]).sum() / len(r):.5f} of all true pairs", flush=True)
def akey(na, mode):
    if mode == "exact": return na
    return np.array([" ".join(sorted(set(x.split()))) for x in na])
def apply(sel):
    d = v.copy(); pos = pd.Series(np.arange(len(d)), index=d.rid.values).reindex(sel.rid.values).values
    iv = ~np.isnan(pos); ix = pos[iv].astype(int)
    sv, qv, yv = d.sid.values.copy(), d.q.values.copy(), d.y.values.copy()
    sv[ix] = sel.sid.values[iv]; qv[ix] = 1.0; yv[ix] = sel.y.values[iv]; d["sid"], d["q"], d["y"] = sv, qv, yv
    rr = sel.rid.values[~iv]
    new = pd.DataFrame({"rid": rr, "sid": sel.sid.values[~iv], "q": 1.0, "y": sel.y.values[~iv], "wt": np.where(ts[rr] < 0, W, 1.0)})
    return pd.concat([d, new], ignore_index=True)
base = lambda e: wscore(v.sid.values, v.q.values >= 0.70, v.y.values, v.wt.values, T, e)
out = {}
for mode in ("exact", "tokens"):
    k1 = akey(n1.na.values, mode) + "|" + s1.country.values
    uniq = pd.Series(np.arange(n), index=k1)[~pd.Index(k1).duplicated(keep=False)]
    for only_missing in (True, False):
        Q = np.flatnonzero(inv & (rf >= 8) & (n2.na.values != "") & (~acc_r if only_missing else True))
        tgt = uniq.reindex(akey(n2.na.values[Q], mode) + "|" + o2c[Q]).values; m = ~np.isnan(tgt)
        sel = pd.DataFrame({"rid": Q[m], "sid": tgt[m].astype(np.int64)}); sel = sel[s1f[sel.sid.values] >= 8]
        sel["y"] = ts[sel.rid.values] == sel.sid.values
        d = apply(sel)
        Fd = lambda e: wscore(d.sid.values, d.q.values >= 0.70, d.y.values, d.wt.values, T, e)
        g = {f: Fd(ents[s1f[ents] == f]) - base(ents[s1f[ents] == f]) for f in (8, 9)}; out[(mode, only_missing)] = g
        print(f"RULE addr={mode} only_not_accepted={only_missing}: pairs {len(sel):,} precision {sel.y.mean():.4f} "
              f"(distractor {(ts[sel.rid.values] < 0).mean():.4f})  gain fold8 {g[8]:+.6f} fold9 {g[9]:+.6f}", flush=True)
gs = []
for fit, ev in ((8, 9), (9, 8)):
    b = max(out, key=lambda k: out[k][fit]); gs.append(out[b][ev]); print(f"fit {fit}: best {b} -> fold {ev} {out[b][ev]:+.6f}", flush=True)
print(f"E33 cross-fitted mean gain: {np.mean(gs):+.6f}", flush=True)
# test counts
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
t2c = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values
tinv = invented(m2.cn.values, set(" ".join(m1.cn.values).split()))
q17 = pd.read_parquet("/scratch/scai/mtech/aib262045/amlc_e10_0.005/work/x/test_q_resc17.parquet", columns=["rid", "q", "c"])
acc = np.zeros(len(m2), bool); acc[q17.rid.values[q17.q.values >= 0.70]] = True
for c in ("US", "India", "France"):
    k = t2c == c; print(f"TEST {c}: invented-name records {tinv[k].mean():.4f}; accepted (US/India from E17) {acc[k & tinv].mean():.4f} vs others {acc[k & ~tinv].mean():.4f}", flush=True)
for c in ("US", "India"):
    k = o2c == c; print(f"TRAIN {c}: invented-name records {inv[k].mean():.4f}; truly matched {(ts[k & inv] >= 0).mean():.4f}", flush=True)
