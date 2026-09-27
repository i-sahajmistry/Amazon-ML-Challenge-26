"""E20 on US / India validation (bge stack, _v10bpw; entity folds 8-9; distractors weighted; macro F0.5 per S1).
Expected-F0.5 decision per S1 instead of one threshold: for each S1, sort its claimants by q and accept the top k
(k = 0..m) that maximises the expected F0.5, with q (optionally recalibrated: sigmoid(a * logit(q) + b)) taken as
each claimant's chance of being a true match. Expectation by Monte Carlo (independent Bernoulli draws, common
random numbers). k = 0 scores 1 only if the S1 has no true match. Calibration (a, b) is cross-fitted fold 8 <-> 9."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays, wscore
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]; n = len(s1f)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet")
sid, q, y, wt = v.sid.values, v.q.values, v.y.values.astype(bool), v.wt.values
def F(acc, e): return wscore(sid, acc, y, wt, T, e)
base = q >= 0.70
# rescue baseline (E13a, t = 0.5): empty S1 takes its best claimant if q >= 0.5
top = pd.Series(q).groupby(sid).transform("max").values
isbest = (q == top) & ~pd.Series(q == top).groupby(sid).cumsum().gt(1).values
n_acc = np.bincount(sid[base], minlength=n)
resc = base | (isbest & (n_acc[sid] == 0) & (q >= 0.5))
# group layout: rows sorted by S1, then q descending
o = np.lexsort((-q, sid)); ss = sid[o]; N = len(o)
new = np.r_[True, ss[1:] != ss[:-1]]; starts = np.flatnonzero(new); gid = np.cumsum(new) - 1; G = len(starts)
k = np.arange(N) - starts[gid] + 1
def gcum(x):
    c = np.cumsum(x, dtype=np.float64); return c - (c[starts] - x[starts])[gid]
lq = np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
def decide(a, b, B=64):
    p = (1 / (1 + np.exp(-(a * lq + b))))[o]
    rng = np.random.default_rng(0); EF = np.zeros(N); E0 = np.zeros(G)
    for _ in range(B):
        z = (rng.random(N) < p).astype(np.float64)
        Tg = np.bincount(gid, weights=z, minlength=G)
        EF += 1.25 * gcum(z) / (0.25 * Tg[gid] + k); E0 += Tg == 0
    EF /= B; E0 /= B
    arg = np.lexsort((-EF, gid))[starts]              # best row (largest k-candidate EF) per S1
    kst = np.where(EF[arg] > E0, k[arg], 0)
    acc = np.empty(N, bool); acc[o] = k <= kst[gid]
    return acc
grid = [(a, b) for a in (0.8, 1.0, 1.25, 1.5) for b in (-0.5, -0.25, 0.0, 0.25, 0.5)]
accs = {}
for a, b in grid:
    accs[(a, b)] = decide(a, b)
    print(f"a={a} b={b:+.2f}  all folds {F(accs[(a, b)], ents) - F(base, ents):+.6f}  vs rescue {F(accs[(a, b)], ents) - F(resc, ents):+.6f}"
          f"  accepted {accs[(a, b)].sum():,} (base {base.sum():,})", flush=True)
print(f"base 0.70 {F(base, ents):.5f}   rescue {F(resc, ents):.5f} ({F(resc, ents) - F(base, ents):+.6f})", flush=True)
gains, gr = [], []
for fit, ev in ((8, 9), (9, 8)):
    ef, ee = ents[s1f[ents] == fit], ents[s1f[ents] == ev]
    sc = {c: F(a_, ef) for c, a_ in accs.items()}; c = max(sc, key=sc.get)
    g = F(accs[c], ee) - F(base, ee); gains.append(g); gr.append(F(accs[c], ee) - F(resc, ee))
    print(f"E20 fit fold {fit} best (a, b)={c} -> fold {ev}: vs 0.70 {g:+.6f}, vs rescue {gr[-1]:+.6f}", flush=True)
print(f"E20 cross-fitted mean gain: vs 0.70 {np.mean(gains):+.6f}, vs 0.70 + rescue {np.mean(gr):+.6f}", flush=True)
