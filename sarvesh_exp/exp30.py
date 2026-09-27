"""E41: 'number shifted DOWN, same core name, alone at that number' (decoys come in groups at one number shifted UP).
(1) US / India validation: true rate + model acceptance for down same-name rows, split by peers (other claimants of the
same S1 at the record's number). (2) France test rows of that kind that the chain rejects -> add list for patch_tsv.py
(F1: down same-name, no peer), (F2: + up>13 same-name, no peer). Prints examples for hand review."""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load, WORK
from harness import truth_arrays
from match import normed
OUT = sys.argv[1]
def ints(a):
    out = np.full(len(a), -1, np.int64)
    for i, x in enumerate(a):
        t = x.split()[0] if x else ""
        if t.isdigit() and len(t) < 12: out[i] = int(t)
    return out
def feats(M1, M2, r, s):
    a, b = ints(M2.num.values[r]), ints(M1.num.values[s]); d = a - b
    sn = M2.cn.values[r] == M1.cn.values[s]
    peers = pd.DataFrame({"s": s, "a": a}).groupby(["s", "a"]).s.transform("size").values - 1
    kind = np.select([(a >= 0) & (b >= 0) & (d < 0), (a >= 0) & (b >= 0) & (d > 13)], ["down", "up>13"], "other")
    return kind, sn, np.where(a >= 0, peers, -1)
s1, _, ts, s1f, rf = truth_arrays()
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"])
k, sn, pe = feats(n1, n2, v.rid.values, v.sid.values)
y, wt, acc = v.y.values.astype(bool), v.wt.values, v.q.values >= 0.70
for kk in ("down", "up>13"):
    for p in (0, 1):
        m = (k == kk) & sn & ((pe == 0) if p == 0 else (pe > 0))
        tr = y[m].sum() / np.where(y[m], 1.0, wt[m]).sum()
        rej = m & ~acc
        print(f"VAL {kk} same-name {'alone' if p == 0 else 'with peers'}: rows {m.sum():,} true(weighted) {tr:.4f} accepted {acc[m].mean():.4f}; "
              f"REJECTED ones: {rej.sum():,} true(weighted) {y[rej].sum() / max(np.where(y[rej], 1.0, wt[rej]).sum(), 1e-9):.4f}", flush=True)
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
X = f"{WORK}/x"
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    kk_ = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[kk_, "q"] = val
rj = pd.read_parquet(f"{X}/reject_fr_dd.parquet"); rjk = set(zip(rj.rid.values, rj.sid.values))
k, sn, pe = feats(m1, m2, fr.rid.values, fr.sid.values)
notrj = np.array([(a, b) not in rjk for a, b in zip(fr.rid.values, fr.sid.values)])
for name, sel in (("F1", (k == "down") & sn & (pe == 0)), ("F2", ((k == "down") | (k == "up>13")) & sn & (pe == 0))):
    m = sel & (fr.q.values < 0.70) & notrj
    print(f"FRANCE {name}: kind rows {sel.sum():,}, accepted {(sel & (fr.q.values >= 0.7)).mean() / max(sel.mean(), 1e-9):.3f}; to add (rejected, not in reject_fr_dd) {m.sum():,}; "
          f"q hist (0,.01,.1,.3,.5,.7): {np.histogram(fr.q.values[m], [0, .01, .1, .3, .5, .7])[0].tolist()}", flush=True)
    pd.DataFrame({"s1_id": t1.entity_id.values[fr.sid.values[m]], "rec_id": o.entity_id.values[fr.rid.values[m]]}).to_parquet(f"{OUT}/add_{name}.parquet")
    if name == "F1":
        for i in np.random.default_rng(0).choice(np.flatnonzero(m), min(25, m.sum()), replace=False):
            rr, ss = fr.rid.values[i], fr.sid.values[i]
            print(f"  q={fr.q.values[i]:.3f}  S1 [{t1.business_name.values[ss]}] [{t1.business_address.values[ss]}]\n           rec [{o.business_name.values[rr]}] [{o.business_address.values[rr]}]", flush=True)
