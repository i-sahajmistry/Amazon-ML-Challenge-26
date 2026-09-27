"""E32: what the with-address misses and the accepted false matches look like (US / India validation, bge stack).
Signatures + examples, to look for another simple pattern."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from harness import truth_arrays
from match import normed
W_ = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"
s1, _, ts, s1f, rf = truth_arrays(); n = len(s1f)
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
v = pd.read_parquet(f"{W_}/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"]).set_index("rid")
def first(x): return x.split()[0] if x else ""
def sig(r, s):
    a = np.array([first(x) for x in n2.num.values[r]]); b = np.array([first(x) for x in n1.num.values[s]])
    return pd.DataFrame({"rec_no_addr": n2.na.values[r] == "", "same_name": n2.cn.values[r] == n1.cn.values[s],
                         "same_addr": n2.na.values[r] == n1.na.values[s], "num_diff": (a != "") & (b != "") & (a != b),
                         "rec_no_num": a == "", "zip_diff": (n2.zip.values[r] != "") & (n1.zip.values[s] != "") & (n2.zip.values[r] != n1.zip.values[s])})
def show(t, r, s, k=12, extra=None):
    print(f"--- {t}: {len(r):,}\n" + sig(r, s).mean().round(3).to_frame().T.to_string(index=False), flush=True)
    for i in np.random.default_rng(1).choice(len(r), min(k, len(r)), replace=False):
        e = f"  q={extra[i]:.3f}" if extra is not None else ""
        print(f"  S1  [{n1.nn.values[s[i]]}] [{n1.na.values[s[i]]}]\n  rec [{n2.nn.values[r[i]]}] [{n2.na.values[r[i]]}]{e}", flush=True)
r = np.flatnonzero((ts >= 0) & (s1f[np.maximum(ts, 0)] >= 8) & (n2.na.values != "")); s = ts[r]
bs = v.sid.reindex(r).values; bq = v.q.reindex(r).values
miss = ~((bs == s) & (bq >= 0.70))
show("ALL with-address true pairs (reference)", r[:300000], s[:300000], k=0)
low = miss & (bs == s)
show("MISSED with address, correct S1 but q < 0.70", r[low], s[low], extra=bq[low])
oth = miss & ~np.isnan(bq) & (bs != s)
show("MISSED with address, another S1 won (true S1 shown)", r[oth], s[oth])
ob = bs[oth].astype(int)
print("    winner S1 vs true S1: same core name {:.3f}, same address {:.3f}".format(
    (n1.cn.values[ob] == n1.cn.values[s[oth]]).mean(), (n1.na.values[ob] == n1.na.values[s[oth]]).mean()), flush=True)
for i in np.random.default_rng(2).choice(np.flatnonzero(oth), min(8, oth.sum()), replace=False):
    print(f"  rec [{n2.nn.values[r[i]]}] [{n2.na.values[r[i]]}]\n   true [{n1.nn.values[s[i]]}] [{n1.na.values[s[i]]}]\n   won  [{n1.nn.values[int(bs[i])]}] [{n1.na.values[int(bs[i])]}]  q={bq[i]:.3f}", flush=True)
nr = miss & np.isnan(bq)
show("MISSED with address, no stage-2 row (not retrieved / other-fold S1)", r[nr], s[nr])
vv = v.reset_index(); fa = vv[(vv.q >= 0.70) & ~vv.y.astype(bool)]
dis = ts[fa.rid.values] < 0
show("FALSE accepted: distractor merged", fa.rid.values[dis], fa.sid.values[dis], extra=fa.q.values[dis])
show("FALSE accepted: record of another S1", fa.rid.values[~dis], fa.sid.values[~dis], extra=fa.q.values[~dis])
