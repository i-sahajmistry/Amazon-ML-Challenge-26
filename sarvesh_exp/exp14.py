"""E22: where France loses its matches (label-free). (1) Records per country in sources 2+3 and the share accepted,
test vs train truth: all countries share one generator, so France's true matched share should equal US / India's.
(2) France matches per S1 through the variant's chain: base q (_v10plw) -> min with a1 -> a2 -> a3 -> restore / reject,
to see which step removes the matches France is missing relative to US / India (mean 3.40 predicted, 3.46 true)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load, WORK
from harness import truth_arrays
s1, _, ts, s1f, rf = truth_arrays()
o_tr = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
print("source2/3 columns:", list(o_tr.columns)[:12], flush=True)
if "country" in o_tr.columns:
    m = ts >= 0
    for c in pd.unique(o_tr.country): k = o_tr.country.values == c; print(f"TRAIN {c}: records {k.sum():,}  truly matched {m[k].mean():.4f}", flush=True)
t1 = load("test", 1); o_te = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
X = f"{WORK}/x"; nt = len(t1)
def rd(t): return pd.read_parquet(f"{X}/test_q{t}.parquet")
e17 = rd("_resc17")
acc_ui = e17[(e17.c != "France") & (e17.q >= 0.70)]
d = rd("_v10plw"); d = d[d.c == "France"].reset_index(drop=True)
isfr = t1.country.values == "France"
def report(name, q):
    a = q >= 0.70; cnt = np.bincount(d.sid.values[a], minlength=nt)[isfr]
    print(f"FRANCE {name:28s} accepted {a.sum():>8,}  per S1 {cnt.mean():.3f}  empty {(cnt == 0).mean():.4f}  4+ {(cnt >= 4).mean():.4f}", flush=True)
    return a
q = d.q.values.copy(); report("base _v10plw", q)
for t in ("_a1pw", "_a2pw", "_a3pw"):
    e = rd(t); e = e[e.c == "France"]
    m = d[["rid", "sid"]].merge(e[["rid", "sid", "q"]], on="rid", how="left", suffixes=("", "_b"))
    qb = np.where(m.sid.values == m.sid_b.values, m.q.values, 0.0)
    report(f"{t} alone", qb)
    q = np.minimum(q, qb); report(f"chain after min {t}", q)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(d.rid.values).values == d.sid.values
    q[k] = val; report(f"after {f}", q)
if "country" in o_te.columns:
    for c in pd.unique(o_te.country):
        k = o_te.country.values == c
        acc = len(acc_ui[acc_ui.c == c]) if c != "France" else (q >= 0.70).sum()
        print(f"TEST {c}: records {k.sum():,}  accepted {acc:,} ({acc / k.sum():.4f})", flush=True)
# France records whose base q >= 0.70 but the chain removed them: how many, and how many sit in S1s left with < 3
base = d.q.values >= 0.70; lost = base & (q < 0.70)
cnt = np.bincount(d.sid.values[q >= 0.70], minlength=nt)
print(f"FRANCE vetoed (base >= 0.70, final < 0.70): {lost.sum():,}; their S1 final count: " +
      str(np.bincount(np.minimum(cnt[d.sid.values[lost]], 5), minlength=6).tolist()) + " (0,1,2,3,4,5+)", flush=True)
print("FRANCE vetoed base-q histogram:", np.histogram(d.q.values[lost], bins=[0.7, 0.8, 0.9, 0.95, 0.99, 1.01])[0].tolist(), flush=True)
