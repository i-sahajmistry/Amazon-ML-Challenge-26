"""E21 'shape' diagnostic. The test sets of all countries come from one generator, so France's true cluster shape
(share of S1s with no match, matches per S1) should look like US / India's. Compare: train truth (US / India),
validation truth vs prediction (US / India, folds 8-9), test prediction E17 (all countries), and France under the
variant's chain (FROM + restore / reject) for thresholds and for a France rescue (best claimant if q >= t)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load, WORK
from harness import truth_arrays
pd.set_option("display.width", 200); pd.set_option("display.precision", 4)
def shape(cnt, cc):
    rows = {}
    for k in pd.unique(cc):
        x = cnt[cc == k]
        rows[k] = dict(S1=len(x), empty=(x == 0).mean(), mean=x.mean(), one=(x == 1).mean(), two=(x == 2).mean(),
                       three=(x == 3).mean(), four_plus=(x >= 4).mean())
    return pd.DataFrame(rows).T
s1, _, ts, s1f, rf = truth_arrays()
T = np.bincount(ts[ts >= 0], minlength=len(s1)); c = s1.country.values
print("== TRAIN truth (all folds)\n", shape(T, c), flush=True)
ents = np.where(s1f >= 8)[0]
print("== VALIDATION truth (folds 8-9)\n", shape(T[ents], c[ents]), flush=True)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet")
q = v.q.values; sid = v.sid.values; n = len(s1)
top = pd.Series(q).groupby(sid).transform("max").values
isbest = (q == top) & ~pd.Series(q == top).groupby(sid).cumsum().gt(1).values
base = q >= 0.70; na = np.bincount(sid[base], minlength=n)
resc = base | (isbest & (na[sid] == 0) & (q >= 0.5))
for name, a in (("0.70", base), ("0.70 + rescue 0.5", resc)):
    print(f"== VALIDATION predicted {name}\n", shape(np.bincount(sid[a], minlength=n)[ents], c[ents]), flush=True)
t1 = load("test", 1)
mr = pd.read_csv("/scratch/scai/mtech/aib262045/amlc_exp/out_E17/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
assert (mr.source1_entity_id.values == t1.entity_id.values).all()
nm = (mr.matched_entity_ids.str.count(",") + (mr.matched_entity_ids != "")).values
print("== TEST predicted, E17 file\n", shape(nm, t1.country.values), flush=True)
# France under the variant's chain
d = pd.read_parquet(f"{WORK}/x/test_q_frchain.parquet")
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{WORK}/x/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(d.rid.values).values == d.sid.values
    d.loc[k, "q"] = val
fr = d[d.c == "France"]; nt = len(t1); isfr = t1.country.values == "France"
q, sid = fr.q.values, fr.sid.values
top = pd.Series(q).groupby(sid).transform("max").values
isbest = (q == top) & ~pd.Series(q == top).groupby(sid).cumsum().gt(1).values
print("== FRANCE by threshold (share empty / matches per S1)", flush=True)
for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
    cnt = np.bincount(sid[q >= t], minlength=nt)[isfr]
    print(f"  thr {t:.2f}: empty {(cnt == 0).mean():.4f}  mean {cnt.mean():.3f}", flush=True)
base = q >= 0.70; na = np.bincount(sid[base], minlength=nt)
print("== FRANCE rescue (thr 0.70; empty S1 takes its best claimant if q >= t)", flush=True)
for t in (0.2, 0.3, 0.4, 0.5, 0.6):
    add = isbest & (na[sid] == 0) & (q >= t)
    cnt = np.bincount(sid[base | add], minlength=nt)[isfr]
    print(f"  t {t:.1f}: rescued {add.sum():,}  empty {(cnt == 0).mean():.4f}  mean {cnt.mean():.3f}", flush=True)
emp = np.flatnonzero((na == 0) & isfr)
bq = pd.Series(q).groupby(sid).max().reindex(emp).fillna(-1).values
print("== FRANCE empty S1s: best-claimant q distribution", np.histogram(bq, bins=[-1.01, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])[0], flush=True)
