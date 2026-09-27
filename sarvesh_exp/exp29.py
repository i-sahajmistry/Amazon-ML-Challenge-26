"""E40: house-number shift direction. Decoys shift the S1's number UP by 1-13; true-match number noise goes both ways.
Buckets over (record number - S1 number): down (<0), up 1-13, up >13, and same-name vs name-edit, plus 'padding'
(same integer, different string). (1) US / India validation rows (best S1 per record): true rate, model acceptance.
(2) Test: acceptance per bucket, US / India (E17) vs France (chain + restore / reject)."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load, WORK
from harness import truth_arrays
from match import normed
def ints(a):
    out = np.full(len(a), -1, np.int64)
    for i, x in enumerate(a):
        t = x.split()[0] if x else ""
        if t.isdigit() and len(t) < 12: out[i] = int(t)
    return out
def buckets(M1, M2, r, s):
    a, b = ints(M2.num.values[r]), ints(M1.num.values[s])
    sn = M2.cn.values[r] == M1.cn.values[s]
    d = a - b; ok = (a >= 0) & (b >= 0) & (d != 0)
    sa, sb = np.array([x.split()[0] if x else "" for x in M2.num.values[r]]), np.array([x.split()[0] if x else "" for x in M1.num.values[s]])
    pad = (a >= 0) & (a == b) & (sa != sb)
    kind = np.select([pad, ok & (d < 0), ok & (d >= 1) & (d <= 13), ok & (d > 13)], ["pad", "down", "up1-13", "up>13"], "none")
    return pd.Series(np.where(kind == "none", "none", kind + np.where(sn, " same-name", " name-edit")))
s1, _, ts, s1f, rf = truth_arrays()
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet", columns=["rid", "sid", "q", "y", "wt"])
bk = buckets(n1, n2, v.rid.values, v.sid.values)
t = pd.DataFrame({"b": bk.values, "y": v.y.values.astype(bool), "wt": v.wt.values, "acc": v.q.values >= 0.70, "c": s1.country.values[v.sid.values]})
t["w_true"] = t.y * 1.0; t["w_all"] = np.where(t.y, 1.0, t.wt)
g = t[t.b != "none"].groupby(["c", "b"]).apply(lambda x: pd.Series({"rows": len(x), "true_rate(weighted)": x.w_true.sum() / x.w_all.sum(),
                                                                   "accepted": x.acc.mean(), "precision_of_accepted": x.y[x.acc].mean() if x.acc.any() else np.nan}))
print("== VALIDATION (US / India, best S1 rows)\n" + g.round(4).to_string(), flush=True)
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
X = f"{WORK}/x"
ui = pd.read_parquet(f"{X}/test_q_resc17.parquet"); ui = ui[ui.c != "France"]
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
b = pd.read_parquet(f"{X}/test_q_v10plw.parquet"); b = b[b.c == "France"][["rid", "sid", "q"]].rename(columns={"q": "q_base"})
fr = fr.merge(b, on=["rid", "sid"], how="left")
d = pd.concat([ui.assign(q_base=np.nan), fr], ignore_index=True)
tb = buckets(m1, m2, d.rid.values, d.sid.values)
u = pd.DataFrame({"b": tb.values, "c": d.c.values, "acc": d.q.values >= 0.70, "acc_base": d.q_base.values >= 0.70, "q": d.q.values})
ns = pd.Series(t1.country.values).value_counts()
h = u[u.b != "none"].groupby(["c", "b"]).agg(rows=("acc", "size"), accepted=("acc", "mean"), fr_base_accepted=("acc_base", "mean"),
                                             q_03_07=("q", lambda x: ((x >= 0.3) & (x < 0.7)).mean()))
h["rows_per_S1"] = h.rows / ns.reindex(h.index.get_level_values(0)).values
print("== TEST (US / India E17, France chain)\n" + h.round(4).to_string(), flush=True)
