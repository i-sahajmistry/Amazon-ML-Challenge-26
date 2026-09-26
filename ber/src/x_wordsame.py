"""Label-free true share of same-address records, per added word. Decoys of an S1 mostly sit at one shifted house number
D; a small, roughly constant share alpha of each decoy type sits at the S1's own number (SAME). So for a word w that
records add to their S1's core name: expected decoys at SAME with w ~= alpha * count(w at D), and the rest of the SAME
records with w are true-match noise. alpha is measured on the words that only decoys use (their SAME / D ratio).
Validated on labelled US / India validation (implied vs actual true share at SAME), then applied to France test.
Also scores any restore set against the estimate: x/test_q{RS}.parquet rows (RS env, e.g. _rule_v10plw) not accepted by
the best file.
  python x_wordsame.py"""
import os, numpy as np, pandas as pd
from x_anatomy import positions, XD
from common import load
from match import normed

MIN = 300


def table(d, lab=None, alpha=None):
    one = d[(d["add"].str.count(" ") == 0) & (d["add"] != "") & d.pos.isin(["SAME", "D"])]
    g = one.groupby(["add", "pos"]).size().unstack(fill_value=0)
    for p in ("SAME", "D"):
        if p not in g:
            g[p] = 0
    g = g[g.SAME + g.D >= MIN]
    if alpha is None:   # the words decoys use: the ones whose SAME share is lowest
        r = (g.SAME / (g.SAME + g.D)).sort_values()
        alpha = float(np.median((g.SAME / g.D.clip(lower=1))[r.index[:6]]))
    g["implied_true"] = (1 - alpha * g.D / g.SAME.clip(lower=1)).clip(lower=0)
    if lab is not None:
        s = one[one.pos == "SAME"].groupby("add")[lab].mean()
        g["actual_true"] = s.reindex(g.index)
    return g.sort_values("SAME", ascending=False), alpha


n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
s1 = load("train", 1)
v = pd.read_parquet(f"{XD}/val_q_v10pw.parquet")
v = positions(v.assign(c=s1.country.values[v.sid.values]), n1, n2)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 80)
for c in ("US", "India"):
    g, a = table(v[v.c.values == c], lab="y")
    ok = g.actual_true.notna()
    print(f"\n== {c} validation, alpha {a:.4f}; implied vs actual true share at SAME: corr "
          f"{np.corrcoef(g.implied_true[ok], g.actual_true[ok])[0, 1]:.3f}, mean abs err "
          f"{np.mean(np.abs(g.implied_true[ok] - g.actual_true[ok])):.3f}", flush=True)
    print(g.head(40).round(3).to_string(), flush=True)

t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
t = pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet")
t = positions(t[t.c.values == "France"].reset_index(drop=True), t1, t2)
t["acc"] = t.q.values >= 0.7
g, a = table(t, lab="acc")
g = g.rename(columns={"actual_true": "accepted"})
print(f"\n== France test, alpha {a:.4f} (accepted = share the best file accepts at SAME)", flush=True)
print(g.head(60).round(3).to_string(), flush=True)
g.to_csv(f"{XD}/wordsame_France.csv")
