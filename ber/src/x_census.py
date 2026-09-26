"""Decoy census: does an S1 with a short shifted-number decoy cluster hide its decoys at its own address?
nD = records in the S1's group at another house number (D or OTHER). If each S1 gets a roughly fixed number of decoys,
a decoy placed at the S1's own number (SAME) leaves nD short. Labelled US / India validation: decoy rate of edited SAME
records (a word added or dropped) by nD. France test: acceptance of edited SAME records by nD, and rule1's additions
(the sibling's same-address restore, LB -0.00125, ~58% true) by nD.
  python x_census.py"""
import numpy as np, pandas as pd
from x_anatomy import positions, XD
from common import load
from match import normed

R1 = f"{XD}/rule1_added.parquet"          # x_rule1.py: rule_fr.py accepts, v10_fr3_llm rejects


def census(d):
    off = d.pos.isin(["D", "OTHER"])
    nD = off.groupby(d.sid).sum()
    d = d.assign(nD=nD.reindex(d.sid.values).values.clip(max=4))
    d["edit"] = (d["add"] != "") | (d["drop"] != "")
    return d


if __name__ == "__main__":
    n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
    s1 = load("train", 1)
    v = pd.read_parquet(f"{XD}/val_q_v10pw.parquet")
    v = census(positions(v.assign(c=s1.country.values[v.sid.values]), n1, n2))
    for c in ("US", "India"):
        d = v[v.c.values == c]
        g = d.groupby("sid")
        print(f"\n== {c} validation: S1s by nD {np.bincount(g.nD.first().values, minlength=5).round(0).tolist()}; "
              f"decoys per S1 by nD {g.apply(lambda x: (~x.y).sum()).groupby(g.nD.first()).mean().round(2).tolist()}", flush=True)
        m = (d.pos == "SAME") & d.edit
        t = d[m].groupby("nD").agg(rows=("y", "size"), decoy=("y", lambda s: 1 - s.mean()), acc=("q", lambda s: (s >= 0.7).mean()))
        t["rows_per_S1"] = t.rows / g.nD.first().value_counts().reindex(t.index).values
        print("  edited SAME records by nD:\n" + t.round(4).to_string(), flush=True)

    t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
    t = census(positions(pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet"), t1, t2))
    for c in ("US", "India", "France"):
        d = t[t.c.values == c]
        g = d.groupby("sid")
        m = (d.pos == "SAME") & d.edit
        x = d[m].groupby("nD").agg(rows=("q", "size"), acc=("q", lambda s: (s >= 0.7).mean()))
        x["rows_per_S1"] = x.rows / g.nD.first().value_counts().reindex(x.index).values
        print(f"\n== {c} test: S1s by nD {np.bincount(g.nD.first().values, minlength=5).tolist()}\n  edited SAME records by nD:\n"
              + x.round(4).to_string(), flush=True)
    r = pd.read_parquet(R1)[["rid", "sid", "qb"]]
    fr = t[t.c.values == "France"]
    nD = fr.groupby("sid").nD.first()
    r["nD"] = nD.reindex(r.sid.values).values
    print("\n== rule1 additions (LB: ~58% true overall) by nD and stage q band:\n"
          + r.groupby(["nD", "qb"], observed=True).size().unstack(fill_value=0).to_string(), flush=True)
    r.to_parquet(f"{XD}/rule1_census.parquet")
