"""Data-driven word lists for the France same-address fixes.
For every word w that records add to their S1's core name at the S1's own house number (SAME): the census distribution
of those records over nD (the S1's records at other numbers, 0..4+), compared with unedited SAME records by total
variation distance TV(w). Decoys placed at the S1's address leave its shifted decoy cluster short, so decoy-made words
have a large TV, while true-noise words spread like unedited records.
  true-like  : TV <= TV_TRUE, SAME / D >= 0.1 (not a pure shifted-decoy word), >= MIN records
  decoy-like : TV >= TV_DEC, >= MIN records
Validated on labelled US / India validation (true rate at SAME by TV bin), then applied to France test:
x/restore_fr_dd (rule1 records whose added words are all true-like, or none) and x/reject_fr_dd (accepted SAME records
adding a decoy-like word). France's rows are v10_fr3_llm's combined q (x/test_q_v10fr3l).
  python x_wordlists.py [TV_TRUE=0.1] [TV_DEC=0.2] [MIN=300]"""
import sys, numpy as np, pandas as pd
from x_anatomy import positions, XD
from x_census import census, R1
from common import load
from match import normed

TV_TRUE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
TV_DEC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 300


def words(d, lab=None):
    """per added word at SAME: count, D count, TV of its nD distribution to the unedited SAME records."""
    s = d[d.pos == "SAME"]
    ref = s[s["add"] == ""].nD.value_counts(normalize=True).reindex(range(5), fill_value=0).values
    e = s[s["add"] != ""].assign(w=lambda x: x["add"].str.split()).explode("w")
    dd = d[d.pos == "D"].assign(w=lambda x: x["add"].str.split()).explode("w").w.value_counts()
    rows = []
    for w, g in e.groupby("w"):
        if len(g) < MIN:
            continue
        dist = g.nD.value_counts(normalize=True).reindex(range(5), fill_value=0).values
        r = dict(w=w, same=len(g), D=int(dd.get(w, 0)), tv=float(np.abs(dist - ref).sum() / 2))
        if lab is not None:
            r["true"] = g[lab].mean()
        rows.append(r)
    t = pd.DataFrame(rows).set_index("w")
    t["ratio"] = t.same / t.D.clip(lower=1)
    return t.sort_values("same", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
    s1 = load("train", 1)
    v = pd.read_parquet(f"{XD}/val_q_v10pw.parquet")
    v = census(positions(v.assign(c=s1.country.values[v.sid.values]), n1, n2))
    for c in ("US", "India"):
        t = words(v[v.c.values == c], lab="y")
        t["cls"] = np.where((t.tv <= TV_TRUE) & (t.ratio >= 0.1), "true-like", np.where(t.tv >= TV_DEC, "decoy-like", "middle"))
        agg = t.assign(n_true=t.same * t["true"]).groupby("cls").agg(words=("same", "size"), rows=("same", "sum"), n_true=("n_true", "sum"))
        agg["true_rate"] = agg.n_true / agg.rows
        print(f"\n== {c} validation: the TV rule vs labels (true rate of SAME records adding the word)\n"
              + agg.round(3).to_string() + "\n  by TV bin:\n"
              + t.groupby(pd.cut(t.tv, [0, 0.05, 0.1, 0.15, 0.2, 0.3, 1])).apply(
                  lambda g: pd.Series({"words": len(g), "rows": g.same.sum(), "true_rate": (g.same * g["true"]).sum() / max(g.same.sum(), 1)})).round(3).to_string(), flush=True)

    t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
    t = pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet")
    t = census(positions(t[t.c.values == "France"].reset_index(drop=True), t1, t2))
    t["acc"] = t.q.values >= 0.7
    fw = words(t)
    true_dd = set(fw[(fw.tv <= TV_TRUE) & (fw.ratio >= 0.1)].index)
    dec_dd = set(fw[fw.tv >= TV_DEC].index)
    print(f"\n== France words at SAME (>= {MIN} records), TV / ratio:\n" + fw.head(70).round(3).to_string(), flush=True)
    print(f"\ntrue-like ({len(true_dd)}): {sorted(true_dd)}\ndecoy-like ({len(dec_dd)}): {sorted(dec_dd)}")
    s = t[t.pos == "SAME"]
    aw = s["add"].str.split().apply(set)
    r = pd.read_parquet(R1)[["rid", "sid"]].merge(t[["rid", "sid", "add"]], on=["rid", "sid"], how="left")
    ra = r["add"].fillna("").str.split().apply(set)
    rest = r[ra.apply(lambda a: a <= true_dd)][["rid", "sid"]]
    rest.to_parquet(f"{XD}/restore_fr_dd.parquet")
    rej = s[s.acc & aw.apply(lambda a: bool(a & dec_dd))][["rid", "sid"]]
    rej.to_parquet(f"{XD}/reject_fr_dd.parquet")
    print(f"\nrestore {len(rest)} rule1 records, reject {len(rej)} accepted records", flush=True)
