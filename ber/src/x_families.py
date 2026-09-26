"""France same-address edit families (from the hand review + x_wordsame ratios):
  SUF7 : adds only words of the uniform 7-word suffix list (fils, associes, cie, services, groupe, developpement,
         france; 'et' allowed with them) - SAME/D ratio 6-15 like the US true suffix op (center/services)
  TYPE : adds a word whose SAME/D ratio is ~0.4-0.8 (the S1 type-word vocabulary: club, ecole, amicale...)
  DEC  : adds a pure decoy word (SAME/D < 0.1)
Checks (1) the families against rule1's LB-implied true share (~0.58): if SUF7 is true and TYPE is a decoy, rule1's
predicted share is SUF7 / all; (2) the France best-file acceptance per family at SAME; (3) the census (nD) per family.
Writes x/fam_France.parquet (rid, sid, fam, pos, acc, nD) for building probes.
  python x_families.py"""
import numpy as np, pandas as pd
from x_anatomy import positions, XD
from x_census import census, R1
from match import normed

SUF7 = {"fils", "associes", "cie", "services", "groupe", "developpement", "france"}
w = pd.read_csv(f"{XD}/wordsame_France.csv", index_col=0)
ratio = (w.SAME / w.D.clip(lower=1))
TYPEW = set(ratio[(ratio > 0.3) & (ratio < 1.2) & (w.SAME >= 150)].index) - SUF7
DECW = set(ratio[ratio < 0.1].index)
print("type words:", len(TYPEW), sorted(TYPEW)[:60], "\ndecoy words:", sorted(DECW), flush=True)


def fam(add):
    a = set(add.split())
    if not a:
        return "NONE"
    if a & DECW:
        return "DEC"
    if a <= SUF7 | {"et"} and a & SUF7:
        return "SUF7"
    if a & TYPEW:
        return "TYPE" if not a & SUF7 else "TYPE+SUF"
    return "OTHER"


t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
t = pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet")
t = census(positions(t[t.c.values == "France"].reset_index(drop=True), t1, t2))
t["acc"] = t.q.values >= 0.7
t["fam"] = [fam(a) for a in t["add"].values]
s = t[t.pos == "SAME"]
print("\nFrance SAME rows by family: rows, accepted share, rejected rows\n"
      + s.groupby("fam").agg(rows=("acc", "size"), acc=("acc", "mean"), rej=("acc", lambda x: (~x).sum())).round(3).to_string())
print("\nFrance SAME rows by family x nD (rows per family, share by nD):\n"
      + pd.crosstab(s.fam, s.nD, normalize="index").round(3).to_string())
r = pd.read_parquet(R1)[["rid", "sid"]].merge(t[["rid", "sid", "fam", "add", "drop"]], on=["rid", "sid"], how="left")
c = r.fam.value_counts()
print("\nrule1 additions by family:\n" + c.to_string())
print(f"predicted rule1 true share if SUF7 (+NONE) true and everything else a decoy: "
      f"{(c.get('SUF7', 0) + c.get('NONE', 0)) / c.sum():.3f}   (LB-implied ~0.58)")
print(f"if SUF7, NONE and TYPE+SUF true: {(c.get('SUF7', 0) + c.get('NONE', 0) + c.get('TYPE+SUF', 0)) / c.sum():.3f}")
print("rule1 TYPE+SUF / OTHER samples:\n" + r[r.fam.isin(["TYPE+SUF", "OTHER"])][["add", "drop", "fam"]].sample(20, random_state=0).to_string())
t[["rid", "sid", "fam", "pos", "acc", "nD", "add", "drop"]].to_parquet(f"{XD}/fam_France.parquet")
