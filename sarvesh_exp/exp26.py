"""E37 (label-free, test): acceptance rate by record type per country. US / India from E17, France from the variant's
chain (frchain + restore / reject). Types as E34 (name-only shared / unique / no-match, invented, normal). If France
deviates strongly for one type, that type is a France-specific lever (LB-only). Also the France chain's acceptance by
type before the vetoes (base _v10plw)."""
import numpy as np, pandas as pd
from common import load, WORK
from match import normed
X = f"{WORK}/x"
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
c2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values
cnt = pd.Series(m1.cn.values + "|" + t1.country.values).value_counts()
amb = cnt.reindex(m2.cn.values + "|" + c2).fillna(0).values
vocab = set(" ".join(m1.cn.values).split())
inv = np.array([bool(x) and all(t not in vocab for t in x.split()) for x in m2.cn.values])
empty = m2.na.values == ""
typ = np.select([empty & (amb >= 2), empty & (amb == 1), empty, inv], ["amb", "uniq", "typo", "invented"], "normal")
ui = pd.read_parquet(f"{X}/test_q_resc17.parquet"); ui = ui[ui.c != "France"]
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
b = pd.read_parquet(f"{X}/test_q_v10plw.parquet"); b = b[b.c == "France"]
acc = np.zeros(len(m2), bool); acc[ui.rid.values[ui.q.values >= 0.70]] = True; acc[fr.rid.values[fr.q.values >= 0.70]] = True
accb = np.zeros(len(m2), bool); accb[b.rid.values[b.q.values >= 0.70]] = True
rows = []
for c in ("US", "India", "France"):
    for k in ("amb", "uniq", "typo", "invented", "normal"):
        m = (c2 == c) & (typ == k)
        rows.append(dict(country=c, type=k, share=m.sum() / (c2 == c).sum(), records=m.sum(), accepted=acc[m].mean(),
                         fr_base_before_veto=accb[m].mean() if c == "France" else np.nan))
print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
