import numpy as np, pandas as pd
from common import WORK
from harness import truth_arrays, build
s1, other, ts, s1f, rf = truth_arrays()
k = pd.read_parquet(f"{WORK}/pred_all_train.parquet")
sid, p, tru, T, ents = build(k, ts, s1f, rf)
d = pd.DataFrame({"sid": sid, "p": p, "t": tru})
d["n_conf"] = d.groupby("sid").p.transform(lambda x: (x >= 0.9).sum()) - (d.p >= 0.9)
d["ctx"] = pd.cut(d.n_conf, [-1, 0, 1, 2, 99], labels=["0 other conf", "1", "2", "3+"])
d["pb"] = pd.cut(d.p, [0.05, 0.2, 0.4, 0.6, 0.8, 0.9])
print(d.dropna().pivot_table(index="pb", columns="ctx", values="t", aggfunc="mean", observed=True).round(3).to_string())
print(d.dropna().pivot_table(index="pb", columns="ctx", values="t", aggfunc="size", observed=True).to_string())
