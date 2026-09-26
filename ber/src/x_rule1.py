"""rule1: the same-address candidates for the France fixes (x_families / x_wordlists / x_frfix read it as x_census.R1).
The records the structural rule accepts (rule_fr.py -> x/test_q_rule_v10plw) that v10_fr3_llm rejects: its combined q
after the three France vetoes (x/test_q_v10fr3l, x_chain.py llm2 llm2_seed) is below the threshold. rule1 as a
submission (all of them restored) scored LB 0.985819; dd restores only those whose added words are true-like.
  python wstat.py test && python rule_fr.py _v10plw && python x_rule1.py   -> x/rule1_added.parquet"""
import numpy as np, pandas as pd
from common import WORK

XD, THR = f"{WORK}/x", 0.70

r = pd.read_parquet(f"{XD}/test_q_rule_v10plw.parquet")
b = pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet")[["rid", "sid", "q"]]
m = r[r.q.values > 0][["rid", "sid", "c"]].merge(b, on=["rid", "sid"], how="left")
a = m[~(m.q.values >= THR)].reset_index(drop=True)
a["qb"] = pd.cut(a.q.fillna(0), [-1, .1, .3, .5, .7, 2], labels=["<.1", ".1-.3", ".3-.5", ".5-.7", "vetoed"])
a[["rid", "sid", "c", "q", "qb"]].to_parquet(f"{XD}/rule1_added.parquet")
print(f"rule accepts {len(m)}; rejected by v10_fr3_llm {len(a)} -> {XD}/rule1_added.parquet by country: "
      f"{a.c.value_counts().to_dict()}", flush=True)
