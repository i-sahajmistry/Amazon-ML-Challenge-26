"""Hedge for the suf7 restore (x_frfix.py): the same records minus those adding groupe / developpement, the only suffix
words the LLM judge rejects (rule1 SUF7 no-rate 0.83 / 0.98 vs 0.01-0.23 for fils / associes / et / france; x_frllm.py).
The judge learned group/holdings = decoy in the US, so it may be biased there; the census says they are true.
Writes x/restore_fr_suf7ng.parquet for x_final.py RESTORE=.
  python x_suf7ng.py"""
import pandas as pd
from x_llm import XD

r = pd.read_parquet(f"{XD}/restore_fr_suf7.parquet")[["rid", "sid"]]
f = pd.read_parquet(f"{XD}/fam_France.parquet")[["rid", "sid", "add"]]
m = r.merge(f, on=["rid", "sid"], how="left")
grp = m["add"].fillna("").str.split().apply(lambda a: bool({"groupe", "developpement"} & set(a))).values
m[~grp][["rid", "sid"]].to_parquet(f"{XD}/restore_fr_suf7ng.parquet")
print(f"suf7 {len(r)}: dropped {grp.sum()} adding groupe / developpement, kept {(~grp).sum()}", flush=True)
