"""Independent check of the France same-address sets (x_frfix.py) with the LLM judge's margins, which were computed
for every unsure row (0.01 < q < 0.99) before those sets existed: original LoRA (x/llm_test_v10p) and the France LoRA
(x/llm_test_v10p_fr). Calibrated on US / India validation band rows: P(no | true), P(no | not true), so a set's
"no" rate r implies a true share x = (b - r) / (b - a), if the judge's error rates carried over to France.
  python x_frllm.py"""
import numpy as np, pandas as pd
from x_llm import XD
from harness import truth_arrays

v = pd.read_parquet(f"{XD}/llm_val_v10p.parquet")
v = v[np.isfinite(v.llm.values)]
s1, other, ts, s1f, rf = truth_arrays()
y = ts[v.rid.values] == v.sid.values
a, b = (v.llm.values[y] < 0).mean(), (v.llm.values[~y] < 0).mean()
print(f"val band rows {len(v)}: true {y.sum()}, not true {(~y).sum()}; P(no|true) {a:.3f}  P(no|not true) {b:.3f}", flush=True)

o = pd.read_parquet(f"{XD}/llm_test_v10p.parquet")[["rid", "sid", "q", "c", "llm"]]
f = pd.read_parquet(f"{XD}/llm_test_v10p_fr.parquet")[["rid", "sid", "llm"]].rename(columns={"llm": "llm_fr"})
t = o[o.c == "France"].merge(f, on=["rid", "sid"], how="left")
sets = {"France band, all": t[["rid", "sid"]],
        "France band, q >= 0.7": t[t.q.values >= 0.7][["rid", "sid"]],
        "France band, q < 0.7": t[t.q.values < 0.7][["rid", "sid"]]}
for n in ("restore_fr_suf7", "reject_fr_type", "restore_fr_dd", "reject_fr_dd", "restore_fr_r1"):
    try:
        sets[n] = pd.read_parquet(f"{XD}/{n}.parquet")[["rid", "sid"]]
    except FileNotFoundError:
        pass
for n, s in sets.items():
    m = s.merge(t, on=["rid", "sid"], how="left")
    k = np.isfinite(m.llm.values)
    r = (m.llm.values[k] < 0).mean() if k.any() else np.nan
    rf_ = (m.llm_fr.values[k & np.isfinite(m.llm_fr.values)] < 0).mean() if k.any() else np.nan
    x = (b - r) / (b - a)
    print(f"{n:24s} n {len(s):7d}  with LLM {k.sum():7d}  stage q med {np.nanmedian(m.q.values):.2f}  "
          f"no-rate {r:.3f} (France LoRA {rf_:.3f})  margin med {np.nanmedian(m.llm.values):+.2f}  implied true {x:.2f}",
          flush=True)

# rule1 (LB: ~57.5% true) by the hand-review families (x_families.py), and SUF7 by added word
from x_census import R1
fam = pd.read_parquet(f"{XD}/fam_France.parquet")[["rid", "sid", "fam", "add"]]
r1 = pd.read_parquet(R1)[["rid", "sid"]].merge(fam, on=["rid", "sid"], how="left").merge(t, on=["rid", "sid"], how="left")
k = np.isfinite(r1.llm.values)


def row(n, m):
    m = m & k
    print(f"{n:34s} n {m.sum():6d}  no-rate {(r1.llm.values[m] < 0).mean():.3f}  "
          f"margin med {np.median(r1.llm.values[m]):+.2f}  stage q med {np.median(r1.q.values[m]):.2f}", flush=True)


print(f"\nrule1 {len(r1)} records, {k.sum()} with an LLM margin", flush=True)
row("rule1, all", np.ones(len(r1), bool))
for fm in r1.fam.fillna("?").unique():
    row(f"rule1, fam {fm}", (r1.fam.fillna("?") == fm).values)
s = (r1.fam == "SUF7").values
for w in ("fils", "associes", "et", "cie", "services", "groupe", "developpement", "france"):
    row(f"rule1, SUF7 adding '{w}'", s & r1["add"].fillna("").str.split().apply(lambda a: w in a).values)
