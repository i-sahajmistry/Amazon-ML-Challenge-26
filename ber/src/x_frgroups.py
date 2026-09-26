"""Hand-review dump of France entity groups: each S1 with every record whose best candidate it is, the final q of the
best file (v10_fr3_llm: x/test_q_v10fr3l), the pre-veto q (x/test_q_v10plw), and the record's runner-up S1 from the
shortlist. Samples random S1s, S1s with an unsure record, and S1s left empty. -> x/frgroups_{random,unsure,empty}.txt
  python x_frgroups.py [N=60] [COUNTRY=France]"""
import os, sys, numpy as np, pandas as pd
from common import WORK, load

XD = f"{WORK}/x"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
C = os.environ.get("COUNTRY", "France")
fin = pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet")
fin = fin[fin.c.values == C].reset_index(drop=True)
base = pd.read_parquet(f"{XD}/test_q_v10plw.parquet").set_index("rid").q
fin["q0"] = base.reindex(fin.rid.values).values
p5 = pd.read_parquet(f"{XD}/p5_test_v10.parquet")
p5 = p5[p5.rid.isin(fin.rid.values)].sort_values(["rid", "p"], ascending=[True, False])
second = p5[p5.duplicated("rid")].drop_duplicates("rid").set_index("rid")
fin["sid2"] = second.sid.reindex(fin.rid.values).fillna(-1).astype(int).values
fin["p2"] = second.p.reindex(fin.rid.values).fillna(0).values
s1 = load("test", 1)
o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
s1c = np.where(s1.country.values == C)[0]
g = {k: v for k, v in fin.groupby("sid")}
acc = fin.assign(a=fin.q.values >= 0.7).groupby("sid").a.sum()
rng = np.random.default_rng(0)
unsure = fin[(fin.q > 0.05) & (fin.q < 0.95) | (fin.q0 >= 0.7) & (fin.q < 0.7)].sid.unique()
empty = np.setdiff1d(s1c, acc[acc > 0].index.values)
print(f"{C}: S1 {len(s1c)}, records with a best S1 {len(fin)}, accepted {int(acc.sum())}, S1s with an unsure record "
      f"{len(unsure)}, empty S1s {len(empty)}", flush=True)


def show(sid):
    r = s1.iloc[sid]
    out = [f"S1 {r.entity_id} | {r.business_name} | {r.business_address}"]
    if sid in g:
        for x in g[sid].sort_values("q", ascending=False).itertuples():
            t = o.iloc[x.rid]
            mark = "ACC" if x.q >= 0.7 else ("VET" if x.q0 >= 0.7 else "rej")
            alt = f"  [2nd {s1.business_name.values[x.sid2][:40]} p={x.p2:.2f}]" if x.sid2 >= 0 and x.p2 > 0.05 else ""
            out.append(f"  {mark} q={x.q:.2f} q0={x.q0:.2f} {t.entity_id[:2]} | {t.business_name} | {t.business_address}{alt}")
    else:
        out.append("  (no record has this S1 as its best candidate)")
    return "\n".join(out)


for name, pool in (("random", s1c), ("unsure", unsure), ("empty", empty)):
    pick = rng.choice(pool, size=min(N, len(pool)), replace=False)
    with open(f"{XD}/frgroups_{C}_{name}.txt", "w") as f:
        f.write("\n\n".join(show(s) for s in pick) + "\n")
    print("wrote", f"{XD}/frgroups_{C}_{name}.txt", flush=True)
