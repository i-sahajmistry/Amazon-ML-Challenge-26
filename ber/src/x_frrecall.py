"""France recall: which France records do the self-training vetoes remove, and which removed / rejected ones look like
true matches by label-free structure? A distractor copies its S1's name with one edit and moves to a shifted house
number; true-match noise keeps the S1's number (or shifts it either way). Per word w that records add to their claimed
S1's core name: stay(w) = share of those records at the S1's first house number (label-free; the parallel session
measured corr 0.987 US / 0.965 India with the word's true-match rate). Candidate sets:
  R1: base stage 2 (with the LLM blend) accepts, a veto removes, same first house number, every added word stay >= 0.7
  R2: not accepted, no word added, a small downward shift (-13..-1): distractors shift up, true noise both ways
Writes x/restore_fr_{r1,r2}.parquet (rid, sid) for x_final.py RESTORE=.
  python x_frrecall.py"""
import numpy as np, pandas as pd
from collections import defaultdict
from common import WORK, load
from match import normed

XD = f"{WORK}/x"
d = pd.read_parquet(f"{XD}/test_q_v10plw.parquet")
d = d[d.c.values == "France"].reset_index(drop=True)
base = d.q.values >= 0.7
ok = {}
for t in ("_v9spw", "_v9s2pw", "_v10s3pw"):
    e = pd.read_parquet(f"{XD}/test_q{t}.parquet").set_index("rid").reindex(d.rid.values)
    ok[t] = (e.q.values >= 0.7) & (e.sid.values == d.sid.values)
final = base & ok["_v9spw"] & ok["_v9s2pw"] & ok["_v10s3pw"]
n1, n2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
ints = lambda x: [int(t[:15]) for t in x.split()]
A = [ints(x) for x in n1.num.values[d.sid.values]]; B = [ints(x) for x in n2.num.values[d.rid.values]]
same = np.array([bool(a) and bool(b) and a[0] == b[0] for a, b in zip(A, B)])
shift = np.array([b[0] - a[0] if a and b else 0 for a, b in zip(A, B)])
added = [set(y.split()) - set(x.split()) for x, y in zip(n1.cn.values[d.sid.values], n2.cn.values[d.rid.values])]
has = np.array([bool(a) and bool(b) for a, b in zip(A, B)])
cnt = defaultdict(lambda: [0, 0])
for w, s, h in zip(added, same, has):
    if h:
        for t in w:
            cnt[t][0] += 1; cnt[t][1] += s
stay = {t: v[1] / v[0] for t, v in cnt.items() if v[0] >= 30}
top = sorted(((v, t, cnt[t][0]) for t, v in stay.items() if cnt[t][0] >= 300), key=lambda x: x[0])
print("France words added (n>=300), lowest stay:", ", ".join(f"{t} {v:.2f}/{n}" for v, t, n in top[:12]))
print("                            highest stay:", ", ".join(f"{t} {v:.2f}/{n}" for v, t, n in top[-12:]), flush=True)
filler = np.array([all(stay.get(t, 0.0) >= 0.7 for t in w) for w in added])     # no added word counts as filler

vet = base & ~final
print(f"France records {len(d)}; base accepts {base.sum()}; after vetoes {final.sum()}; vetoed {vet.sum()}", flush=True)
for t in ok:
    print(f"  removed by {t}: {(base & ~ok[t]).sum()}  (only by it: {(base & ~ok[t] & np.all([ok[u] for u in ok if u != t], axis=0)).sum()})")
print(f"  vetoed at the same first number: {(vet & same).sum()}, of which only filler-like / no words added: {(vet & same & filler).sum()}")
# same street: the addresses without their numbers must match (France has many same-name S1s on other streets)
import re
from rapidfuzz import process, fuzz
strip = lambda v: [re.sub(r"\d+", " ", x) for x in v]
street = process.cpdist(strip(n1.na.values[d.sid.values]), strip(n2.na.values[d.rid.values]), scorer=fuzz.token_set_ratio,
                        workers=-1) >= 85
print(f"same street among vetoed same-number records: {(vet & same & street).sum()} of {(vet & same).sum()}", flush=True)
r1 = vet & same & street & filler
r2 = ~final & ~base & has & street & (shift <= -1) & (shift >= -13) & np.array([not w for w in added])
for name, m in (("R1 vetoed, same number, filler words", r1), ("R2 rejected, no word added, small downward shift", r2)):
    q = d.q.values[m]
    print(f"{name}: {m.sum()} records over {len(np.unique(d.sid.values[m]))} S1s; base q quartiles "
          f"{np.round(np.percentile(q, [25, 50, 75]), 3).tolist() if m.any() else []}", flush=True)
s1t = load("test", 1); o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
t = lambda df, i: f"{df.business_name.values[i]} | {df.business_address.values[i]}"
rng = np.random.default_rng(5)
for name, m in (("R1", r1), ("R2", r2)):
    print(f"=== {name} samples")
    for i in rng.choice(np.flatnonzero(m), min(15, m.sum()), replace=False):
        print(f"  {d.q.values[i]:.2f} +{sorted(added[i])}  S1 {t(s1t, d.sid.values[i])}\n        REC {t(o, d.rid.values[i])}")
    d.loc[m, ["rid", "sid"]].to_parquet(f"{XD}/restore_fr_{name.lower()}.parquet")
print("wrote", f"{XD}/restore_fr_r1.parquet", f"{XD}/restore_fr_r2.parquet", flush=True)
