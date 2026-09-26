"""Direction audit: the decoy generator shifts house numbers UP (99.4% US / 92.3% India of small shifts) while true-match
noise is symmetric. Which other fields change in one direction for distractors but not for true matches?
Train records vs their top-1 retrieved S1, per country: for each signed field, among pairs where it is non-zero, the
share that is positive, for true matches and for distractors, and how many pairs.
Then the ceiling check: v10's validation errors (x/val_q_v10pw) by house-number shift direction.
  python x_asym.py"""
import numpy as np, pandas as pd
from common import WORK, load
from harness import truth_arrays, wscore
from match import normed

s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
c = c[c["rank"].values == 0]
r, s = c.rid.values, c.sid.values
n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
ints = lambda x: [int(t[:15]) for t in x.split()]
A = [ints(x) for x in n1.num.values[s]]; B = [ints(x) for x in n2.num.values[r]]
cnt = lambda col, idx, df: df[col].values[idx]
tok = lambda v: np.array([len(x.split()) for x in v])
F = {
    "first number shift": np.array([b[0] - a[0] if a and b and a[0] != b[0] else 0 for a, b in zip(A, B)]),
    "second number shift": np.array([b[1] - a[1] if len(a) > 1 and len(b) > 1 and a[1] != b[1] else 0 for a, b in zip(A, B)]),
    "numbers count": np.array([len(b) - len(a) for a, b in zip(A, B)]),
    "name tokens": tok(n2.nn.values[r]) - tok(n1.nn.values[s]),
    "core-name tokens": tok(n2.cn.values[r]) - tok(n1.cn.values[s]),
    "core-name chars": np.array([len(y) - len(x) for x, y in zip(n1.cn.values[s], n2.cn.values[r])]),
    "address tokens": tok(n2.na.values[r]) - tok(n1.na.values[s]),
    "zip shift": np.array([int(y) - int(x) if x and y and x != y else 0 for x, y in zip(n1.zip.values[s], n2.zip.values[r])]),
}
true = ts[r] == s
ctry = s1.country.values[s]
for name, v in F.items():
    for cc in ("US", "India"):
        row = []
        for lab, m in (("true", true), ("distractor", ~true)):
            k = m & (ctry == cc) & (v != 0)
            small = k & (np.abs(v) <= 13)
            row.append(f"{lab} n={k.sum():8d} up {np.mean(v[k] > 0):.3f} (|d|<=13: {np.mean(v[small] > 0) if small.any() else float('nan'):.3f})")
        print(f"{name:20s} {cc:5s}  " + "   ".join(row), flush=True)

# ceiling check: which of v10's validation errors have a small house-number shift, and in which direction
v = pd.read_parquet(f"{WORK}/x/val_q_v10pw.parquet")
Av = [ints(x) for x in n1.num.values[v.sid.values]]; Bv = [ints(x) for x in n2.num.values[v.rid.values]]
sh = np.array([b[0] - a[0] if a and b and a[0] != b[0] else 0 for a, b in zip(Av, Bv)])
acc = v.q.values >= 0.7
fp, fn = acc & ~v.y.values, ~acc & v.y.values
T = np.bincount(ts[ts >= 0], minlength=len(s1)); ents = np.where(s1f >= 8)[0]
base = wscore(v.sid.values, acc, v.y.values, v.wt.values, T, ents)
for lab, m in (("false positives (weighted)", fp), ("false negatives", fn)):
    w = v.wt.values[m]
    print(f"{lab}: {m.sum()} rows; small upward shift {np.sum(w[(sh[m] >= 1) & (sh[m] <= 13)]) / w.sum():.3f}, "
          f"small downward {np.sum(w[(sh[m] <= -1) & (sh[m] >= -13)]) / w.sum():.3f}", flush=True)
up = (sh >= 1) & (sh <= 13); down = (sh <= -1) & (sh >= -13)
print(f"validation F0.5 {base:.5f}; oracle fixing every error with a small upward shift "
      f"{wscore(v.sid.values, np.where(up, v.y.values, acc), v.y.values, v.wt.values, T, ents):.5f}; with a small downward "
      f"shift {wscore(v.sid.values, np.where(down, v.y.values, acc), v.y.values, v.wt.values, T, ents):.5f}", flush=True)
