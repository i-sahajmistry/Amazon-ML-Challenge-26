"""E39 France over-acceptance check (label-free). Per S1: claimants (records whose best S1 it is), accepted, rejected,
by country, for E16fr3-like decisions (US / India: E17 q; France: chain + restore / reject). If decoys per S1 are fixed
by the generator (the census' premise), rejected claimants per S1 should match across countries; a France shortfall is
decoys we accept. Then: which kinds of accepted France records look like the rejected US / India claimants
(same address + name edit, number shift, added word, legal form change, no number...). Also train truth: decoys (true
distractors) per S1 by country and their kind mix, and US / India test rejected-claimant kinds for comparison."""
import numpy as np, pandas as pd
from common import load, WORK
from match import normed
X = f"{WORK}/x"
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
ui = pd.read_parquet(f"{X}/test_q_resc17.parquet"); ui = ui[ui.c != "France"]
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
d = pd.concat([ui, fr], ignore_index=True)
n = len(t1); c1 = t1.country.values
acc = d.q.values >= 0.70
cl = np.bincount(d.sid.values, minlength=n); ac = np.bincount(d.sid.values[acc], minlength=n)
n_rec = pd.Series(pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values).value_counts()
print("per S1 (test): records, claimants, accepted, rejected claimants, records with no row", flush=True)
for c in ("US", "India", "France"):
    k = c1 == c; ns = k.sum()
    print(f"  {c:7s} records {n_rec[c] / ns:.3f}  claimants {cl[k].mean():.3f}  accepted {ac[k].mean():.3f}  rejected {cl[k].mean() - ac[k].mean():.3f}  "
          f"unclaimed {(n_rec[c] - cl[k].sum()) / ns:.3f}", flush=True)
def first(x): return x.split()[0] if x else ""
def kinds(r, s, M1, M2):
    a = np.array([first(x) for x in M2.num.values[r]]); b = np.array([first(x) for x in M1.num.values[s]])
    same_addr = M2.na.values[r] == M1.na.values[s]; same_name = M2.cn.values[r] == M1.cn.values[s]
    num_diff = (a != "") & (b != "") & (a != b); no_addr = M2.na.values[r] == ""
    return pd.Series(np.select([no_addr & same_name, no_addr, same_addr & same_name, same_addr, num_diff & same_name, num_diff, same_name],
                               ["no-addr same-name", "no-addr name-edit", "same-addr same-name", "same-addr name-edit",
                                "num-shift same-name", "num-shift name-edit", "other-addr same-name"], "other-addr name-edit"))
for c in ("US", "India", "France"):
    k = d.c.values == c
    A = kinds(d.rid.values[k & acc], d.sid.values[k & acc], m1, m2).value_counts(normalize=True)
    R = kinds(d.rid.values[k & ~acc], d.sid.values[k & ~acc], m1, m2).value_counts(normalize=True)
    ns = (c1 == c).sum()
    print(f"== {c}: kinds per S1 (accepted | rejected claimants)", flush=True)
    for kk in sorted(set(A.index) | set(R.index)):
        print(f"   {kk:22s} acc/S1 {A.get(kk, 0) * (k & acc).sum() / ns:.3f}   rej/S1 {R.get(kk, 0) * (k & ~acc).sum() / ns:.3f}", flush=True)
# train truth: distractors per S1 and their kind vs the S1 they imitate
import os
os.environ["DFOLD"] = "1"
from harness import truth_arrays
s1, _, ts, s1f, rf = truth_arrays()
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
imi = np.load(f"{WORK}/imitated_train.npy")
for c in ("US", "India"):
    k = s1.country.values == c; ns = k.sum()
    dz = np.flatnonzero((ts < 0) & (imi >= 0)); dz = dz[s1.country.values[imi[dz]] == c]
    tm = np.flatnonzero(ts >= 0); tm = tm[s1.country.values[ts[tm]] == c]
    print(f"== TRAIN {c}: true matches/S1 {len(tm) / ns:.3f}, distractors/S1 {len(dz) / ns:.3f}", flush=True)
    print("   distractor kinds/S1: " + ", ".join(f"{a} {b * len(dz) / ns:.3f}" for a, b in kinds(dz, imi[dz], n1, n2).value_counts(normalize=True).items()), flush=True)
    print("   true-match kinds/S1: " + ", ".join(f"{a} {b * len(tm) / ns:.3f}" for a, b in kinds(tm, ts[tm], n1, n2).value_counts(normalize=True).items()), flush=True)
