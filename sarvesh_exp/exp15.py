"""E23 Occam check: test decoys are near-copies (same name, shifted house number; or same address, other name), train
distractors are not. Compare simple pair signatures in (A) train true matches, (B) train distractors vs the S1 they
imitate, (C) validation accepted pairs (US / India, q >= 0.70) split by truth, (D) test accepted pairs (E17 US / India,
France chain). If accepted test pairs carry a decoy signature far more often than true matches do, the excess is
decoys we accept: a precision lever."""
import os, numpy as np, pandas as pd
os.environ.setdefault("DFOLD", "1")
from common import load, WORK
from harness import truth_arrays
from match import normed
def nums(x): return [t for t in x.split()]
def sig(n1, n2, r, s):
    """per pair: num conflict (first numbers disagree both ways), small shift (|diff| <= 13), same core name, same addr"""
    R = [nums(x) for x in n2.num.values[r]]; S = [nums(x) for x in n1.num.values[s]]
    conf = np.array([bool(a) and bool(b) and a[0] not in b and b[0] not in a for a, b in zip(R, S)])
    shift = np.array([c and a[0].isdigit() and b[0].isdigit() and len(a[0]) < 9 and len(b[0]) < 9 and abs(int(a[0]) - int(b[0])) <= 13
                      for c, a, b in zip(conf, R, S)])
    name = n2.cn.values[r] == n1.cn.values[s]; addr = n2.na.values[r] == n1.na.values[s]
    return pd.DataFrame({"num_conflict": conf, "small_shift": shift, "same_name": name, "same_addr": addr,
                         "name_same+num_conflict": name & conf, "addr_same+name_diff": addr & ~name})
def show(title, df, by=None):
    if by is None: print(f"== {title}  (n={len(df):,})\n" + df.mean().to_frame().T.round(4).to_string(index=False), flush=True)
    else: print(f"== {title}\n" + df.groupby(by).mean().round(4).assign(n=df.groupby(by).size()).to_string(), flush=True)
s1, _, ts, s1f, rf = truth_arrays()
n1 = normed("train", 1); n2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
c1 = s1.country.values
m = np.flatnonzero(ts >= 0)
A = sig(n1, n2, m, ts[m]); show("A TRAIN true matches, by country", A.assign(c=c1[ts[m]]), "c")
imi = np.load(f"{WORK}/imitated_train.npy"); dz = np.flatnonzero((ts < 0) & (imi >= 0))
B = sig(n1, n2, dz, imi[dz]); show("B TRAIN distractors vs imitated S1, by country", B.assign(c=c1[imi[dz]]), "c")
v = pd.read_parquet("/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x/val_q_v10bpw.parquet")
va = v[v.q >= 0.70]
C = sig(n1, n2, va.rid.values, va.sid.values); show("C VALIDATION accepted (q >= 0.70), by truth", C.assign(true_match=va.y.values), "true_match")
t1 = load("test", 1); m1 = normed("test", 1); m2 = pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
X = f"{WORK}/x"
ui = pd.read_parquet(f"{X}/test_q_resc17.parquet"); ui = ui[(ui.c != "France") & (ui.q >= 0.70)]
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
acc = pd.concat([ui, fr[fr.q >= 0.70]], ignore_index=True)
D = sig(m1, m2, acc.rid.values, acc.sid.values); show("D TEST accepted (E17 file), by country", D.assign(c=acc.c.values), "c")
# France: the vetoed high-confidence records (base >= 0.70, chain < 0.70)
b = pd.read_parquet(f"{X}/test_q_v10plw.parquet"); b = b[b.c == "France"].reset_index(drop=True)
mm = b.merge(fr[["rid", "sid", "q"]], on=["rid", "sid"], how="left", suffixes=("", "_f"))
vet = mm[(mm.q >= 0.70) & (mm.q_f.fillna(0) < 0.70)]
E = sig(m1, m2, vet.rid.values, vet.sid.values); show("E FRANCE vetoed (base >= 0.70, final < 0.70)", E)
# test accepted US / India by q band
show("F TEST accepted US / India, by q band", D[acc.c.values != "France"].assign(
    band=pd.cut(acc.q.values[acc.c.values != "France"], [0.69, 0.8, 0.9, 0.99, 1.01]).astype(str)), "band")
