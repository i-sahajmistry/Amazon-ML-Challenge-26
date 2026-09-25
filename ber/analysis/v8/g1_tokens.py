"""Label-free "added word" statistics per country: can the data itself find the distractor words that v7w's
hand-written France list (groupe, holding, france, ...) encodes?

For each S2/S3 record, take its top-1 retrieved S1 and the set of name tokens the record ADDS (in the record's
normalised name, not in the S1's). Per (split, country) and token t:
  N(t)  records whose name contains t
  A(t)  records where t is added            -> add_rate = A / N   (label-free, computable on test)
  S(t)  S1 names containing t               -> s1_rate = S / #S1 (distractor words are rare in S1 names)
Train only, from labels:
  D(t)  added occurrences on records that are NOT a true match of that top-1 S1 -> bad = D / A
Writes gen/tokens_{split}_{country}.parquet and prints the top tokens by A among add_rate >= 0.5.
  python g1_tokens.py"""
import os, numpy as np, pandas as pd
from collections import Counter
from multiprocessing import get_context
Pool = get_context("fork").Pool  # py3.14 defaults to forkserver; workers must inherit _G
from common import WORK, load
from match import normed
from harness import truth_arrays

OUT = os.environ.get("GEN_DIR", f"{WORK}/gen")
os.makedirs(OUT, exist_ok=True)
_G = {}


def _count(bounds):
    lo, hi = bounds
    rn, sn, bad = _G["rn"], _G["sn"], _G["bad"]
    N, A, D = Counter(), Counter(), Counter()
    for i in range(lo, hi):
        R = set(rn[i].split())
        add = R - set(sn[i].split())
        N.update(R)
        A.update(add)
        if bad is not None and bad[i]:
            D.update(add)
    return N, A, D


def run(split):
    n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c1 = load(split, 1).country.values
    c2 = pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values
    k = pd.read_parquet(f"{WORK}/cand_{split}.parquet", columns=["rid", "sid", "rank"])
    k = k[k["rank"] == 0]
    rid, sid = k.rid.values, k.sid.values
    bad = None
    if split == "train":
        ts = truth_arrays()[2]
        bad = ts[rid] != sid
    for c in sorted(set(c2[rid])):
        m = c2[rid] == c
        _G.update(rn=n2.nn.values[rid[m]], sn=n1.nn.values[sid[m]], bad=None if bad is None else bad[m])
        n = int(m.sum())
        step = n // 256 + 1
        with Pool(16) as p:
            parts = p.map(_count, [(i, min(i + step, n)) for i in range(0, n, step)])
        N, A, D = Counter(), Counter(), Counter()
        for a, b, d in parts:
            N.update(a); A.update(b); D.update(d)
        S = Counter(t for s in n1.nn.values[c1 == c] for t in set(s.split()))
        toks = [t for t, v in N.items() if v >= 20]
        df = pd.DataFrame({"tok": toks, "N": [N[t] for t in toks], "A": [A[t] for t in toks],
                           "S": [S[t] for t in toks]})
        df["add_rate"] = df.A / df.N
        df["s1_rate"] = df.S / max((c1 == c).sum(), 1)
        if bad is not None:
            df["D"] = [D[t] for t in toks]
            df["bad"] = df.D / df.A.clip(lower=1)
        df.to_parquet(f"{OUT}/tokens_{split}_{c}.parquet")
        top = df[df.add_rate >= 0.5].sort_values("A", ascending=False).head(40)
        print(f"\n== {split} {c}: records {n}, tokens {len(df)}; top added tokens (add_rate >= 0.5)", flush=True)
        print(top.round(4).to_string(index=False), flush=True)
        if bad is not None:
            # does the label-free add_rate separate distractor words from benign ones?
            big = df[df.A >= 200]
            for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
                s = big[(big.add_rate >= lo) & (big.add_rate < hi)]
                print(f"  add_rate [{lo},{hi}): tokens {len(s)}, added occurrences {s.A.sum()}, "
                      f"share on wrong/unmatched records {s.D.sum() / max(s.A.sum(), 1):.3f}", flush=True)


if __name__ == "__main__":
    for sp in ("train", "test"):
        run(sp)
