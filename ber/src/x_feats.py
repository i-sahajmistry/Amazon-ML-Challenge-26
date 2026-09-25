"""Extra pair features aimed at how distractors are made (see x_errors / x_dgroups logs):
- integer-aware address numbers: "0044" == "44" is noise, "58" vs "59" is a different business;
- legal-form edits: "Private Limited" -> "LLP" or "Limited" -> "Pvt Ltd" marks a distractor, dropped words are noise.
  python x_feats.py train|test   -> x/extra_{split}.parquet, rows in feats2 order"""
import sys, math, numpy as np, pandas as pd
from multiprocessing import Pool
from common import WORK
from match import normed

LEGAL = {"private": 1, "praivet": 1, "praibhet": 1, "prayvet": 1, "limited": 2, "limitet": 2, "limted": 2, "lmtd": 2,
         "llp": 4, "elelpi": 4, "elelpee": 4, "llc": 8, "incorporated": 16, "corporation": 32, "company": 64, "cie": 64,
         "pc": 128, "pllc": 128, "lp": 256, "plc": 256, "partners": 256, "sarl": 512, "sas": 512, "sasu": 512,
         "eurl": 512, "sa": 512, "sci": 512, "snc": 512, "holdings": 1024, "group": 1024, "center": 2048,
         "centre": 2048, "services": 4096, "enterprises": 4096, "trust": 8192, "foundation": 8192, "society": 8192}
POP = np.array([bin(i).count("1") for i in range(1 << 14)], np.int8)
_G = {}


def legal_bits(nn):
    b = 0
    for t in nn.split():
        b |= LEGAL.get(t, 0)
    s = f" {nn} "
    return b | (4 if " l l p " in s else 0) | (8 if " l l c " in s else 0)


def ints(s):
    return tuple(int(x) for x in s.split()[:8]) if s else ()


def _chunk(i):
    lo, hi = _G["job"][i]
    out = np.empty((hi - lo, 6), np.float32)
    IA, IB = _G["ia"], _G["ib"]
    for j, (x, y) in enumerate(zip(_G["sa"][lo:hi], _G["rb"][lo:hi])):
        A, B = IA[x], IB[y]
        if A and B:
            a0, SA, SB = A[0], set(A), set(B)
            out[j] = (a0 == B[0], a0 in SB, math.log1p(min(abs(a0 - b) for b in B)),
                      len(SA & SB) / len(SA | SB), len(SA - SB), len(SB - SA))
        else:
            out[j] = (-1, -1, -1, -1, len(A), len(B))
    return out


def main(split):
    keys = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"])
    n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    la = n1.nn.map(legal_bits).values.astype(np.int32)[keys.sid.values]
    lb = n2.nn.map(legal_bits).values.astype(np.int32)[keys.rid.values]
    _G.update(ia=n1.num.map(ints).tolist(), ib=n2.num.map(ints).tolist(), sa=keys.sid.values, rb=keys.rid.values)
    step = len(keys) // 2048 + 1
    _G["job"] = [(i, min(i + step, len(keys))) for i in range(0, len(keys), step)]
    with Pool(32) as p:
        I = np.concatenate(p.map(_chunk, range(len(_G["job"]))))
    out = pd.DataFrame(I, columns=["int_first_eq", "int_a0_in_b", "int_a0_mindiff", "int_jacc", "int_a_only", "int_b_only"])
    out["legal_added"] = POP[lb & ~la]
    out["legal_dropped"] = POP[la & ~lb]
    out["legal_a"], out["legal_b"], out["legal_xor"] = la, lb, la ^ lb
    out.astype(np.float32).to_parquet(f"{WORK}/x/extra_{split}.parquet")
    print(split, out.shape, out.describe().T[["mean", "min", "max"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
