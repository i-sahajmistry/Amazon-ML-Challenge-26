"""Naive-Bayes edit features. For each pair, which name / address tokens are added or dropped between the S1 entity
and the record, scored by how much more often that edit appears on true pairs than on distractor pairs
(log-likelihood ratio). The generator adds filler words ("center", "partners") as noise but swaps real words
("consulting" -> "consultants", "holdings") to make distractors; typos fall into a rare-token class, digits into
<num> (numbers have their own features in x_feats). Tables use S1 folds 0-3 only, like the CE.
  python x_llr.py   -> x/llr_train.parquet, x/llr_test.parquet (feats2 row order)"""
import numpy as np, pandas as pd
from collections import Counter
from multiprocessing import Pool
from common import WORK
from match import normed
from harness import truth_arrays

RARE = 5
_G = {}
COLS = [f"{f}_{e}_{s}" for f in ("n", "a") for e in ("add", "drop") for s in ("sum", "min", "max", "cnt")]


def _vocab_chunk(i):
    c = Counter()
    for x in _G["texts"][i]:
        c.update(set(x.split()))
    return c


def canon(texts, freq):
    """token sets with digits -> <num>, rare tokens -> <rare>"""
    out = []
    for x in texts:
        s = set()
        for t in x.split():
            s.add("<num>" if t.isdigit() else t if freq.get(t, 0) >= RARE else "<rare>")
        out.append(frozenset(s))
    return out


def _canon_chunk(i):
    return canon(_G["texts"][i], _G["freq"])


def token_sets(texts, freq=None):
    step = len(texts) // 256 + 1
    _G["texts"] = [texts[i:i + step] for i in range(0, len(texts), step)]
    _G["freq"] = freq                      # set before the fork so workers see it
    with Pool(32) as p:
        if freq is None:
            c = Counter()
            for x in p.imap_unordered(_vocab_chunk, range(len(_G["texts"]))):
                c.update(x)
            return c
        return [s for part in p.map(_canon_chunk, range(len(_G["texts"]))) for s in part]


def _count_chunk(i):
    lo, hi = _G["job"][i]
    c = {k: Counter() for k in ("n_add", "n_drop", "a_add", "a_drop")}
    for x, y in zip(_G["sa"][lo:hi], _G["rb"][lo:hi]):
        for f in ("n", "a"):
            A, B = _G[f + "1"][x], _G[f + "2"][y]
            c[f + "_add"].update(B - A)
            c[f + "_drop"].update(A - B)
    return c


def _feat_chunk(i):
    lo, hi = _G["job"][i]
    out = np.zeros((hi - lo, len(COLS)), np.float32)
    for j, (x, y) in enumerate(zip(_G["sa"][lo:hi], _G["rb"][lo:hi])):
        k = 0
        for f in ("n", "a"):
            A, B = _G[f + "1"][x], _G[f + "2"][y]
            for e, toks in (("add", B - A), ("drop", A - B)):
                tab, dflt = _G["llr"][f + "_" + e]
                v = [tab.get(t, dflt) for t in toks]
                out[j, k:k + 4] = (sum(v), min(v), max(v), len(v)) if v else (0.0, 0.0, 0.0, 0.0)
                k += 4
    return out


def run(fn, n_pairs, chunks=2048):
    step = n_pairs // chunks + 1
    _G["job"] = [(i, min(i + step, n_pairs)) for i in range(0, n_pairs, step)]
    with Pool(32) as p:
        return p.map(fn, range(len(_G["job"])))


def load_sets(split, freq):
    n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    return {"n1": token_sets(n1.nn.tolist(), freq["n"]), "a1": token_sets(n1.na.tolist(), freq["a"]),
            "n2": token_sets(n2.nn.tolist(), freq["n"]), "a2": token_sets(n2.na.tolist(), freq["a"])}


def main():
    _, _, ts, s1f, rf = truth_arrays()
    # token frequencies over train + test text (no labels), so test-only vocabulary (France) is not all <rare>
    alln = pd.concat([normed(sp, src) for sp in ("train", "test") for src in (1, 2, 3)], ignore_index=True)
    freq = {"n": token_sets(alln.nn.tolist()), "a": token_sets(alln.na.tolist())}
    del alln
    _G.update(load_sets("train", freq))
    k = pd.read_parquet(f"{WORK}/feats2_train.parquet", columns=["rid", "sid", "rank"])
    r, s, rank = k.rid.values, k.sid.values, k["rank"].values
    fit = rf[r] < 4
    pos = fit & (ts[r] == s)
    neg = fit & (ts[r] < 0) & (rank == 0)          # a distractor against the S1 it imitates
    tables = {}
    for nm, msk in (("pos", pos), ("neg", neg)):
        _G.update(sa=s[msk], rb=r[msk])
        parts = run(_count_chunk, int(msk.sum()), 256)
        tables[nm] = {key: sum((pt[key] for pt in parts), Counter()) for key in parts[0]}
        tables[nm]["N"] = int(msk.sum())
    llr = {}
    for key in ("n_add", "n_drop", "a_add", "a_drop"):
        P, N = tables["pos"][key], tables["neg"][key]
        Np, Nn = tables["pos"]["N"], tables["neg"]["N"]
        toks = set(P) | set(N)
        llr[key] = ({t: float(np.log((P[t] + 1) / (Np + 2)) - np.log((N[t] + 1) / (Nn + 2))) for t in toks}, 0.0)
        top = sorted(llr[key][0].items(), key=lambda kv: kv[1])
        print(key, "most distractor-like:", [(t, round(v, 2)) for t, v in top[:12] if P[t] + N[t] > 200],
              "\n   most noise-like:", [(t, round(v, 2)) for t, v in top[::-1][:12] if P[t] + N[t] > 200], flush=True)
    _G["llr"] = llr
    for split in ("train", "test"):
        if split == "test":
            _G.update(load_sets("test", freq))
            k = pd.read_parquet(f"{WORK}/feats2_test.parquet", columns=["rid", "sid"])
        _G.update(sa=k.sid.values, rb=k.rid.values)
        F = pd.DataFrame(np.concatenate(run(_feat_chunk, len(k))), columns=COLS)
        F.to_parquet(f"{WORK}/x/llr_{split}.parquet")
        print(split, F.shape, flush=True)


if __name__ == "__main__":
    main()
