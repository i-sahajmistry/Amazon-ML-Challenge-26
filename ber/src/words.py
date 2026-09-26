"""Distractor-word features, the same procedure for every country label (replaces a hand-written word list).

Distractors are made by editing a real entity's name, often by ADDING a word (train: holdings, group, north,
downtown, industries, india...). Which words play that role differs by country and language, so each word is scored
from label-free statistics of its own country plus its multilingual embedding:
  1. tokens: for every record, the normalised-name tokens it adds relative to its top-1 retrieved S1 (per split and
     country): N, A (added), add_rate, s1_rate, share of pure insertions (no similar token dropped), end/start share.
  2. word model: LightGBM on those statistics plus a k-NN score over the multilingual e5 embedding ("groupe" sits
     next to "group"), target = share of a word's added occurrences on non-matching records (train labels).
     Leave-one-country-out: a train country's words are scored by a model fitted on the OTHER train countries, and
     a country missing from train by a model fitted on all of them, so train features are as uncertain as test ones.
  3. pair features (feats2 row order): over the tokens a record adds / drops relative to the candidate S1:
     max, sum and count (>= 0.8) of p_bad, max add_rate, max p_bad of dropped tokens, counts.
  python words.py   -> work/x/words_{train,test}.parquet, work/dicts/words.parquet"""
import os, numpy as np, pandas as pd, lightgbm as lgb
from collections import Counter
from multiprocessing import get_context
from rapidfuzz.distance import JaroWinkler
from common import WORK, load
from match import normed
from harness import truth_arrays

XD = f"{WORK}/x"
NP = int(os.environ.get("NPROC", 16))
MIN_N, MIN_A = 20, 50
STAT = ["logN", "logA", "add_rate", "s1_rate", "ins_share", "end_share", "start_share", "len", "digit"]
WORDF = ["w_add_max", "w_add_sum", "w_add_n8", "w_addrate_max", "w_drop_max", "w_n_add", "w_n_drop"]
_G = {}


def _pool():
    return get_context("fork").Pool(NP)


def _count(b):
    rn, sn = _G["rn"], _G["sn"]
    N, A, ins, end, start = Counter(), Counter(), Counter(), Counter(), Counter()
    for i in range(*b):
        r = rn[i].split(); R, S = set(r), set(sn[i].split())
        N.update(R)
        dropped = S - R
        for j, t in enumerate(r):
            if t in S:
                continue
            A[t] += 1
            ins[t] += not any(JaroWinkler.similarity(t, d) >= 0.8 for d in dropped)
            end[t] += j == len(r) - 1
            start[t] += j == 0
    return N, A, ins, end, start


def token_stats(split):
    """one row per (country, token) of records' normalised names; 'bad' on train."""
    n1, n2 = normed(split, 1), pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c1 = load(split, 1).country.values
    c2 = pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values
    k = pd.read_parquet(f"{WORK}/cand_{split}.parquet", columns=["rid", "sid", "rank"])
    k = k[k["rank"] == 0]
    rid, sid = k.rid.values, k.sid.values
    bad = truth_arrays()[2][rid] != sid if split == "train" else None
    out = []
    for c in sorted(set(c2[rid])):
        m = c2[rid] == c
        _G.update(rn=n2.nn.values[rid[m]], sn=n1.nn.values[sid[m]])
        n = int(m.sum()); step = n // (NP * 8) + 1
        with _pool() as p:
            parts = p.map(_count, [(i, min(i + step, n)) for i in range(0, n, step)])
        N, A, ins, end, start = (sum((x[j] for x in parts), Counter()) for j in range(5))
        S = Counter(t for s in n1.nn.values[c1 == c] for t in set(s.split()))
        toks = [t for t, v in N.items() if v >= MIN_N and A[t] >= MIN_A]
        d = pd.DataFrame({"tok": toks, "N": [N[t] for t in toks], "A": [A[t] for t in toks], "S": [S[t] for t in toks],
                          "ins": [ins[t] for t in toks], "end": [end[t] for t in toks], "start": [start[t] for t in toks]})
        if bad is not None:   # labelled target: share of the added occurrences on records that are not this S1's
            _G.update(badm=bad[m])
            with _pool() as p:
                D = sum(p.map(_bad_count, [(i, min(i + step, n)) for i in range(0, n, step)]), Counter())
            d["bad"] = [D[t] / A[t] for t in toks]
        d["add_rate"], d["s1_rate"] = d.A / d.N, d.S / max((c1 == c).sum(), 1)
        d["ins_share"], d["end_share"], d["start_share"] = d.ins / d.A, d.end / d.A, d.start / d.A
        d["logN"], d["logA"] = np.log1p(d.N / n * 1e6), np.log1p(d.A / n * 1e6)
        d["len"], d["digit"] = d.tok.str.len(), d.tok.str.contains(r"\d").astype(float)
        out.append(d.assign(country=c, split=split))
        print(split, c, "records", n, "scored tokens", len(d), flush=True)
    return pd.concat(out, ignore_index=True)


def _bad_count(b):
    rn, sn, badm = _G["rn"], _G["sn"], _G["badm"]
    D = Counter()
    for i in range(*b):
        if badm[i]:
            D.update(set(rn[i].split()) - set(sn[i].split()))
    return D


def embed(tokens):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tk = AutoTokenizer.from_pretrained(f"{WORK}/e5_ft"); m = AutoModel.from_pretrained(f"{WORK}/e5_ft").cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tokens), 2048):
            b = tk([f"query: {t}" for t in tokens[i:i + 2048]], padding=True, truncation=True, max_length=16,
                   return_tensors="pt").to("cuda")
            h = m(**b).last_hidden_state
            e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def knn_bad(Eq, Er, bad, A, k=10, self_ref=False):
    S = Eq @ Er.T
    if self_ref:
        np.fill_diagonal(S, -1.0)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, 1)
    w = np.clip(s, 0, None) ** 4 * np.log1p(A[idx])
    return (w * bad[idx]).sum(1) / np.maximum(w.sum(1), 1e-9), s.max(1)


def word_scores(T, E):
    """p_bad for every (split, country, token): leave-one-country-out over the train countries."""
    tr = (T.split.values == "train")
    train_c = sorted(set(T.country[tr]))
    T["p_bad"] = np.nan
    for c in sorted(set(T.country)):
        ref = tr & (T.country.values != c) if c in train_c else tr
        q = T.country.values == c
        a = T[ref].copy()
        a["knn_bad"], a["knn_sim"] = knn_bad(E[ref], E[ref], a.bad.values, a.A.values, self_ref=True)
        b = T[q].copy()
        b["knn_bad"], b["knn_sim"] = knn_bad(E[q], E[ref], a.bad.values, a.A.values)
        X = STAT + ["knn_bad", "knn_sim"]
        m = lgb.train(dict(objective="cross_entropy", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                           verbose=-1, seed=0), lgb.Dataset(a[X], a.bad.values, weight=np.log1p(a.A.values)), 300)
        T.loc[q, "p_bad"] = m.predict(b[X])
        T.loc[q, "knn_bad"] = b.knn_bad.values
        print(f"word model for {c}: fitted on {sorted(set(a.country))}, "
              f"top words: {', '.join(T[q].sort_values('A', ascending=False).query('p_bad >= 0.8').tok.head(12))}",
              flush=True)
    return T


def _pair(b):
    lo, hi = b
    sa, rb, a_nn, b_nn, ctry, P, R = _G["sa"], _G["rb"], _G["a_nn"], _G["b_nn"], _G["ctry"], _G["P"], _G["R"]
    out = np.zeros((hi - lo, len(WORDF)), np.float32)
    for j in range(lo, hi):
        S, Rt = set(a_nn[sa[j]].split()), set(b_nn[rb[j]].split())
        c = ctry[rb[j]]
        pa = [P.get((c, t), np.nan) for t in Rt - S]
        pa = [x for x in pa if x == x]
        ra = [R.get((c, t), 0.0) for t in Rt - S]
        pd_ = [x for x in (P.get((c, t), np.nan) for t in S - Rt) if x == x]
        out[j - lo] = (max(pa, default=0.0), sum(pa), sum(x >= 0.8 for x in pa), max(ra, default=0.0),
                       max(pd_, default=0.0), len(Rt - S), len(S - Rt))
    return out


def pair_features(split, T):
    keys = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"])
    t = T[T.split == split]
    _G.update(sa=keys.sid.values, rb=keys.rid.values, a_nn=normed(split, 1).nn.values,
              b_nn=pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True).nn.values,
              ctry=pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values,
              P=dict(zip(zip(t.country, t.tok), t.p_bad)), R=dict(zip(zip(t.country, t.tok), t.add_rate)))
    n = len(keys); step = n // (NP * 16) + 1
    with _pool() as p:
        W = np.concatenate(p.map(_pair, [(i, min(i + step, n)) for i in range(0, n, step)]))
    pd.DataFrame(W, columns=WORDF).to_parquet(f"{XD}/words_{split}.parquet")
    print(split, "pair word features", W.shape, "mean", W.mean(0).round(3).tolist(), flush=True)


def main():
    os.makedirs(XD, exist_ok=True); os.makedirs(f"{WORK}/dicts", exist_ok=True)
    T = pd.concat([token_stats("train"), token_stats("test")], ignore_index=True)
    T = word_scores(T, embed(T.tok.tolist()))
    T.drop(columns=["ins", "end", "start"]).to_parquet(f"{WORK}/dicts/words.parquet")
    for sp in ("train",) if os.environ.get("SKIP_TEST") else ("train", "test"):   # SKIP_TEST: train only
        pair_features(sp, T)


if __name__ == "__main__":
    main()
