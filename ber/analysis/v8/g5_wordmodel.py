"""A country-agnostic "distractor word" model to replace x_final.py's hand-written French list.

Unit = (country, token) for name tokens that records ADD relative to their top-1 retrieved S1 (g1_tokens.py).
Target (train only) = bad: share of those added occurrences on records that are not a true match of that S1.
Label-free token features, computed the same way in every country:
  stats  log N, log A, add_rate, s1_rate, token length, digit flag
  shape  of the added occurrences: share that are INSERTIONS (no similar dropped token in the pair, e.g. "holdings")
         vs SUBSTITUTIONS (a similar token was dropped: "praivet" for "private"), and share at the name's end/start
  emb    the multilingual bi-encoder's embedding of the token (so "groupe" can inherit what "group" taught)
Leave-one-country-out: fit on US tokens -> predict India tokens, and the reverse (an unseen country, like France).
Then fit on both and rank France's added tokens.
  python g5_wordmodel.py   -> gen/wordmodel_{country}.parquet (test) and a LOCO report"""
import os, numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import get_context
from rapidfuzz.distance import JaroWinkler
from common import WORK, load
from match import normed

OUT = os.environ.get("GEN_DIR", f"{WORK}/gen")
MIN_A = 100
Pool = get_context("fork").Pool
_G = {}


def _shape(bounds):
    lo, hi = bounds
    rn, sn, cand = _G["rn"], _G["sn"], _G["cand"]
    ins, end, start = {}, {}, {}
    for i in range(lo, hi):
        r, s = rn[i].split(), sn[i].split()
        S = set(s); R = set(r)
        dropped = [t for t in S - R]
        for j, t in enumerate(r):
            if t in S or t not in cand:
                continue
            sub = any(JaroWinkler.similarity(t, d) >= 0.8 for d in dropped)
            ins[t] = ins.get(t, 0) + (not sub)
            end[t] = end.get(t, 0) + (j == len(r) - 1)
            start[t] = start.get(t, 0) + (j == 0)
    return ins, end, start


def shapes(split):
    n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c2 = pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values
    k = pd.read_parquet(f"{WORK}/cand_{split}.parquet", columns=["rid", "sid", "rank"])
    k = k[k["rank"] == 0]
    rid, sid = k.rid.values, k.sid.values
    out = []
    for c in sorted(set(c2[rid])):
        tok = pd.read_parquet(f"{OUT}/tokens_{split}_{c}.parquet")
        tok = tok[tok.A >= MIN_A].copy()
        m = c2[rid] == c
        _G.update(rn=n2.nn.values[rid[m]], sn=n1.nn.values[sid[m]], cand=set(tok.tok))
        n = int(m.sum()); step = n // 128 + 1
        with Pool(16) as p:
            parts = p.map(_shape, [(i, min(i + step, n)) for i in range(0, n, step)])
        agg = {key: {} for key in ("ins", "end", "start")}
        for part in parts:
            for key, dct in zip(("ins", "end", "start"), part):
                for t, v in dct.items():
                    agg[key][t] = agg[key].get(t, 0) + v
        for key in agg:
            tok[f"{key}_share"] = tok.tok.map(agg[key]).fillna(0).values / tok.A.values
        tok["country"], tok["split"] = c, split
        n_rec = int(m.sum())
        tok["logN"] = np.log1p(tok.N / n_rec * 1e6)          # per million records: comparable across countries
        tok["logA"] = np.log1p(tok.A / n_rec * 1e6)
        out.append(tok)
        print(split, c, "tokens", len(tok), flush=True)
    return pd.concat(out, ignore_index=True)


def embed(tokens):
    """fine-tuned e5 bi-encoder (mean pooling + L2, as its sentence-transformers config), plain transformers."""
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


STAT = ["logN", "logA", "add_rate", "s1_rate", "ins_share", "end_share", "start_share", "len", "digit"]


def knn_bad(E_q, E_ref, bad_ref, w_ref, k=10, self_ref=False):
    """similarity-weighted mean 'bad' of the k nearest labelled reference tokens (self excluded when E_q is E_ref)."""
    S = E_q @ E_ref.T
    if self_ref:
        np.fill_diagonal(S, -1.0)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, 1)
    wt = np.clip(s, 0, None) ** 4 * np.log1p(w_ref[idx])
    return (wt * bad_ref[idx]).sum(1) / np.maximum(wt.sum(1), 1e-9), s.max(1)


def fit_predict(tr, te):
    X = lambda d: d[STAT + ["knn_bad", "knn_sim"]]
    m = lgb.train(dict(objective="cross_entropy", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
                       verbose=-1, seed=0), lgb.Dataset(X(tr), tr.bad.values, weight=np.log1p(tr.A.values)), 300)
    return m.predict(X(te)), m


def wauc(y, p, w):
    o = np.argsort(p); y, w = y[o], w[o]
    pos = w * y; neg = w * (1 - y)
    return float((np.cumsum(neg) * pos).sum() / max(pos.sum() * neg.sum(), 1e-9))


def main():
    T = pd.concat([shapes("train"), shapes("test")], ignore_index=True)
    T["len"] = T.tok.str.len(); T["digit"] = T.tok.str.contains(r"\d").astype(float)
    E = embed(T.tok.tolist())
    tr, te = T.split.values == "train", T.split.values != "train"
    print(f"\nLeave-one-country-out (token level; a token is a distractor word if bad >= 0.9, noise if <= 0.5):", flush=True)
    for src, dst in (("US", "India"), ("India", "US")):
        # only the source country's labels are used: its own tokens get leave-self-out neighbours
        s_, d_ = tr & (T.country.values == src), tr & (T.country.values == dst)
        a, b = T[s_].copy(), T[d_].copy()
        a["knn_bad"], a["knn_sim"] = knn_bad(E[s_], E[s_], a.bad.values, a.A.values, self_ref=True)
        b["knn_bad"], b["knn_sim"] = knn_bad(E[d_], E[s_], a.bad.values, a.A.values)
        p, _ = fit_predict(a, b)
        sel = (b.bad >= 0.9) | (b.bad <= 0.5)
        y = (b.bad.values[sel] >= 0.9).astype(float); w = b.A.values[sel].astype(float)
        print(f"  {src} -> {dst}: AUC model {wauc(y, p[sel], w):.3f} | add_rate alone {wauc(y, b.add_rate.values[sel], w):.3f}"
              f" | knn alone {wauc(y, b.knn_bad.values[sel], w):.3f}  ({int(y.sum())} distractor / {int((1 - y).sum())} noise tokens)",
              flush=True)
    T.loc[tr, "knn_bad"], T.loc[tr, "knn_sim"] = knn_bad(E[tr], E[tr], T.bad.values[tr], T.A.values[tr], self_ref=True)
    T.loc[te, "knn_bad"], T.loc[te, "knn_sim"] = knn_bad(E[te], E[tr], T.bad.values[tr], T.A.values[tr])
    p, m = fit_predict(T[tr], T[te])
    T.loc[te, "p_bad"] = p
    print(pd.Series(m.feature_importance("gain"), index=STAT + ["knn_bad", "knn_sim"]).sort_values(ascending=False).round(0).to_string())
    for c in sorted(set(T.country[te])):
        d = T[te & (T.country == c)].sort_values("A", ascending=False)
        d[["tok", "N", "A", "add_rate", "s1_rate", "ins_share", "end_share", "knn_bad", "p_bad"]].to_parquet(f"{OUT}/wordmodel_{c}.parquet")
        hi = d[d.p_bad >= 0.8].head(30); lo = d[d.p_bad < 0.5].head(25)
        print(f"\n== test {c}: predicted distractor words (p_bad >= 0.8, by added count):", flush=True)
        print("  " + ", ".join(f"{t} ({a}, {q:.2f})" for t, a, q in zip(hi.tok, hi.A, hi.p_bad)), flush=True)
        print(f"   predicted noise (p_bad < 0.5): " + ", ".join(f"{t} ({q:.2f})" for t, q in zip(lo.tok, lo.p_bad)), flush=True)


if __name__ == "__main__":
    main()
