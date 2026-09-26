"""Extra pair features aimed at how distractors are made (see x_errors / x_dgroups logs), with no per-language lists:
- integer-aware address numbers: "0044" == "44" is noise, "58" vs "59" is a different business;
- legal-form edits, over the country's learned legal forms and their spelling families (lexicon.py): how many
  are added / dropped / differ;
- name edits between core names (legal forms removed), typo-tolerant: words the record adds / drops, how alike an
  added and a dropped word are (typo or variant vs a real swap), and per-country statistics of those words from the
  split's own records (no labels): how common the word is in that country's S1 names, and how much more often it is
  inserted into records than it occurs in S1 names ("suspicion": inserted distractor words score high in any
  language, e.g. holdings / group in the US, groupe / holding in France).
  python x_feats.py train|test   -> x/extra_{split}.parquet, rows in feats2 order"""
import sys, math, numpy as np, pandas as pd
from collections import Counter, defaultdict
from multiprocessing import Pool
from rapidfuzz import fuzz
from common import WORK, load, lexicon
from match import normed

COLS = ["int_first_eq", "int_a0_in_b", "int_a0_mindiff", "int_jacc", "int_a_only", "int_b_only",
        "legal_added", "legal_dropped", "legal_a", "legal_b", "legal_xor",
        "ed_add", "ed_del", "ed_swap_sim", "ed_add_susp", "ed_add_df", "ed_del_df",
        "int_first_shift", "int_near_shift"]   # signed: distractors shift the house number up (+1..13 for 92-99% of
                                                # them), true-match number noise goes either way
MIN_INS, PSEUDO_COS, NONE_DF = 20, 0.9, math.log(1e-6)
_G = {}


def ints(s):
    return tuple(int(x) for x in s.split()[:8]) if s else ()


def unmatched(X, Y):
    return [t for t in X if not any(fuzz.ratio(t, u) >= 80 for u in Y)]


def _chunk(i):
    lo, hi = _G["job"][i]
    out = np.full((hi - lo, len(COLS)), np.nan, np.float32)
    IA, IB, LA, LB, CA, CB, CT, ST = (_G[k] for k in ("ia", "ib", "la", "lb", "ca", "cb", "ct", "st"))
    for j, (x, y) in enumerate(zip(_G["sa"][lo:hi], _G["rb"][lo:hi])):
        A, B = IA[x], IB[y]
        if A and B:
            a0, SA, SB = A[0], set(A), set(B)
            out[j, :6] = (a0 == B[0], a0 in SB, math.log1p(min(abs(a0 - b) for b in B)),
                          len(SA & SB) / len(SA | SB), len(SA - SB), len(SB - SA))
            near = min(B, key=lambda b: abs(b - a0))
            out[j, 17:19] = (max(-100, min(100, B[0] - a0)), max(-100, min(100, near - a0)))
        else:
            out[j, :6] = (-1, -1, -1, -1, len(A), len(B))
        la, lb = LA[x], LB[y]
        out[j, 6:11] = (len(lb - la), len(la - lb), len(la), len(lb), len(la ^ lb))
        ua, ub = unmatched(CA[x], CB[y]), unmatched(CB[y], CA[x])
        st = ST[CT[x]]
        out[j, 11:13] = (len(ub), len(ua))
        if ua and ub:
            out[j, 13] = max(fuzz.ratio(t, u) for t in ub for u in ua)
        if ub:
            out[j, 14] = max(st.get(t, (0.0, NONE_DF))[0] for t in ub)
            out[j, 15] = max(st.get(t, (0.0, NONE_DF))[1] for t in ub)
        if ua:
            out[j, 16] = max(st.get(t, (0.0, NONE_DF))[1] for t in ua)
    return out


def word_stats(keys, s1c, cn1, cn2):
    """per country: word -> (suspicion, log share of S1 core names containing it)."""
    df, n_s1 = defaultdict(Counter), Counter(s1c)
    for c, x in zip(s1c, cn1):
        df[c].update(set(x.split()))
    pm = keys[(keys["rank"].values == 0) & (keys.score.values >= PSEUDO_COS)]
    ins, n_pm = defaultdict(Counter), Counter()
    for s, r in zip(pm.sid.values, pm.rid.values):
        c = s1c[s]
        n_pm[c] += 1
        ins[c].update(set(cn2[r].split()) - set(cn1[s].split()))
    stats = {}
    for c in n_s1:
        susp = {w: math.log((v / max(n_pm[c], 1) + 1e-5) / (df[c][w] / n_s1[c] + 1e-5))
                for w, v in ins[c].items() if v >= MIN_INS}
        med = float(np.median(list(susp.values()))) if susp else 0.0
        st = {w: (0.0, math.log(v / n_s1[c] + 1e-6)) for w, v in df[c].items()}
        for w, v in susp.items():
            st[w] = (v - med, st.get(w, (0.0, NONE_DF))[1])
        stats[c] = st
        top = sorted(susp, key=susp.get, reverse=True)[:15]
        print(f"{c}: most inserted-vs-S1 words " + ", ".join(f"{w} {susp[w] - med:+.1f}" for w in top), flush=True)
    return stats


def main(split):
    keys = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid", "score", "rank"])
    n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    s1c = load(split, 1).country.values
    oc = pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values
    legal = {c: set(v) for c, v in lexicon()["legal"].items()}
    canon = lexicon().get("legal_canon", {})     # spelling variants of one legal form count as that form
    lset = lambda nn, cs: [frozenset(canon.get(c, {}).get(t, t) for t in x.split() if t in legal.get(c, ()))
                           for x, c in zip(nn, cs)]
    cn1, cn2 = n1.cn.values, n2.cn.values
    stats = word_stats(keys, s1c, cn1, cn2)
    codes = {c: i for i, c in enumerate(stats)}
    _G.update(ia=n1.num.map(ints).tolist(), ib=n2.num.map(ints).tolist(), la=lset(n1.nn.values, s1c),
              lb=lset(n2.nn.values, oc), ca=[x.split() for x in cn1], cb=[x.split() for x in cn2],
              ct=[codes[c] for c in s1c], st=[stats[c] for c in codes], sa=keys.sid.values, rb=keys.rid.values)
    step = len(keys) // 2048 + 1
    _G["job"] = [(i, min(i + step, len(keys))) for i in range(0, len(keys), step)]
    with Pool(32) as p:
        out = pd.DataFrame(np.concatenate(p.map(_chunk, range(len(_G["job"])))), columns=COLS)
    out.to_parquet(f"{WORK}/x/extra_{split}.parquet")
    print(split, out.shape, out.describe().T[["mean", "min", "max"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
