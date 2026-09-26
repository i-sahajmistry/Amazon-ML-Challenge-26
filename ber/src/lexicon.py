"""Per-country token lexicon learned from the records instead of hard-coded tables. Abbreviations mean different
things in different countries ("tn" is Tennessee in the US and Tamil Nadu in India, "ste" is suite in the US and
sainte in France, "de" / "la" are French words), and a new country needs its own. Learned without labels from
pseudo-matches: each S2/S3 record paired with its top retrieved S1 when the embedding cosine is high.
  abbr[country][field]: short token -> expansion. The token is left unmatched in a pseudo-match while the other side
      has the expansion: one word it is a letter subsequence of (same first letter; "rd" -> "road"), or a 2-3 word
      span it is the initials of ("tn" -> "tamil nadu"). Kept when frequent and unambiguous in that country.
  legal[country]: legal-form tokens: frequent in the last two words of that country's S1 names and dropped between
      pseudo-matches far more often than that country's other such words (median drop rate + 0.1), plus words that
      almost always come right before one particular legal word ("private" limited) and record-side spellings of them (pvt, praivet).
  legal_canon[country]: legal word -> its family's most common S1 spelling. Two legal words are one family when one is
      a spelling / abbreviation of the other AND pseudo-matches swap them mostly with each other ("pvt", "praivet" ->
      "private"); different forms that happen to share letters stay apart ("sa" vs "sarl").
  python lexicon.py        # learns from work/cand_{train,test}.parquet (retrieve.py), writes work/lexicon.json"""
import json, os, re
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np, pandas as pd

MIN_COS, MIN_N, MIN_CONF, MIN_RATE, LEGAL_SHARE, LEGAL_DROP = 0.9, 30, 0.8, 0.2, 0.005, 0.1


def toks(x):
    from common import tokens   # the tokeniser normalisation uses (late import: common reads this module's output)
    return tokens(x)


def subseq(t, w):
    it = iter(w)
    return all(ch in it for ch in t)


def _mine(pairs):
    """pairs: [(S1 text, record text)]. Counts (short token, expansion) explanations and unmatched short tokens."""
    cnt, unm = Counter(), Counter()
    for a, b in pairs:
        A, B = toks(a), toks(b)
        for X, Y in ((A, B), (B, A)):
            sx, sy = set(X), set(Y)
            uy = [w for w in Y if w not in sx]
            su = set(uy)
            for t in sx - sy:
                if not t.isalpha() or len(t) > 4:
                    continue
                unm[t] += 1
                exp = {w for w in su if len(w) > len(t) and w[0] == t[0] and subseq(t, w)}
                if 2 <= len(t) <= 3:
                    n = len(t)
                    exp |= {" ".join(Y[i:i + n]) for i in range(len(Y) - n + 1)
                            if all(w in su for w in Y[i:i + n]) and "".join(w[0] for w in Y[i:i + n]) == t}
                for e in exp:
                    cnt[t, e] += 1
    return cnt, unm


def _drops(pairs):
    """name pseudo-matches: how often a word of the S1 name is missing from the record (and vice versa)."""
    has, drop, hasb, add = Counter(), Counter(), Counter(), Counter()
    for a, b in pairs:
        A, B = set(toks(a)), set(toks(b))
        has.update(A); drop.update(A - B); hasb.update(B); add.update(B - A)
    return has, drop, hasb, add


def _swaps(args):
    """name pseudo-matches: how often a legal word of the record replaces a legal word of the S1 name."""
    pairs, legal = args
    cnt = Counter()
    for a, b in pairs:
        A, B = set(toks(a)) & legal, set(toks(b)) & legal
        for u in B - A:
            gone = A - B
            for t in [t for t in gone if related(u, t)] or gone:     # a spelling of a dropped word, else a change
                cnt[u, t] += 1; cnt[t, u] += 1
    return cnt


def related(u, t):
    return (skel(u) == skel(t) and len(skel(u)) >= 2) or any(
        x[0] == y[0] and subseq(x, y) and len(y) - len(x) >= 2 for x, y in ((u, t), (t, u)))


def skel(w):
    return w[0] + re.sub("[aeiouhy]", "", w[1:])


def learn(work):
    from common import load   # late import: common imports this module
    rows = defaultdict(lambda: {"name": [], "addr": []})
    last, n_s1, pair, words = defaultdict(Counter), Counter(), defaultdict(Counter), defaultdict(Counter)
    for split in ("train", "test"):
        s1 = load(split, 1)
        o = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
        c = pd.read_parquet(f"{work}/cand_{split}.parquet", columns=["rid", "sid", "score", "rank"])
        c = c[(c["rank"] == 0) & (c.score >= MIN_COS)]
        for f, col in (("name", "business_name"), ("addr", "business_address")):
            a, b = s1[col].values[c.sid.values], o[col].values[c.rid.values]
            for k, x, y in zip(s1.country.values[c.sid.values], a, b):
                rows[k][f].append((x, y))
        n_s1.update(s1.country.values)
        for k, n in zip(s1.country.values, s1.business_name.values):
            t = toks(n)
            last[k].update(set(t[-2:]))
            words[k].update(set(t))
            if len(t) > 1:
                pair[k][t[-2], t[-1]] += 1
    lex = {"abbr": {}, "legal": {}, "legal_canon": {}}
    with Pool(32) as p:
        for k in rows:
            lex["abbr"][k] = {}
            for f in ("name", "addr"):
                pr = rows[k][f]
                pr = [pr[i] for i in np.random.default_rng(0).permutation(len(pr))[:2_000_000]]
                ch = [pr[i:i + 20000] for i in range(0, len(pr), 20000)]
                cnt, unm = Counter(), Counter()
                for c1, u1 in p.map(_mine, ch):
                    cnt.update(c1); unm.update(u1)
                by = defaultdict(Counter)
                for (t, e), n in cnt.items():
                    by[t][e] = n
                m = {}
                for t, es in by.items():
                    e, n = es.most_common(1)[0]
                    short = len(t) <= 2 or len(e.replace(" ", "")) - len(t) >= 2      # a dropped letter is a typo
                    rate = MIN_RATE if len(t) > 1 else 0.5                           # single letters: stronger evidence
                    if short and n >= MIN_N and n / sum(es.values()) >= MIN_CONF and n / unm[t] >= rate:
                        m[t] = e
                lex["abbr"][k][f] = m
            pr = rows[k]["name"]
            has, drop, hasb, add = Counter(), Counter(), Counter(), Counter()
            for h, d, hb, a in p.map(_drops, [pr[i:i + 20000] for i in range(0, len(pr), 20000)]):
                has.update(h); drop.update(d); hasb.update(hb); add.update(a)
            n1 = n_s1[k]
            cand = [w for w, v in last[k].items() if v / n1 >= LEGAL_SHARE and w.isalpha() and has[w] >= MIN_N]
            print(k, "suffix words (share of S1 names / drop rate / add rate in pseudo-matches):", " ".join(
                f"{w}={last[k][w] / n1:.3f}/{drop[w] / has[w]:.2f}/{add[w] / max(hasb[w], 1):.2f}"
                for w in sorted(cand, key=lambda w: -last[k][w])[:40]), flush=True)
            cut = float(np.median([drop[w] / has[w] for w in cand])) + LEGAL_DROP
            legal = {w for w in cand if drop[w] / has[w] >= cut}
            # the first word of a two-word form: nearly always followed by one particular legal word
            legal |= {a for (a, b), v in pair[k].items()
                      if b in legal and a.isalpha() and v / n1 >= LEGAL_SHARE and v / words[k][a] >= 0.8}
            # record-side spellings of those forms ("pvt", "ltd", transliterated "praivet"): mostly added, same skeleton
            # or an abbreviation of a legal word
            legal |= {u for u, v in hasb.items() if v >= MIN_N and u.isalpha() and add[u] / v >= 0.5 and any(
                (skel(u) == skel(t) and len(skel(u)) >= 2) or (u[0] == t[0] and subseq(u, t) and len(t) - len(u) >= 2) for t in legal)}
            lex["legal"][k] = sorted(legal)
            sw = Counter()
            for c1 in p.map(_swaps, [(pr[i:i + 20000], legal) for i in range(0, len(pr), 20000)]):
                sw.update(c1)
            tot = Counter()
            for (u, t), v in sw.items():
                tot[u] += v
            parent = {w: w for w in legal}
            find = lambda w: w if parent[w] == w else find(parent[w])
            for (u, t), v in sw.items():
                # u is the rarer spelling in S1 names, and mostly swaps with t
                if v >= MIN_N and v / tot[u] >= 0.6 and words[k][u] <= words[k][t] and related(u, t):
                    parent[find(u)] = find(t)
            fam = defaultdict(list)
            for w in legal:
                fam[find(w)].append(w)
            lex["legal_canon"][k] = {w: max(ws, key=lambda x: (words[k][x], x)) for ws in fam.values() for w in ws
                                     if len(ws) > 1}
            print(k, "legal families:", "; ".join(" ".join(sorted(ws)) for ws in fam.values() if len(ws) > 1)[:600],
                  flush=True)
    json.dump(lex, open(f"{work}/lexicon.json", "w"), indent=1, sort_keys=True)
    return lex


if __name__ == "__main__":
    from common import WORK
    lex = learn(WORK)
    for k in lex["abbr"]:
        for f in ("name", "addr"):
            m = lex["abbr"][k][f]
            print(f"{k} {f}: {len(m)} -> " + ", ".join(f"{t}={e}" for t, e in sorted(m.items())[:80]), flush=True)
        print(f"{k} legal: {' '.join(lex['legal'][k])}", flush=True)
