"""Learn each country's decoy words without labels. True matches never add the words the distractor generator adds
(train: holdings / group / partners / place words in the US, enterprises / industries / ventures in India, share of
true matches adding them 0.0000), and distractors often sit at a shifted house number while true matches keep it. So
a word added (core-name token not in the claimed S1's core name) mostly by records whose first number differs from
the S1's and shares nothing with it is a decoy word. Per country: shifted share among the records adding each word.
Train (claimed S1 = top-1 retrieved) checks it against labels; test (claimed S1 = v9p stage-2 best) writes
work/decoy_words.json for x_final.py WORDVETO=.
  python x_decoyveto.py [MIN_SHIFT] [MIN_N]"""
import sys, json, numpy as np, pandas as pd
from collections import defaultdict
from common import WORK, load
from harness import truth_arrays
from match import normed

MIN_SHIFT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
MIN_N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ints = lambda x: [int(t[:15]) for t in x.split()]


def stats(split, rid, sid, ctry):
    """per country: word -> [records adding it with both numbers present, of which shifted] + baseline shift share."""
    n1, n2 = normed(split, 1), pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c1, c2, u1, u2 = n1.cn.values, n2.cn.values, n1.num.values, n2.num.values
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    base = defaultdict(lambda: [0, 0])
    for r, s, c in zip(rid, sid, ctry):
        a, b = ints(u2[r]), ints(u1[s])
        if not a or not b:
            continue
        sh = a[0] not in b and b[0] not in a
        add = set(c2[r].split()) - set(c1[s].split())
        base[c][0] += 1; base[c][1] += sh
        for w in add:
            out[c][w][0] += 1; out[c][w][1] += sh
    return out, base


def learned(out, base):
    words = {}
    for c, d in out.items():
        b = base[c][1] / max(base[c][0], 1)
        ws = {w: v[1] / v[0] for w, v in d.items() if v[0] >= MIN_N and v[1] / v[0] >= MIN_SHIFT}
        words[c] = sorted(ws, key=ws.get, reverse=True)
        top = sorted(((v[1] / v[0], w, v[0]) for w, v in d.items() if v[0] >= MIN_N), reverse=True)[:30]
        print(f"{c}: baseline shifted share {b:.3f}; {len(ws)} words >= {MIN_SHIFT}; top: " +
              ", ".join(f"{w} {s:.2f}/{n}" for s, w, n in top), flush=True)
    return words


s1, other, ts, s1f, rf = truth_arrays()
c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
c = c[c["rank"].values == 0]
out, base = stats("train", c.rid.values, c.sid.values, s1.country.values[c.sid.values])
tw = learned(out, base)
# check on train labels: share of records adding a learned word that are distractors, and of true matches adding one
cn1 = normed("train", 1).cn.values
cn2 = pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True).cn.values
for ctry in ("US", "India"):
    m = s1.country.values[c.sid.values] == ctry
    W = set(tw.get(ctry, []))
    hit = np.array([bool(W & (set(cn2[r].split()) - set(cn1[s].split()))) for r, s in zip(c.rid.values[m], c.sid.values[m])])
    true = ts[c.rid.values[m]] == c.sid.values[m]
    print(f"train {ctry}: records adding a learned word {hit.mean():.3%}, of which distractors {(~true)[hit].mean():.4f}; "
          f"true matches adding one {hit[true].mean():.4%}", flush=True)

q = pd.read_parquet(f"{WORK}/x/test_q_v9pw.parquet")
out, base = stats("test", q.rid.values, q.sid.values, q.c.values)
w = learned(out, base)
json.dump(w, open(f"{WORK}/decoy_words.json", "w"), indent=1)
print("wrote", f"{WORK}/decoy_words.json", {k: len(v) for k, v in w.items()}, flush=True)
# accepted test records (v9p, q >= 0.7) that add a learned word of their country, with samples
t1 = load("test", 1); t2 = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
cn1, cn2 = normed("test", 1).cn.values, pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True).cn.values
rng = np.random.default_rng(0)
for ctry in ("US", "India", "France"):
    g = q[(q.c == ctry) & (q.q >= 0.7)]
    W = set(w.get(ctry, []))
    hit = np.array([bool(W & (set(cn2[r].split()) - set(cn1[s].split()))) for r, s in zip(g.rid.values, g.sid.values)])
    print(f"test {ctry}: accepted records adding a learned word {hit.sum()} ({hit.mean():.3%})", flush=True)
    for i in rng.choice(np.flatnonzero(hit), min(12, hit.sum()), replace=False):
        r, s = g.rid.values[i], g.sid.values[i]
        print(f"   {g.q.values[i]:.2f} +{sorted(W & (set(cn2[r].split()) - set(cn1[s].split())))}  S1 {t1.business_name.values[s]} | "
              f"{t1.business_address.values[s]}\n         REC {t2.business_name.values[r]} | {t2.business_address.values[r]}", flush=True)
