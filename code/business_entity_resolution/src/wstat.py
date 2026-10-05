"""House-number word statistics (no labels, per country, from each split's own records).
A decoy copies its S1's name with one edit and moves to a shifted house number; a true record's name edits are noise
and it keeps the S1's number. So for a word w that records add to (or drop from) the name of the S1 they claim, the
share of those records still at the S1's first house number tells filler words (services, cie, fils: ~0.7) from decoy
words (holdings, participations: ~0.02) in any language. On US / India train this share correlates 0.99 / 0.97 with
the word's true-match rate. Rates are normalised per country: 0 = the country's decoy-word level (5th percentile over
frequent words), 1 = records whose name is unchanged.
  python wstat.py train|test   -> x/wstat_{split}.parquet (rows in feats2 order), x/wstat_words_{split}.parquet"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from rapidfuzz import fuzz
from common import WORK, load
from match import normed

XD = f"{WORK}/x"
COLS = ["wa_n", "wa_k", "wa_min", "wa_max", "wa_mean", "wd_n", "wd_k", "wd_min", "wd_max", "wd_mean"]
MIN_N, M = 30, 20          # words seen fewer times get no statistic; M = pseudo-count toward the country's rate
_G = {}


def edits(a, b):
    """words of a missing from b and of b missing from a, typo-tolerant (ratio >= 80 counts as present)."""
    A, B = a.split(), b.split()
    add = [t for t in dict.fromkeys(A) if t not in B and not any(fuzz.ratio(t, u) >= 80 for u in B)]
    drop = [t for t in dict.fromkeys(B) if t not in A and not any(fuzz.ratio(t, u) >= 80 for u in A)]
    return add, drop


def _claims(i):
    lo, hi = _G["job"][i]
    out = []
    for r, s in zip(_G["rid"][lo:hi], _G["sid"][lo:hi]):
        out.append(edits(_G["nr"][r], _G["ns"][s]))
    return out


def _feats(i):
    lo, hi = _G["job"][i]
    out = np.full((hi - lo, len(COLS)), np.nan, np.float32)
    for j, (r, s) in enumerate(zip(_G["rid"][lo:hi], _G["sid"][lo:hi])):
        c = _G["ctry"][s]
        add, drop = edits(_G["nr"][r], _G["ns"][s])
        for k, (ws, tab) in enumerate(((add, _G["za"]), (drop, _G["zd"]))):
            z = [tab[(c, w)] for w in ws if (c, w) in tab]
            out[j, 5 * k] = len(ws)
            out[j, 5 * k + 1] = len(z)
            if z:
                out[j, 5 * k + 2:5 * k + 5] = min(z), max(z), sum(z) / len(z)
    return out


def jobs(n, parts=1024):
    step = n // parts + 1
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def main(split):
    s1 = load(split, 1)
    R = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    S = normed(split, 1)
    k = pd.read_parquet(f"{XD}/{'oof5_train' if split == 'train' else 'p5_test'}_v10.parquet")
    best = k.sort_values("p", ascending=False).drop_duplicates("rid")          # each record's claimed S1 (stage 1)
    first = lambda s: np.array([x.split()[0] if x else "" for x in s])
    rn, sn = first(R.num.values)[best.rid.values], first(S.num.values)[best.sid.values]
    ok = (rn != "") & (sn != "")
    best = best[ok]
    same = (rn == sn)[ok]
    ctry = s1.country.values
    _G.update(nr=R.nn.values, ns=S.nn.values, rid=best.rid.values, sid=best.sid.values, ctry=ctry, job=jobs(len(best)))
    with Pool(32) as p:
        ed = [e for part in p.map(_claims, range(len(_G["job"]))) for e in part]
    c = ctry[best.sid.values]
    unch_all = np.array([not e[0] and not e[1] for e in ed])
    za, zd = {}, {}
    for side, tab in ((0, za), (1, zd)):
        rows = [(c[i], w, same[i]) for i, e in enumerate(ed) for w in e[side]]
        W = pd.DataFrame(rows, columns=["c", "w", "same"]).groupby(["c", "w"]).same.agg(["size", "sum"])
        W = W[W["size"] >= MIN_N]
        for cc in W.index.get_level_values(0).unique():
            unch = unch_all & (c == cc)
            hi = same[unch].mean()                                           # name unchanged: true records
            base = same[c == cc].mean()
            x = W.loc[cc]
            rate = (x["sum"] + M * base) / (x["size"] + M)
            big = x["size"] >= 300
            lo = np.percentile(np.repeat(rate[big].values, x["size"][big].values // 100 + 1), 5)
            z = ((rate - lo) / (hi - lo)).clip(-0.5, 1.5)
            print(f"{split} {cc} {'add' if side == 0 else 'drop'}: words {len(x)}, lo {lo:.3f}, hi {hi:.3f}", flush=True)
            for w, v in z.items():
                tab[(cc, w)] = float(v)
            pd.DataFrame({"c": cc, "w": x.index, "n": x["size"].values, "rate": rate.values, "z": z.values,
                          "side": "add" if side == 0 else "drop"}).to_parquet(
                f"{XD}/wstat_words_{split}_{cc}_{side}.parquet")
    f = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"])
    _G.update(za=za, zd=zd, rid=f.rid.values, sid=f.sid.values, job=jobs(len(f)))
    with Pool(32) as p:
        F = np.concatenate(p.map(_feats, range(len(_G["job"]))))
    pd.DataFrame(F, columns=COLS).to_parquet(f"{XD}/wstat_{split}.parquet")
    print(split, "pairs", len(F), "with an added word", np.mean(F[:, 0] > 0).round(3), flush=True)


if __name__ == "__main__":
    assert edits("mc groupement sa", "mc soins") == (["groupement", "sa"], ["soins"])
    assert edits("lille comite sas", "lille comte sas") == ([], [])      # typo, not an edit
    main(sys.argv[1])
