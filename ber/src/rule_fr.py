"""Structural acceptance for a country without training labels (France here). A record is taken as true, whatever else
its name changes, when it keeps all of these from its S1:
- the house number, suffix included (20 and 20 bis differ);
- the second number of a compound (102-21 vs 102-42; India plot numbers);
- the S1's distinctive street words (street segment, minus the country's most common address words);
- the first and the rarest core-name word;
and it adds no decoy word (house-number word statistic z < ZMAX, wstat.py).
On US / India validation such records are > 99% true matches (decoys move to a shifted number). Models trained on
US / India text reject French type-word swaps and suffixes at the same address ("Mc Soins" -> "Mc Groupement [SA]").
  python rule_fr.py BASE_TAG [COUNTRIES]  -> x/test_q_rule{BASE_TAG}.parquet: q = 1 for those records (S1 from
                                             BASE_TAG), else 0. COUNTRIES: comma-separated, default 'unlabelled' (the
                                             countries without training labels, common.unlabelled)
  python rule_fr.py val                   -> the same rule on US / India validation folds 8-9, with labels"""
import os, re, sys, numpy as np, pandas as pd
from collections import Counter
from rapidfuzz import fuzz
from common import WORK, load, norm_addr, countries
from match import normed

XD = f"{WORK}/x"
ZMAX = float(os.environ.get("ZMAX", 0.1))
_HN = re.compile(r"(\d+)(?:\s*(bis|ter|quater)\b|([a-z])\b)?")   # a single letter counts only when glued: 18b
_D = re.compile(r"\d")


def house(na):
    """first house number with its suffix: '20 bis rue x' -> '20bis', '0165 rue' -> '165'"""
    m = _HN.search(na)
    return f"{int(m.group(1)[:12])}{m.group(2) or m.group(3) or ''}" if m else ""


def second(num):
    t = num.split()
    return int(t[1][:12]) if len(t) > 1 else -1


def street_seg(addr):
    """the comma-separated part holding the first number (the first part if none)"""
    parts = addr.split(",")
    return next((p for p in parts if _D.search(p)), parts[0])


def has(tok, toks):
    return any(t == tok or fuzz.ratio(t, tok) >= 80 for t in toks)


def rule(split, d):
    """d: rid, sid, c -> boolean mask"""
    s1 = load(split, 1)
    R = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True).iloc[d.rid.values].reset_index(drop=True)
    S = normed(split, 1).iloc[d.sid.values].reset_index(drop=True)
    ok = np.zeros(len(d), bool)
    for c in d.c.unique():
        m = np.where(d.c.values == c)[0]
        sc = s1[s1.country == c]
        seg = norm_addr(sc.business_address.map(street_seg), sc.country)
        common = {w for w, _ in Counter(t for x in seg.values for t in set(x.split())).most_common(200)}
        dist = pd.Series([[t for t in x.split() if len(t) >= 3 and not _D.search(t) and t not in common] for x in seg.values],
                         index=sc.index)
        df = Counter(t for x in normed(split, 1).cn.values[sc.index.values] for t in set(x.split()))
        z = pd.read_parquet(f"{XD}/wstat_words_{split}_{c}_0.parquet")
        zd = dict(zip(z.w, z.z))
        for i in m:
            s = d.sid.values[i]
            ra, sa = R.na.values[i], S.na.values[i]
            hr, hs = house(ra), house(sa)
            if not hr or hr != hs:
                continue
            r2, s2 = second(R.num.values[i]), second(S.num.values[i])
            if r2 >= 0 and s2 >= 0 and r2 != s2:
                continue
            rt = ra.split()
            ds = dist[s]
            if ds and sum(has(t, rt) for t in ds) * 2 < len(ds):
                continue
            rc, scn = R.cn.values[i].split(), S.cn.values[i].split()
            if not rc or not scn or rc[0] != scn[0]:
                continue
            rare = min(scn, key=lambda t: df.get(t, 0))
            if not has(rare, R.nn.values[i].split()):
                continue
            if any(zd.get(t, 1.0) < ZMAX for t in set(R.nn.values[i].split()) - set(S.nn.values[i].split())):
                continue
            ok[i] = True
    return ok


if __name__ == "__main__":
    assert house("20 bis rue x") == "20bis" and house("25 r du moulin") == "25" and house("0165 rue a") == "165"
    assert street_seg("Nantes, 18 Rue Capitaine Corhumel, Pays de la Loire") == " 18 Rue Capitaine Corhumel"
    if sys.argv[1] == "val":
        os.environ["DFOLD"] = "1"
        from harness import truth_arrays
        from match import assign
        s1_, _, ts, s1f, _ = truth_arrays()
        k = pd.read_parquet(f"{XD}/oof5_train_v10.parquet")
        b = assign(k[["rid", "sid"]], k.p.values, 0.0)
        b = b[s1f[b.sid.values] >= 8].reset_index(drop=True)
        b["c"] = s1_.country.values[b.sid.values]
        ok = rule("train", b)
        y, rej = ts[b.rid.values] == b.sid.values, b.p.values < 0.5
        for c in b.c.unique():
            m = (b.c.values == c) & ok
            print(f"{c}: rule records {m.sum()}, true {y[m].mean():.4f}; model-rejected {(m & rej).sum()}, true there "
                  f"{y[m & rej].mean():.4f}; decoys in rule {(m & (ts[b.rid.values] < 0)).sum()}", flush=True)
        sys.exit()
    TAG, C = sys.argv[1], countries(sys.argv[2] if len(sys.argv) > 2 else "unlabelled")
    d = pd.read_parquet(f"{XD}/test_q{TAG}.parquet")
    d = d[d.c.isin(C)].reset_index(drop=True)
    ok = rule("test", d)
    print(f"{C}: records {len(d)}, rule accepts {ok.sum()}, of which below 0.70 now {(ok & (d.q.values < 0.7)).sum()}", flush=True)
    d.assign(q=ok.astype(np.float32))[["rid", "sid", "q", "c"]].to_parquet(f"{XD}/test_q_rule{TAG}.parquet")
