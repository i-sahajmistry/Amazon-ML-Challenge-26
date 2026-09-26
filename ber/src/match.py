"""Pairwise features + LightGBM matcher.
  python match.py train  -> features on train candidates, fit on S1 folds 4-8, tune threshold on fold 9
  python match.py test   -> score test candidates, write output/matching_results.tsv + candidate_pairs.tsv
Decision rule: every S2/S3 record belongs to at most one S1 entity (true in train GT), so each record is
assigned to its highest-probability S1 candidate if that probability clears the threshold."""
import sys, os, json, re, numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import Pool
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from collections import Counter
from common import load, s1_fold, norm_name, core_name, norm_addr, skeleton, f05, WORK, ROOT

KF = int(os.environ.get("KF", 5))  # candidates per S2/S3 record that go to the matcher
# Adaptive shortlist of the retrieved top-20 (empty = fixed top-KF); what it keeps is candidate_pairs.tsv:
#   SHORTLIST="softmax:T:c"  per record, the fewest top candidates whose softmax(score / T) sums to >= c  (Sarvesh)
#   SHORTLIST="gap:d"        per record, every candidate scoring within d of the record's best             (Sarvesh)
#   SHORTLIST="model:tau"    candidates whose calibrated probability (shortlist.py, retrieval features only) >= tau
#   SHORTLIST="text:tau"     the same with text similarities of each pair too (shortlist.py, TEXT_F)
SHORTLIST = os.environ.get("SHORTLIST", "")
SHORT_F = ["score", "rank", "gap_top1", "gap_next", "top1", "n02", "n05", "n10", "soft"]
# SHORTLIST="text:tau": the shortlist model also sees cheap text similarities of each retrieved pair and their gap to
# the record's best (v9): at tau 0.005 it keeps 99.36% of true S1s with 1.17 candidates per record vs 99.28% / 1.34
# for SHORT_F alone (train records of S1 folds 7-9)
TEXT_F = SHORT_F + ["t_nn", "t_cn", "t_na", "t_nn_gap", "t_cn_gap", "t_na_gap"]
OUT = os.environ.get("OUT", f"{ROOT}/output")
_NUM = re.compile(r"\d+")
_ZIP = re.compile(r"\b\d{5,6}\b")


def _norm_chunk(df):
    nn = norm_name(df.business_name, df.country)
    na = norm_addr(df.business_address, df.country)
    cn = core_name(nn, df.country)
    return pd.DataFrame({
        "nn": nn, "cn": cn, "sk": skeleton(cn), "na": na,
        "num": na.map(lambda x: " ".join(_NUM.findall(x))),
        "zip": na.map(lambda x: (_ZIP.findall(x) or [""])[-1]),
        "nonascii": df.business_name.map(lambda x: not x.isascii()),
    }, index=df.index)


def normed(split, src):
    f = f"{WORK}/pq/norm2_{split}_{src}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    df = load(split, src)
    with Pool(32) as p:
        step = len(df) // 256 + 1
        out = pd.concat(p.map(_norm_chunk, [df.iloc[i:i + step] for i in range(0, len(df), step)]))
    out.to_parquet(f)
    return out


_G = {}  # read-only globals shared with forked workers


def _df_chunk(i):
    col, lo, hi = _G["job"][i]
    c = Counter()
    for x in _G[col][lo:hi]:
        c.update(set(x.split()))
    return c


def _idf(col, texts):
    """log(N/df) over all texts of a split (S1 + S2 + S3)."""
    _G[col] = texts
    step = len(texts) // 256 + 1
    _G["job"] = [(col, i, i + step) for i in range(0, len(texts), step)]
    with Pool(32) as p:
        df = Counter()
        for c in p.imap_unordered(_df_chunk, range(len(_G["job"]))):
            df.update(c)
    n = len(texts)
    return {t: float(np.log(n / v)) for t, v in df.items()}, float(np.log(n))


def _wj_chunk(i):
    """idf-weighted token overlap for pairs lo..hi: weighted jaccard, shared idf, max idf of unshared tokens per side."""
    lo, hi = _G["job2"][i]
    out = np.zeros((hi - lo, 8), np.float32)
    for j, (ia, ib) in enumerate(zip(_G["sa"][lo:hi], _G["rb"][lo:hi])):
        (icn, dcn), (ina, dna) = _G["idf"][_G["pk"][ia]]            # idf of the pair's country (or the whole split)
        for f, (A, B, idf, dflt) in enumerate([(_G["a_cn"][ia], _G["b_cn"][ib], icn, dcn),
                                               (_G["a_na"][ia], _G["b_na"][ib], ina, dna)]):
            A, B = set(A.split()), set(B.split())
            w = lambda S: sum(idf.get(t, dflt) for t in S)
            inter, un = w(A & B), w(A | B)
            out[j, 4 * f:4 * f + 4] = (inter / un if un else 0.0, inter,
                                       max((idf.get(t, dflt) for t in A - B), default=0.0),
                                       max((idf.get(t, dflt) for t in B - A), default=0.0))
    return out


def features(split):
    f = f"{WORK}/feats2_{split}.parquet"
    if os.environ.get("SMOKE"):
        return _features(split)
    if os.path.exists(f):
        d = pd.read_parquet(f)
        return d[["rid", "sid"]], d.drop(columns=["rid", "sid"]), load(split, 1),             pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    keys, F, s1, other = _features(split)
    pd.concat([keys, F], axis=1).to_parquet(f)
    return keys, F, s1, other


def retrieval_feats(c):
    """per retrieved candidate (rows of a record contiguous, in rank order): gaps to the best / next score, how many
    candidates are nearly tied with the best, and the candidate's softmax share of the record's scores."""
    r, sc = c.rid.to_numpy(), c.score.to_numpy(np.float64)
    g = c.groupby("rid").score
    c["top1"] = g.transform("max")
    c["gap_top1"] = c.top1 - c.score
    c["next"] = g.shift(-1).fillna(-1)
    c["gap_next"] = c.score - c["next"]
    top1 = c.top1.to_numpy(np.float64)
    for d in (0.02, 0.05, 0.1):
        c[f"n{int(d * 100):02d}"] = pd.Series(sc >= top1 - d).groupby(r).transform("sum").to_numpy(np.int16)
    e = np.exp((sc - top1) / 0.05)
    c["soft"] = e / pd.Series(e).groupby(r).transform("sum").to_numpy()
    return c


def text_feats(c, n1, n2):
    """cheap text similarities of every retrieved pair (normalised name, core name, address) and each one's gap to
    the record's best pair: one rapidfuzz comparison per retrieved pair, linear in records x K like retrieval."""
    P = dict(workers=-1, dtype=np.float32)
    for name, col, scorer in (("t_nn", "nn", fuzz.token_set_ratio), ("t_cn", "cn", fuzz.ratio),
                              ("t_na", "na", fuzz.token_set_ratio)):
        c[name] = process.cpdist(n1[col].values[c.sid.values].tolist(), n2[col].values[c.rid.values].tolist(),
                                 scorer=scorer, **P)
        c[name + "_gap"] = c.groupby("rid")[name].transform("max").values - c[name].values
    return c


def keep_mask(c):
    """which retrieved candidates go to the matcher (see SHORTLIST)."""
    if not SHORTLIST:
        return (c["rank"] < KF).to_numpy()
    kind, *args = SHORTLIST.split(":")
    if kind == "gap":
        return c.score.to_numpy() >= c.top1.to_numpy() - float(args[0])
    if kind == "softmax":
        T, cut = args
        e = np.exp((c.score.to_numpy(np.float64) - c.top1.to_numpy(np.float64)) / float(T))
        g = pd.Series(e).groupby(c.rid.to_numpy())
        before = (g.cumsum() - e) / g.transform("sum")          # probability mass of the better-ranked candidates
        return (before < float(cut) - 1e-9).to_numpy()          # keep until the running total reaches c
    assert kind in ("model", "text"), SHORTLIST
    F, f = (SHORT_F, "shortlist.txt") if kind == "model" else (TEXT_F, "shortlist_text.txt")   # shortlist.py
    return lgb.Booster(model_file=f"{WORK}/{f}").predict(c[F], num_threads=32) >= float(args[0])


def _features(split):
    s1, n1 = load(split, 1), normed(split, 1)
    other = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c = retrieval_feats(pd.read_parquet(f"{WORK}/cand_{split}.parquet"))   # context from the full top-20 list
    if SHORTLIST.startswith("text:"):
        c = text_feats(c, n1, n2)
    c = c[keep_mask(c)]
    if os.environ.get("SMOKE"):
        c = c[c.rid < 20000]
    c = c.reset_index(drop=True)
    c["s1_rank"] = c.groupby("sid").score.rank(ascending=False, method="first").astype(np.int16)
    c["s1_nref"] = c.groupby("sid").rid.transform("size").astype(np.int16)
    c["s1_ntop"] = c.assign(t=c["rank"] == 0).groupby("sid").t.transform("sum").astype(np.int16)
    a, b = n1.iloc[c.sid.values].reset_index(drop=True), n2.iloc[c.rid.values].reset_index(drop=True)
    P = dict(workers=-1, dtype=np.float32)
    F = c[["score", "rank", "gap_top1", "gap_next", "s1_rank", "s1_nref", "s1_ntop"]].copy()
    for col in ["nn", "cn"]:
        x, y = a[col].tolist(), b[col].tolist()
        F[f"{col}_ratio"] = process.cpdist(x, y, scorer=fuzz.ratio, **P)
        F[f"{col}_tset"] = process.cpdist(x, y, scorer=fuzz.token_set_ratio, **P)
        F[f"{col}_tsort"] = process.cpdist(x, y, scorer=fuzz.token_sort_ratio, **P)
        F[f"{col}_partial"] = process.cpdist(x, y, scorer=fuzz.partial_ratio, **P)
        F[f"{col}_jw"] = process.cpdist(x, y, scorer=JaroWinkler.normalized_similarity, **P)
    xs, ys = a.cn.str.replace(" ", "").tolist(), b.cn.str.replace(" ", "").tolist()  # website-style names
    F["cn_nospace"] = process.cpdist(xs, ys, scorer=fuzz.ratio, **P)
    F["cn_eq"] = (a.cn.values == b.cn.values)
    x, y = a.na.tolist(), b.na.tolist()
    F["na_ratio"] = process.cpdist(x, y, scorer=fuzz.ratio, **P)
    F["na_tset"] = process.cpdist(x, y, scorer=fuzz.token_set_ratio, **P)
    F["na_partial"] = process.cpdist(x, y, scorer=fuzz.partial_token_set_ratio, **P)
    F["num_tset"] = process.cpdist(a.num.tolist(), b.num.tolist(), scorer=fuzz.token_set_ratio, **P)
    F["num_first_eq"] = (a.num.str.split(" ").str[0].values == b.num.str.split(" ").str[0].values) & (b.num.values != "")
    F["zip_eq"] = np.where((a.zip.values == "") | (b.zip.values == ""), -1, (a.zip.values == b.zip.values).astype(np.int8))
    F["b_addr_empty"] = (b.na.values == "")
    F["b_num_empty"] = (b.num.values == "")
    F["b_nonascii"] = b.nonascii.values
    F["src3"] = other.entity_id.str.startswith("S3").values[c.rid.values]
    F["len_a"], F["len_b"] = a.nn.str.len().values, b.nn.str.len().values
    # --- v2 features: consonant skeleton (Indic transliteration), truncated numbers, token rarity, name frequency
    x, y = a.sk.tolist(), b.sk.tolist()
    F["sk_ratio"] = process.cpdist(x, y, scorer=fuzz.ratio, **P)
    F["sk_tset"] = process.cpdist(x, y, scorer=fuzz.token_set_ratio, **P)
    F["num_partial"] = process.cpdist(a.num.tolist(), b.num.tolist(), scorer=fuzz.partial_token_set_ratio, **P)
    # token rarity: over the whole split, or (IDF_BY_COUNTRY=1) within each country, so a word's weight does not depend
    # on which other countries share the split
    k1, k2 = s1.country.values, other.country.values
    groups = sorted(set(k1) | set(k2)) if os.environ.get("IDF_BY_COUNTRY") else [None]
    _G["idf"] = [(_idf("t_cn", pd.concat([n1.cn[k1 == g if g else slice(None)], n2.cn[k2 == g if g else slice(None)]]).tolist()),
                  _idf("t_na", pd.concat([n1.na[k1 == g if g else slice(None)], n2.na[k2 == g if g else slice(None)]]).tolist()))
                 for g in groups]
    _G["pk"] = np.searchsorted(groups, k1) if groups != [None] else np.zeros(len(k1), np.int64)   # per S1
    _G.update(a_cn=n1.cn.tolist(), b_cn=n2.cn.tolist(), a_na=n1.na.tolist(), b_na=n2.na.tolist(),
              sa=c.sid.values, rb=c.rid.values)
    step = len(c) // 1024 + 1
    _G["job2"] = [(i, min(i + step, len(c))) for i in range(0, len(c), step)]
    with Pool(32) as p:
        W = np.concatenate(p.map(_wj_chunk, range(len(_G["job2"]))))
    for j, nm in enumerate(["cn_wj", "cn_shared_idf", "cn_a_unshared_max", "cn_b_unshared_max",
                            "na_wj", "na_shared_idf", "na_a_unshared_max", "na_b_unshared_max"]):
        F[nm] = W[:, j]
    # how common is the name: S1 entities / records in the same country with the identical core name
    key1 = n1.cn + "|" + s1.country.values; key2 = n2.cn + "|" + other.country.values
    F["a_cn_s1_count"] = key1.map(key1.value_counts()).values[c.sid.values]
    F["a_cn_rec_count"] = key1.map(key2.value_counts()).fillna(0).values[c.sid.values]
    F["b_cn_s1_count"] = key2.map(key1.value_counts()).fillna(0).values[c.rid.values]
    F = F.astype(np.float32)
    return c[["rid", "sid"]], F, s1, other


def assign(keys, p, thr):
    """keys: rid,sid; p: prob. Each rid -> its argmax sid if p >= thr."""
    d = keys.assign(p=p).sort_values("p", ascending=False).drop_duplicates("rid")
    return d[d.p >= thr]


def to_sets(pairs, s1, other):
    ids1, ids2 = s1.entity_id.values, other.entity_id.values
    out = {k: set() for k in ids1}
    for s, r in zip(ids1[pairs.sid.values], ids2[pairs.rid.values]):
        out[s].add(r)
    return out


def main_train():
    keys, F, s1, other = features("train")
    gt = load("train", "ground_truth")
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m").query("m != ''")
    rid_of = pd.Series(np.arange(len(other)), index=other.entity_id)
    sid_of = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    true_sid = np.full(len(other), -1, np.int64)
    true_sid[rid_of.loc[pairs.m].values] = sid_of.loc[pairs.source1_entity_id].values
    y = (true_sid[keys.rid.values] == keys.sid.values)
    # record fold = fold of its true S1, else a hash fold (unmatched records)
    s1f = s1_fold(s1.entity_id.tolist())
    rf = s1_fold(other.entity_id.tolist())
    rf[true_sid >= 0] = s1f[true_sid[true_sid >= 0]]
    kf = rf[keys.rid.values]
    tr, va = (kf >= 4) & (kf <= 8), kf == 9
    print("pairs", len(keys), "train", tr.sum(), "val", va.sum(), "pos rate", y[tr].mean(), flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.5, bagging_freq=1, num_threads=32, verbose=-1)
    dtr = lgb.Dataset(F[tr], y[tr]); dva = lgb.Dataset(F[va], y[va], reference=dtr)
    m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    m.save_model(f"{WORK}/gbm.txt")
    imp = pd.Series(m.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print(imp.round(0).to_string(), flush=True)
    # threshold on fold 9, macro F0.5 over fold-9 S1 entities
    p = m.predict(F[va], num_threads=32)
    kv = keys[va].reset_index(drop=True)
    kv.assign(p=p, y=y[va]).to_parquet(f"{WORK}/val_pred.parquet")
    vs1 = s1.entity_id.values[s1f == 9]
    truth = {k: set() for k in vs1}
    for s, r in zip(pairs.source1_entity_id.values, pairs.m.values):
        if s in truth:
            truth[s].add(r)
    tune(kv, p, y[va], truth, s1, other)


def tune(kv, p, yv, truth, s1, other):
    best = (0, 0.5)
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = to_sets(assign(kv, p, thr), s1, other)
        sc = f05(pred, truth)
        print(f"thr {thr:.2f}  F0.5 {sc:.5f}", flush=True)
        best = max(best, (sc, float(thr)))
    # ceiling: perfect matcher on these candidates
    ceil = to_sets(kv[yv], s1, other)
    print("best", best, "candidate ceiling F0.5", round(f05(ceil, truth), 5), flush=True)
    json.dump({"thr": best[1], "val_f05": best[0]}, open(f"{WORK}/gbm_thr.json", "w"))


def main_tune():
    s1, other = load("train", 1), pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    v = pd.read_parquet(f"{WORK}/val_pred.parquet")
    gt = load("train", "ground_truth")
    gt = gt[s1_fold(gt.source1_entity_id.tolist()) == 9]
    truth = {k: set(x for x in m.split(",") if x) for k, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    tune(v[["rid", "sid"]], v.p.values, v.y.values, truth, s1, other)


def write_tsv(path, col, sets, order):
    with open(path, "w", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for k in order:
            f.write(f"{k}\t{','.join(sorted(sets[k]))}\n")


def main_test():
    keys, F, s1, other = features("test")
    m = lgb.Booster(model_file=f"{WORK}/gbm.txt")
    thr = float(os.environ.get("THR", json.load(open(f"{WORK}/gbm_thr.json"))["thr"]))
    p = m.predict(F, num_threads=32)
    os.makedirs(OUT, exist_ok=True)
    order = s1.entity_id.tolist()
    write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(keys, s1, other), order)
    write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(assign(keys, p, thr), s1, other), order)
    print("wrote", OUT, "thr", thr, flush=True)


if __name__ == "__main__":
    {"train": main_train, "test": main_test, "tune": main_tune}[sys.argv[1]]()
