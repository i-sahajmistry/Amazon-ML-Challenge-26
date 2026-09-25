"""Pairwise features + LightGBM matcher.
  python match.py train  -> features on train candidates, fit on S1 folds 4-8, tune threshold on fold 9
  python match.py test   -> score test candidates, write output/matching_results.tsv + candidate_pairs.tsv
Decision rule: every S2/S3 record belongs to at most one S1 entity (true in train GT), so each record is
assigned to its highest-probability S1 candidate if that probability clears the threshold."""
import sys, os, json, re, numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import Pool
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from common import load, s1_fold, norm_name, core_name, norm_addr, f05, WORK, ROOT

KF = int(os.environ.get("KF", 5))  # candidates per S2/S3 record that go to the matcher
OUT = os.environ.get("OUT", f"{ROOT}/output")
_NUM = re.compile(r"\d+")
_ZIP = re.compile(r"\b\d{5,6}\b")


def _norm_chunk(df):
    nn = norm_name(df.business_name)
    na = norm_addr(df.business_address)
    return pd.DataFrame({
        "nn": nn, "cn": core_name(nn), "na": na,
        "num": na.map(lambda x: " ".join(_NUM.findall(x))),
        "zip": na.map(lambda x: (_ZIP.findall(x) or [""])[-1]),
        "nonascii": df.business_name.map(lambda x: not x.isascii()),
    }, index=df.index)


def normed(split, src):
    f = f"{WORK}/pq/norm_{split}_{src}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    df = load(split, src)
    with Pool(32) as p:
        step = len(df) // 256 + 1
        out = pd.concat(p.map(_norm_chunk, [df.iloc[i:i + step] for i in range(0, len(df), step)]))
    out.to_parquet(f)
    return out


def features(split):
    s1, n1 = load(split, 1), normed(split, 1)
    other = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c = pd.read_parquet(f"{WORK}/cand_{split}.parquet")
    # context from the full top-K list before truncating
    g = c.groupby("rid").score
    c["top1"] = g.transform("max")
    c["gap_top1"] = c.top1 - c.score
    c["next"] = g.shift(-1).fillna(-1)
    c["gap_next"] = c.score - c["next"]
    c = c[c["rank"] < KF].reset_index(drop=True)
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
    params = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.5, bagging_freq=1, num_threads=32, verbose=-1)
    dtr = lgb.Dataset(F[tr], y[tr]); dva = lgb.Dataset(F[va], y[va], reference=dtr)
    m = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    m.save_model(f"{WORK}/gbm.txt")
    imp = pd.Series(m.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print(imp.round(0).to_string(), flush=True)
    # threshold on fold 9, macro F0.5 over fold-9 S1 entities
    p = m.predict(F[va], num_threads=32)
    kv = keys[va].reset_index(drop=True)
    vs1 = s1.entity_id.values[s1f == 9]
    truth = {k: set() for k in vs1}
    for s, r in zip(pairs.source1_entity_id.values, pairs.m.values):
        if s in truth:
            truth[s].add(r)
    tune(kv, p, y[va], truth, s1, other)


def tune(kv, p, yv, truth, s1, other):
    best = (0, 0.5)
    for thr in np.arange(0.3, 0.96, 0.05):
        pred = to_sets(assign(kv, p, thr), s1, other)
        sc = f05(pred, truth)
        print(f"thr {thr:.2f}  F0.5 {sc:.5f}", flush=True)
        best = max(best, (sc, float(thr)))
    # ceiling: perfect matcher on these candidates
    ceil = to_sets(kv[yv], s1, other)
    print("best", best, "candidate ceiling F0.5", round(f05(ceil, truth), 5), flush=True)
    json.dump({"thr": best[1], "val_f05": best[0]}, open(f"{WORK}/gbm_thr.json", "w"))


def write_tsv(path, col, sets, order):
    with open(path, "w", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for k in order:
            f.write(f"{k}\t{','.join(sorted(sets[k]))}\n")


def main_test():
    keys, F, s1, other = features("test")
    m = lgb.Booster(model_file=f"{WORK}/gbm.txt")
    thr = json.load(open(f"{WORK}/gbm_thr.json"))["thr"]
    p = m.predict(F, num_threads=32)
    os.makedirs(OUT, exist_ok=True)
    order = s1.entity_id.tolist()
    write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(keys, s1, other), order)
    write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(assign(keys, p, thr), s1, other), order)
    print("wrote", OUT, "thr", thr, flush=True)


if __name__ == "__main__":
    {"train": main_train, "test": main_test}[sys.argv[1]]()
