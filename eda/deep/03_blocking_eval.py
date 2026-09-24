"""Evaluate 02_retrieval outputs: blocking recall, F0.5 ceilings, simple-rule F0.5, test shift.

Reads OUT_DIR/retrieval/*.parquet and OUT_DIR/pair_features.parquet. Writes 03_blocking_eval.json.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from common import OUT_DIR, WORKERS, core_tokens, load, log, records, save_json, train_pairs, translit, pmap_frame

RET = f"{OUT_DIR}/retrieval"
KS = [1, 2, 3, 5, 10, 20, 50, 100]


def f05(tp, n_pred, n_true):
    """Per-entity F0.5 following the challenge rules (singletons: 1 if nothing predicted)."""
    tp, n_pred, n_true = map(np.asarray, (tp, n_pred, n_true))
    p = np.divide(tp, n_pred, out=np.zeros(len(tp)), where=n_pred > 0)
    r = np.divide(tp, n_true, out=np.zeros(len(tp)), where=n_true > 0)
    den = 0.25 * p + r
    f = np.divide(1.25 * p * r, den, out=np.zeros(len(tp)), where=den > 0)
    return np.where(n_true == 0, (n_pred == 0).astype(float), f)


def truth_for(q_ids):
    tp = train_pairs()[["s1", "id"]]
    return tp[tp.s1.isin(set(q_ids))]


def recall_tables(topk, truth, feats):
    out = {}
    n_true = truth.groupby("s1").size()
    queries = topk.s1.unique()
    n_true_q = n_true.reindex(queries).fillna(0).values
    for view, g in topk.groupby("view"):
        m = g.merge(truth, left_on=["s1", "rid"], right_on=["s1", "id"], how="inner")
        best_rank = m.groupby(["s1", "id"])["rank"].min()
        rec = {f"R@{k}": float((best_rank <= k).sum() / len(truth)) for k in KS}
        ent_full = {}
        ceil = {}
        for k in KS:
            found = (best_rank[best_rank <= k].reset_index().groupby("s1").size()
                     .reindex(queries).fillna(0).values)
            ent_full[f"all@{k}"] = float(np.mean(found[n_true_q > 0] == n_true_q[n_true_q > 0]))
            ceil[f"F05ceil@{k}"] = float(f05(found, found, n_true_q).mean())
        out[view] = {"pair_recall": rec, "entity_all_found": ent_full, "macro_f05_ceiling": ceil}
    # union of name and addr views at the same k
    u = topk[topk.view.isin(["name", "addr"])].merge(truth, left_on=["s1", "rid"], right_on=["s1", "id"])
    br = u.groupby(["s1", "id"])["rank"].min()
    out["name|addr union"] = {"pair_recall": {f"R@{k}": float((br <= k).sum() / len(truth)) for k in KS}}
    u3 = topk.merge(truth, left_on=["s1", "rid"], right_on=["s1", "id"])
    br3 = u3.groupby(["s1", "id"])["rank"].min()
    out["all-views union"] = {"pair_recall": {f"R@{k}": float((br3 <= k).sum() / len(truth)) for k in KS}}
    # recall of the 'both' view by pair category
    b = topk[topk.view == "both"].merge(truth, left_on=["s1", "rid"], right_on=["s1", "id"])
    rank = b.groupby(["s1", "id"])["rank"].min().rename("rank").reset_index()
    t = truth.merge(rank, on=["s1", "id"], how="left").merge(feats[["s1", "id", "pcat"]], on=["s1", "id"], how="left")
    t["rank"] = t["rank"].fillna(10 ** 6)
    out["both_recall_by_category"] = {
        c: {f"R@{k}": float((g["rank"] <= k).mean()) for k in [5, 10, 20, 50, 100]} | {"n": int(len(g))}
        for c, g in t.groupby("pcat")}
    return out


def score_tables(topk, truth, truescore):
    """Cosine of true pairs vs the best non-match per query, per view; singletons' top-1."""
    out = {}
    n_true = truth.groupby("s1").size()
    for view, g in topk.groupby("view"):
        g = g.merge(truth.assign(match=True), left_on=["s1", "rid"], right_on=["s1", "id"], how="left")
        g["match"] = g["match"].fillna(False).astype(bool)
        best_nonmatch = g[~g.match].groupby("s1").score.max()
        ts = truescore[truescore.view == view]
        single = g[g.s1.isin(set(g.s1) - set(n_true.index))].groupby("s1").score.max()
        qs = [.01, .05, .1, .25, .5, .75, .9, .99]
        out[view] = {
            "true_pair_cos_q": np.quantile(ts.score, qs).round(3).tolist(),
            "best_nonmatch_cos_q": np.quantile(best_nonmatch, qs).round(3).tolist(),
            "singleton_top1_cos_q": np.quantile(single, qs).round(3).tolist() if len(single) else None,
            "share_true_above_best_nonmatch": float(
                (ts.merge(best_nonmatch.rename("bn").reset_index(), on="s1").eval("score > bn")).mean()),
        }
    return out


def _prep_texts(ch):
    """Transliterated core names and addresses for a chunk of candidate pairs (worker process)."""
    return pd.DataFrame({"n1": [" ".join(core_tokens(translit(x))) for x in ch.n1],
                         "n2": [" ".join(core_tokens(translit(x))) for x in ch.n2],
                         "a1": [translit(x) for x in ch.a1], "a2": [translit(x) for x in ch.a2]})


def rule_eval(topk, truth, rev, s1_df, rec_df, kmax=20):
    """Macro F0.5 of simple rules on top-`kmax` 'both' candidates, with and without mutual-best."""
    c = topk[(topk.view == "both") & (topk["rank"] <= kmax)][["s1", "rid", "rank", "score"]].copy()
    c = c.merge(truth.assign(match=True), left_on=["s1", "rid"], right_on=["s1", "id"], how="left").drop(columns="id")
    c["match"] = c["match"].fillna(False).astype(bool)
    n1 = s1_df.set_index("entity_id").loc[c.s1]
    n2 = rec_df.set_index("entity_id").loc[c.rid]
    c["n1"], c["a1"] = n1.business_name.values, n1.business_address.values
    c["n2"], c["a2"] = n2.business_name.values, n2.business_address.values

    t = pmap_frame(_prep_texts, c[["n1", "n2", "a1", "a2"]])
    c["n_tsr"] = process.cpdist(t.n1.tolist(), t.n2.tolist(), scorer=fuzz.token_set_ratio, workers=WORKERS)
    c["a_tsr"] = process.cpdist(t.a1.tolist(), t.a2.tolist(), scorer=fuzz.token_set_ratio, workers=WORKERS)
    c.loc[t.a2.str.len().values == 0, "a_tsr"] = 0
    best = rev.set_index("rid").best_s1
    c["mutual"] = best.reindex(c.rid).values == c.s1.values

    queries = topk.s1.unique()
    n_true = truth.groupby("s1").size().reindex(queries).fillna(0).values

    def macro(mask):
        sel = c[mask]
        tp = sel.groupby("s1").match.sum().reindex(queries).fillna(0).values
        npred = sel.groupby("s1").size().reindex(queries).fillna(0).values
        f = f05(tp, npred, n_true)
        prec = tp.sum() / max(npred.sum(), 1)
        rec = tp.sum() / max(n_true.sum(), 1)
        return float(f.mean()), float(prec), float(rec)

    res = {"cosine": [], "name_addr_grid": [], "name_addr_grid_mutual": [], "combined": []}
    for thr in np.arange(0.3, 0.96, 0.05):
        res["cosine"].append([round(float(thr), 2), *macro(c.score >= thr)])
    for a in range(40, 101, 10):
        for b in range(40, 101, 10):
            m = (c.n_tsr >= a) & (c.a_tsr >= b)
            res["name_addr_grid"].append([a, b, *macro(m)])
            res["name_addr_grid_mutual"].append([a, b, *macro(m & c.mutual)])
    # a single combined score: address-weighted mean, with cosine as tie-breaker
    c["comb"] = 0.4 * c.n_tsr + 0.6 * c.a_tsr
    for thr in range(50, 100, 2):
        res["combined"].append([thr, *macro(c.comb >= thr), *macro((c.comb >= thr) & c.mutual)])
    res["best"] = {k: max(v, key=lambda r: r[-3] if k != "combined" else max(r[1], r[4]))
                   for k, v in res.items() if v}
    res["n_candidates"] = int(len(c))
    res["candidate_match_rate"] = float(c.match.mean())
    res["mutual_rate_true"] = float(c.loc[c.match, "mutual"].mean())
    res["mutual_rate_false"] = float(c.loc[~c.match, "mutual"].mean())
    return res


def shift_tables():
    """Train vs test: top-1 cosine and candidate counts; mixture estimate of distractor share."""
    out = {}
    tr = {}
    for split, countries in [("train", ["US", "India"]), ("test", ["US", "India", "France"])]:
        for country in countries:
            tag = f"{split}_{country}"
            tk = pd.read_parquet(f"{RET}/topk_{tag}.parquet")
            b = tk[tk.view == "both"]
            top1 = b[b["rank"] == 1].score
            cnt = {f"n_cand>={t}": float(b[b.score >= t].groupby("s1").size().reindex(b.s1.unique()).fillna(0).mean())
                   for t in [0.5, 0.6, 0.7, 0.8]}
            rev = pd.read_parquet(f"{RET}/rev_{tag}.parquet")
            rs = rev[rev.in_random_sample]
            out[tag] = {"top1_cos_q": np.quantile(top1, [.05, .1, .25, .5, .75]).round(3).tolist(), **cnt,
                        "rec_best_s1_cos_q": np.quantile(rs.best, [.05, .1, .25, .5, .75]).round(3).tolist()}
            if split == "train":
                bins = np.linspace(0, 1, 51)
                hm = np.histogram(rs.loc[rs.is_matched, "best"], bins=bins, density=True)[0]
                hd = np.histogram(rs.loc[~rs.is_matched, "best"], bins=bins, density=True)[0]
                tr[country] = (hm, hd, bins)
                out[tag]["distractor_share_true"] = float((~rs.is_matched).mean())
                out[tag]["rec_best_cos_q_matched"] = np.quantile(rs.loc[rs.is_matched, "best"], [.1, .25, .5]).round(3).tolist()
                out[tag]["rec_best_cos_q_distractor"] = np.quantile(rs.loc[~rs.is_matched, "best"], [.5, .75, .9]).round(3).tolist()
    # mixture fit: test record best-S1 histogram = w*matched + (1-w)*distractor (train shapes)
    for tag in ["train_US", "train_India", "test_US", "test_India", "test_France"]:
        country = tag.split("_")[1]
        rev = pd.read_parquet(f"{RET}/rev_{tag}.parquet")
        rs = rev[rev.in_random_sample].best
        if country in tr:
            hm, hd, bins = tr[country]
        else:  # unseen country: pool the train shapes
            hm = (tr["US"][0] + tr["India"][0]) / 2
            hd = (tr["US"][1] + tr["India"][1]) / 2
            bins = tr["US"][2]
        h = np.histogram(rs, bins=bins, density=True)[0]
        ws = np.linspace(0, 1, 1001)
        err = [np.sum((h - (1 - w) * hm - w * hd) ** 2) for w in ws]
        out[tag]["mixture_distractor_share_est"] = float(ws[int(np.argmin(err))])
    return out


def main():
    feats = pd.read_parquet(f"{OUT_DIR}/pair_features.parquet", columns=["s1", "id", "pcat"])
    res = {}
    for country in ["US", "India"]:
        tag = f"train_{country}"
        log(tag)
        topk = pd.read_parquet(f"{RET}/topk_{tag}.parquet")
        truth = truth_for(topk.s1.unique())
        ts = pd.read_parquet(f"{RET}/truescore_{tag}.parquet")
        rev = pd.read_parquet(f"{RET}/rev_{tag}.parquet")
        s1 = load("train_s1")
        rec = records("train")
        res[tag] = {"n_queries": int(topk.s1.nunique()), "n_true_pairs": int(len(truth)),
                    "recall": recall_tables(topk, truth, feats),
                    "scores": score_tables(topk, truth, ts)}
        log(tag, "recall+scores")
        res[tag]["rules"] = rule_eval(topk, truth, rev, s1[s1.country == country], rec[rec.country == country])
        log(tag, "rules")
    res["shift"] = shift_tables()
    save_json(res, "03_blocking_eval.json")
    log("done")


if __name__ == "__main__":
    main()
