"""Noise taxonomy and similarity for ALL train ground-truth pairs (7.6M), plus cluster-level stats.

Outputs (OUT_DIR):
  pair_features.parquet   one row per true pair: noise flags + similarity scores
  01_pair_noise.json      rates by country x source, similarity quantiles, cluster stats
"""
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import (BRACKET_RE, DOMAIN_RE, INDIC_RE, JUNK_PREFIX_RE, LANDMARK_RE, LATIN_LETTER_RE,
                    LEET_RE, NULLS, NUM_RE, OUT_DIR, POBOX_RE, WORKERS, addr_tokens, address_state,
                    core_tokens, log, norm, pmap_frame, save_json, script_of, skeleton, train_pairs,
                    translit)


def flags(chunk):
    """Per-pair Python-level features (runs in worker processes)."""
    rows = []
    for s1, rid, src, country, n1, a1, n2, a2 in zip(
            chunk.s1, chunk.id, chunk.src, chunk.country, chunk.name1, chunk.addr1, chunk.name2, chunk.addr2):
        f = {}
        # ---- name
        nn1, nn2 = norm(n1), norm(n2)
        t1, t2 = translit(n1), translit(n2)
        c1, c2 = core_tokens(t1), core_tokens(t2)
        dom = DOMAIN_RE.match(n2)
        f["n_script"] = script_of(n2)
        f["n_mixed_script"] = bool(INDIC_RE.search(n2)) and bool(LATIN_LETTER_RE.search(n2))
        f["n_raw_eq"] = n1 == n2
        f["n_case_eq"] = n1.casefold() == n2.casefold()
        f["n_norm_eq"] = nn1 == nn2
        f["n_core_eq"] = c1 == c2
        f["n_core_set_eq"] = set(c1) == set(c2)
        f["n_reorder"] = sorted(t1.split()) == sorted(t2.split()) and t1 != t2
        f["n_legal_changed"] = c1 == c2 and t1 != t2
        f["n_word_added"] = set(c1) < set(c2)
        f["n_word_dropped"] = set(c2) < set(c1) and bool(c2)
        f["n_dup_word"] = any(a == b for a, b in zip(nn2.split(), nn2.split()[1:]))
        f["n_domain"] = bool(dom)
        if dom:
            body = dom.group(1).replace("-", "")
            f["n_domain_concat"] = body in ("".join(c1), "".join(t1.split()))
            f["n_domain_initials"] = body.startswith("".join(w[0] for w in c1[:2])) and not f["n_domain_concat"]
        else:
            f["n_domain_concat"] = f["n_domain_initials"] = False
        f["n_junk_prefix"] = bool(JUNK_PREFIX_RE.match(n2))
        f["n_bracket"] = bool(BRACKET_RE.search(n2))
        f["n_leet"] = bool(LEET_RE.search(n2))
        f["n_double_space"] = "  " in n2
        f["n_all_caps"] = n2.upper() == n2 and n2.lower() != n2
        f["n_lower"] = n2.lower() == n2 and n2.upper() != n2
        f["n1_skel"], f["n2_skel"] = skeleton(" ".join(c1)), skeleton(" ".join(c2))
        f["n1_t"], f["n2_t"], f["n1_n"], f["n2_n"] = " ".join(c1), " ".join(c2), nn1, nn2
        # ---- address
        an1, an2 = norm(a1), norm(a2)
        at1, at2 = translit(a1), translit(a2)
        tok1, tok2 = addr_tokens(at1), addr_tokens(at2)
        num1, num2 = set(NUM_RE.findall(a1)), set(NUM_RE.findall(a2))
        f["a_empty"] = not a2.strip()
        f["a_null_token"] = any(t in NULLS for t in an2.split())
        f["a_script"] = script_of(a2)
        f["a_all_caps"] = a2.upper() == a2 and a2.lower() != a2
        f["a_pobox_added"] = bool(POBOX_RE.search(a2)) and not POBOX_RE.search(a1)
        f["a_landmark"] = bool(LANDMARK_RE.search(a2))
        f["a_reorder"] = sorted(tok1) == sorted(tok2) and tok1 != tok2
        f["a_tokens_subset"] = set(tok2) < set(tok1) and bool(tok2)
        f["a_new_tokens"] = len(set(tok2) - set(tok1))
        f["a_dropped_tokens"] = len(set(tok1) - set(tok2))
        f["a_n_comp1"], f["a_n_comp2"] = a1.count(",") + 1, (a2.count(",") + 1 if a2.strip() else 0)
        f["num_s1_has"] = bool(num1)
        f["num_equal"] = num1 == num2 and bool(num1)
        f["num_overlap"] = bool(num1 & num2)
        f["num_rec_missing_all"] = bool(num1) and not num2
        f["num_added"] = bool(num2 - num1)
        f["num_truncated"] = any(
            (x != y and (y.startswith(x) or y.endswith(x))) for x in (num2 - num1) for y in num1)
        st1, form1 = address_state(a1, country)
        st2, form2 = address_state(a2, country)
        f["state_form1"], f["state_form2"] = form1, form2
        f["state_same"] = st1 is not None and st1 == st2
        f["state_diff"] = st1 is not None and st2 is not None and st1 != st2
        f["a1_t"], f["a2_t"], f["a1_n"], f["a2_n"] = at1, at2, an1, an2
        f["s1"], f["id"], f["src"], f["country"] = s1, rid, src, country
        rows.append(f)
    return pd.DataFrame(rows)


def sims(df):
    """Vectorised, multi-threaded pairwise similarity scores."""
    def cp(a, b, scorer):
        return process.cpdist(df[a].tolist(), df[b].tolist(), scorer=scorer, workers=WORKERS,
                              dtype=np.float32)
    out = pd.DataFrame(index=df.index)
    out["n_tsr_raw"] = cp("n1_n", "n2_n", fuzz.token_set_ratio)   # no transliteration
    out["n_tsr"] = cp("n1_t", "n2_t", fuzz.token_set_ratio)       # transliterated core tokens
    out["n_tsort"] = cp("n1_t", "n2_t", fuzz.token_sort_ratio)
    out["n_ratio"] = cp("n1_t", "n2_t", fuzz.ratio)
    out["n_partial"] = cp("n1_t", "n2_t", fuzz.partial_ratio)
    out["n_jw"] = cp("n1_t", "n2_t", JaroWinkler.normalized_similarity) * 100
    out["n_skel_ratio"] = cp("n1_skel", "n2_skel", fuzz.ratio)
    out["a_tsr_raw"] = cp("a1_n", "a2_n", fuzz.token_set_ratio)
    out["a_tsr"] = cp("a1_t", "a2_t", fuzz.token_set_ratio)
    out["a_tsort"] = cp("a1_t", "a2_t", fuzz.token_sort_ratio)
    out["a_partial"] = cp("a1_t", "a2_t", fuzz.partial_ratio)
    out.loc[df.a_empty.values, ["a_tsr_raw", "a_tsr", "a_tsort", "a_partial"]] = np.nan
    return out


def rate_table(f, cols):
    g = f.groupby(["country", "src"])[cols].mean()
    g.loc[("ALL", "ALL"), :] = f[cols].mean()
    return {f"{c}|{s}": {k: round(float(v), 5) for k, v in row.items()} for (c, s), row in g.iterrows()}


def quantiles(f, cols, by):
    out = {}
    for key, g in f.groupby(by):
        q = g[cols].quantile([.01, .05, .1, .25, .5]).round(1)
        out["|".join(map(str, key if isinstance(key, tuple) else (key,)))] = {
            c: q[c].tolist() for c in cols} | {"n": int(len(g))}
    return out


def main():
    log("loading pairs")
    j = train_pairs()
    if os.environ.get("LIMIT"):
        j = j.sample(int(os.environ["LIMIT"]), random_state=0)
    log("pairs", len(j), "workers", WORKERS)
    f = pmap_frame(flags, j)
    log("flags done")
    s = sims(f)
    f = pd.concat([f, s], axis=1)
    log("sims done")

    # derived categories
    f["n_latin"] = f.n_script == "Latin"
    f["n_unrelated"] = f.n_latin & ~f.n_domain & (f.n_tsr < 40)
    f["n_typo_like"] = f.n_latin & ~f.n_core_eq & (f.n_ratio >= 85)
    f["n_translit_gain"] = f.n_tsr - f.n_tsr_raw

    bool_cols = [c for c in f.columns if f[c].dtype == bool]
    res = {"n_pairs": int(len(f)), "rates": rate_table(f, bool_cols)}
    res["script_share"] = (f.groupby(["country", "src"]).n_script.value_counts(normalize=True)
                           .round(4).unstack(fill_value=0).to_dict(orient="index"))
    res["script_share"] = {f"{k[0]}|{k[1]}": v for k, v in res["script_share"].items()}
    res["state_forms"] = {
        c: {"s1": g.state_form1.value_counts(normalize=True, dropna=False).round(4).to_dict(),
            "rec": g.state_form2.value_counts(normalize=True, dropna=False).round(4).to_dict()}
        for c, g in f.groupby("country")}
    res["new_tokens_hist"] = f.a_new_tokens.clip(upper=6).value_counts(normalize=True).sort_index().round(4).to_dict()
    res["dropped_tokens_hist"] = f.a_dropped_tokens.clip(upper=8).value_counts(normalize=True).sort_index().round(4).to_dict()

    sim_cols = ["n_tsr_raw", "n_tsr", "n_tsort", "n_ratio", "n_partial", "n_jw", "n_skel_ratio",
                "a_tsr_raw", "a_tsr", "a_tsort", "a_partial"]
    res["sim_quantiles_by_country_src"] = quantiles(f, sim_cols, ["country", "src"])
    res["sim_quantiles_by_name_script"] = quantiles(f, ["n_tsr_raw", "n_tsr", "n_skel_ratio", "a_tsr"], ["n_script"])
    f["pcat"] = np.select(
        [~f.n_latin, f.n_domain, f.n_unrelated, f.a_empty, f.n_norm_eq],
        ["indic_name", "domain_name", "unrelated_name", "empty_address", "name_exact_norm"], "other")
    res["sim_quantiles_by_category"] = quantiles(f, ["n_tsr", "a_tsr"], ["pcat"])
    res["category_share"] = f["pcat"].value_counts(normalize=True).round(4).to_dict()

    # hard-for-name pairs: can the address alone carry them?
    weak_name = f.n_tsr < 50
    res["weak_name_pairs"] = {
        "share": float(weak_name.mean()),
        "addr_tsr_quantiles": f.loc[weak_name, "a_tsr"].quantile([.05, .1, .25, .5]).round(1).tolist(),
        "addr_empty_share": float(f.loc[weak_name, "a_empty"].mean()),
        "num_overlap_share": float(f.loc[weak_name, "num_overlap"].mean()),
    }
    both_weak = weak_name & ((f.a_tsr < 60) | f.a_empty)
    res["both_weak_share"] = float(both_weak.mean())
    res["both_weak_examples"] = f.loc[both_weak].sample(min(25, int(both_weak.sum())), random_state=0)[
        ["country", "src", "n1_n", "n2_n", "a1_n", "a2_n"]].to_dict(orient="records")

    # cluster-level (per S1)
    g = f.groupby("s1")
    cl = pd.DataFrame({
        "n": g.size(), "min_n_tsr": g.n_tsr.min(), "min_a_tsr": g.a_tsr.min(),
        "any_indic": g.n_latin.agg(lambda x: (~x).any()), "any_empty_addr": g.a_empty.any(),
        "any_unrelated": g.n_unrelated.any(), "any_domain": g.n_domain.any(),
        "n_s2": g.src.agg(lambda x: (x == "S2").sum()), "n_s3": g.src.agg(lambda x: (x == "S3").sum()),
        "country": g.country.first()})
    easy = (cl.min_n_tsr >= 80) & (cl.min_a_tsr >= 80)
    res["cluster"] = {
        "share_all_matches_easy(n_tsr>=80&a_tsr>=80)": float(easy.mean()),
        "share_with_indic_match": float(cl.any_indic.mean()),
        "share_with_empty_addr_match": float(cl.any_empty_addr.mean()),
        "share_with_unrelated_name_match": float(cl.any_unrelated.mean()),
        "share_with_domain_match": float(cl.any_domain.mean()),
        "min_n_tsr_quantiles": cl.min_n_tsr.quantile([.05, .1, .25, .5]).round(1).tolist(),
        "min_a_tsr_quantiles": cl.min_a_tsr.quantile([.05, .1, .25, .5]).round(1).tolist(),
        "s2_s3_joint": pd.crosstab(cl.n_s2.clip(upper=5), cl.n_s3.clip(upper=6)).to_dict(),
    }

    # within-source duplicates: how similar are two S2 (or two S3) records of the same S1?
    d = f[["s1", "src", "n2_t", "a2_t"]].copy()
    d["k"] = d.groupby(["s1", "src"]).cumcount()
    b = d[d.k == 1].set_index(["s1", "src"])
    a = d[d.k == 0].set_index(["s1", "src"]).loc[b.index]
    wn = process.cpdist(a.n2_t.tolist(), b.n2_t.tolist(), scorer=fuzz.token_set_ratio, workers=WORKERS)
    wa = process.cpdist(a.a2_t.tolist(), b.a2_t.tolist(), scorer=fuzz.token_set_ratio, workers=WORKERS)
    res["within_source_dups"] = {
        "n_pairs": int(len(a)),
        "name_tsr_quantiles": np.quantile(wn, [.1, .25, .5]).round(1).tolist(),
        "addr_tsr_quantiles": np.quantile(wa, [.1, .25, .5]).round(1).tolist(),
        "exact_same_name_and_addr": float(np.mean((a.n2_t.values == b.n2_t.values) & (a.a2_t.values == b.a2_t.values))),
    }

    keep = ["s1", "id", "src", "country", "pcat"] + bool_cols + sim_cols + [
        "n_script", "a_script", "a_new_tokens", "a_dropped_tokens", "state_form1", "state_form2"]
    f[keep].to_parquet(f"{OUT_DIR}/pair_features.parquet")
    save_json(res, "01_pair_noise.json")
    log("done")


if __name__ == "__main__":
    main()
