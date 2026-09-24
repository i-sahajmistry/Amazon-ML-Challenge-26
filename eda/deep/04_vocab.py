"""Vocabulary inventories that normalisation and blocking depend on. Writes 04_vocab.json.

  * legal-form endings of names per country/source (last 1-2 tokens)
  * frequent address tokens per country/source (street types, fillers)
  * last address component (state/region form) per country/source
  * city candidates (S1: component before the state) and block sizes by city
  * frequent core name tokens and how many names consist only of frequent tokens
  * same-name groups inside S1: sizes and whether members share a state
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

from common import core_tokens, load as _load, log, norm, pmap_frame, save_json, translit

def load(name):
    df = _load(name)
    lim = int(os.environ.get("LIMIT", 0))
    return df.sample(lim, random_state=0) if lim and lim < len(df) else df


FILES = ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]


def counts(chunk):
    """Token-level counters for one chunk; merged in the parent."""
    last1, last2, addr, lastcomp, core = Counter(), Counter(), Counter(), Counter(), Counter()
    for n, a in zip(chunk.business_name, chunk.business_address):
        t = norm(n).split()
        if t:
            last1[t[-1]] += 1
            if len(t) > 1:
                last2[" ".join(t[-2:])] += 1
        for tok in set(norm(a).split()):
            if not tok.isdigit():
                addr[tok] += 1
        comps = [c.strip() for c in a.split(",") if c.strip()]
        if comps:
            lastcomp[comps[-1]] += 1
        core.update(set(core_tokens(translit(n))))
    return pd.DataFrame({"obj": [[last1, last2, addr, lastcomp, core]]})


def merged(df):
    parts = pmap_frame(counts, df[["business_name", "business_address"]])
    tot = [Counter() for _ in range(5)]
    for objs in parts.obj:
        for t, o in zip(tot, objs):
            t.update(o)
    return tot


def top(c, n, total):
    return [[k, v, round(v / total, 4)] for k, v in c.most_common(n)]


def main():
    res = {}
    for f in FILES:
        df = load(f)
        for country, g in df.groupby("country"):
            log(f, country, len(g))
            last1, last2, addr, lastcomp, core = merged(g)
            n = len(g)
            key = f"{f}|{country}"
            freq = {k for k, _ in core.most_common(200)}
            only_frequent = np.mean([set(core_tokens(translit(x))) <= freq
                                     for x in g.business_name.sample(min(n, 100_000), random_state=0)])
            res[key] = {
                "n": n,
                "name_last_token": top(last1, 40, n),
                "name_last_2_tokens": top(last2, 25, n),
                "addr_tokens": top(addr, 60, n),
                "addr_last_component": top(lastcomp, 40, n),
                "core_name_tokens": top(core, 60, n),
                "core_vocab_size": len(core),
                "share_names_only_top200_tokens": float(only_frequent),
            }

    # cities from S1: the component just before the state (US/India) or region (France)
    city = {}
    for f in ["train_s1", "test_s1"]:
        df = load(f)
        comps = df.business_address.str.split(",")
        cand = comps.map(lambda c: c[-2].strip().lower() if len(c) >= 2 else "")
        for country, g in cand.groupby(df.country):
            vc = g.value_counts()
            city[f"{f}|{country}"] = {
                "distinct": int(len(vc)), "top": [[k, int(v)] for k, v in vc.head(40).items()],
                "share_in_top10": float(vc.head(10).sum() / len(g)),
                "share_in_top100": float(vc.head(100).sum() / len(g))}
    res["s1_city_component"] = city

    # same-name groups inside S1
    s1 = load("train_s1")
    s1["nn"] = [norm(x) for x in s1.business_name]
    s1["state"] = s1.business_address.str.split(",").str[-1].str.strip().str.lower()
    grp = s1.groupby(["country", "nn"])
    size = grp.size()
    multi = size[size > 1]
    same_state = grp.state.nunique().loc[multi.index] == 1
    res["s1_same_name_groups"] = {
        "share_s1_in_group": float(multi.sum() / len(s1)),
        "group_size_hist": size.clip(upper=10).value_counts().sort_index().to_dict(),
        "largest": [[k[1], int(v)] for k, v in size.sort_values(ascending=False).head(15).items()],
        "share_groups_all_same_state": float(same_state.mean()),
    }
    save_json(res, "04_vocab.json")
    log("done")


if __name__ == "__main__":
    main()
