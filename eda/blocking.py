"""Recall vs. candidate volume for simple blocking keys, measured on the train ground truth."""
import numpy as np
import pandas as pd
from load import load
from textnorm import norm, name_tokens, nums

s1 = load("train_s1").sample(150_000, random_state=5)
recs = pd.concat([load("train_s2"), load("train_s3")])
pairs = pd.read_parquet("cache/train_pairs.parquet")
pairs = pairs[pairs.s1.isin(set(s1.entity_id))]
print("sampled S1", len(s1), "true pairs", len(pairs))


def key_sets(df, fn):
    return [fn(n, a) for n, a in zip(df.business_name, df.business_address)]


KEYS = {
    "longest name token": lambda n, a: {max(sorted(name_tokens(n)), key=len, default="")} - {""},
    "any name token": lambda n, a: name_tokens(n),
    "any address number": lambda n, a: nums(a),
    "longest addr number": lambda n, a: {max(sorted(nums(a)), key=len, default="")} - {""},
    "name 3-char prefix": lambda n, a: {norm(n)[:3]} - {""},
}
rec_country = dict(zip(recs.entity_id, recs.country))
s1k = s1.set_index("entity_id")
for kname, fn in KEYS.items():
    rk = pd.DataFrame({"id": recs.entity_id.values, "c": recs.country.values, "k": key_sets(recs, fn)}).explode("k").dropna()
    bsize = rk.groupby(["c", "k"]).size()
    sk = pd.DataFrame({"s1": s1.entity_id.values, "c": s1.country.values, "k": key_sets(s1, fn)}).explode("k").dropna()
    # candidate volume = sum of block sizes hit by each S1 (upper bound; ignores overlap between keys)
    vol = sk.merge(bsize.rename("n").reset_index(), on=["c", "k"], how="left").n.fillna(0)
    # recall: true pair shares at least one key
    rset = rk.groupby("id").k.agg(set)
    sset = sk.groupby("s1").k.agg(set)
    hit = [bool(sset.get(a, set()) & rset.get(b, set())) for a, b in zip(pairs.s1, pairs.id)]
    per_s1 = vol.groupby(sk.s1.values).sum()
    print(f"{kname:22s} recall={np.mean(hit):.3f}  cand/S1 mean={per_s1.reindex(s1.entity_id).fillna(0).mean():,.0f} median={per_s1.median():,.0f}  max block={bsize.max():,}")

# distractors: do unmatched S2/S3 records copy S1 names?
allp = pd.read_parquet("cache/train_pairs.parquet")
matched = set(allp.id)
s1_names = set(norm(n) for n in load("train_s1").business_name)
rs = recs.sample(300_000, random_state=6)
rs["m"] = rs.entity_id.isin(matched)
rs["name_in_s1"] = [norm(n) in s1_names for n in rs.business_name]
print("\nrecord's normalized name exists verbatim in S1:", rs.groupby("m").name_in_s1.mean().to_dict())
print("distractor samples:")
for r in rs[~rs.m].sample(10, random_state=7).itertuples():
    print(f"   {r.entity_id[:2]} [{r.country}] {r.business_name} | {r.business_address}")
