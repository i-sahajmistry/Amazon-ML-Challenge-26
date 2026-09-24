import pandas as pd
from load import load

gt = load("train_gt")
s1, s2, s3 = load("train_s1"), load("train_s2"), load("train_s3")
print("GT rows", len(gt), "unique S1", gt.source1_entity_id.nunique(),
      "S1 ids == GT ids:", set(gt.source1_entity_id) == set(s1.entity_id))

lists = gt.matched_entity_ids.str.strip()
pairs = (gt.assign(m=lists.where(lists != "", None).str.split(","))
           .explode("m").dropna(subset=["m"]).rename(columns={"source1_entity_id": "s1", "m": "id"}))
pairs["src"] = pairs.id.str[:2]
n = pairs.groupby("s1").size().reindex(gt.source1_entity_id).fillna(0).astype(int)
n2 = pairs[pairs.src == "S2"].groupby("s1").size().reindex(gt.source1_entity_id).fillna(0).astype(int)
n3 = pairs[pairs.src == "S3"].groupby("s1").size().reindex(gt.source1_entity_id).fillna(0).astype(int)
print("\ntotal pairs", len(pairs), pairs.src.value_counts().to_dict())
print("singleton rate", (n == 0).mean())
print("matches per S1 distribution:\n", n.value_counts().sort_index().to_string())
print("S2 matches per S1:\n", n2.value_counts().sort_index().to_string())
print("S3 matches per S1:\n", n3.value_counts().sort_index().to_string())
print("mean matches (non-singletons)", n[n > 0].mean())

ctry = s1.set_index("entity_id").country
by_c = pd.DataFrame({"n": n.values, "c": ctry.reindex(n.index).values})
print("\nby country: singleton rate / mean matches")
print(by_c.groupby("c").n.agg(singleton=lambda x: (x == 0).mean(), mean="mean", p90=lambda x: x.quantile(.9), max="max"))

print("\nid appears in >1 S1 list:", pairs.id.duplicated().sum())
all_ids = pd.concat([s2.entity_id, s3.entity_id])
matched = set(pairs.id)
print("GT ids missing from sources:", len(matched - set(all_ids)))
print("S2 matched frac", s2.entity_id.isin(matched).mean(), " S3 matched frac", s3.entity_id.isin(matched).mean())

# country consistency of pairs
c_all = pd.concat([s2, s3]).set_index("entity_id").country
pairs["c1"] = ctry.reindex(pairs.s1).values
pairs["c2"] = c_all.reindex(pairs.id).values
print("pair same-country rate", (pairs.c1 == pairs.c2).mean())
print("unmatched S2/S3 by country:")
um = pd.concat([s2, s3])
um = um[~um.entity_id.isin(matched)]
print(um.groupby([um.entity_id.str[:2], "country"]).size())
pairs[["s1", "id", "src"]].to_parquet("cache/train_pairs.parquet")
