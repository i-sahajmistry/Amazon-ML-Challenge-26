"""Label-free sanity check of a test output: per-country no-match rate and matches per S1 vs the training prior
(US and India both have 5.6% no-match and 3.46 matches per entity in the ground truth).
  python x_prior.py <output_dir> [<output_dir> ...]"""
import sys, pandas as pd
from common import load

c = load("test", 1).set_index("entity_id").country
for d in sys.argv[1:]:
    m = pd.read_csv(f"{d}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    k = m.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0)
    g = pd.DataFrame({"c": c.loc[m.source1_entity_id].values, "k": k.values}).groupby("c").k
    print(d, "\n", pd.DataFrame({"S1": g.size(), "no_match": g.apply(lambda x: (x == 0).mean()).round(4),
                                "per_S1": g.mean().round(3)}).to_string(), flush=True)
