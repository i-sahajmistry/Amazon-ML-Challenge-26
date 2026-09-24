"""Join every ground-truth pair with both records' fields -> cache/train_pairs_joined.parquet."""
import pandas as pd
from load import load

s1 = load("train_s1").rename(columns=lambda c: c + "_1")
recs = pd.concat([load("train_s2"), load("train_s3")]).rename(columns=lambda c: c + "_2")
pairs = pd.read_parquet("cache/train_pairs.parquet")
j = (pairs.merge(s1, left_on="s1", right_on="entity_id_1")
          .merge(recs, left_on="id", right_on="entity_id_2")
          .drop(columns=["entity_id_1", "entity_id_2", "country_2"])
          .rename(columns={"country_1": "country"}))
j.to_parquet("cache/train_pairs_joined.parquet")
print(j.shape)
