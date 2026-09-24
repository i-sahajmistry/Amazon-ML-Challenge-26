import pandas as pd
j = pd.read_parquet("cache/train_pairs_joined.parquet")
for ctry, k in [("US", 7), ("India", 9)]:
    ids = j[j.country == ctry].s1.drop_duplicates().sample(k, random_state=3)
    for i in ids:
        g = j[j.s1 == i]
        print(f"\n=== [{ctry}] {g.business_name_1.iat[0]} | {g.business_address_1.iat[0]}")
        for r in g.itertuples():
            print(f"   {r.src}  {r.business_name_2} | {r.business_address_2}")
