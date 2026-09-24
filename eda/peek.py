import pandas as pd
from load import load
s2 = load("train_s2")
ind = s2[s2.country == "India"]
jp = ind[ind.business_name.str.contains(r"^[^\w\s]{2,}\s", regex=True)]
print(jp.business_name.head(8).to_list())
print("first-token of junk-prefix names (India S2):", jp.business_name.str.split().str[0].value_counts().head(8).to_dict())
us = s2[s2.country == "US"]
print("first-token of junk-prefix names (US S2):", us[us.business_name.str.contains(r"^[^\w\s]{2,}\s", regex=True)].business_name.str.split().str[0].value_counts().head(8).to_dict())
for f in ["test_s1", "test_s2", "test_s3"]:
    d = load(f)
    print(f"\n--- {f} France samples")
    for r in d[d.country == "France"].sample(8, random_state=2).itertuples():
        print(f"  {r.business_name} | {r.business_address}")
