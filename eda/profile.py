import re
import pandas as pd
from load import load

pd.set_option("display.width", 200)
DEV = re.compile(r"[ऀ-ॿ]")
NONASCII = re.compile(r"[^\x00-\x7F]")

for n in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
    df = load(n)
    print(f"\n######## {n}  rows={len(df):,}")
    print("dup entity_id:", df.entity_id.duplicated().sum(),
          "| prefix ok:", df.entity_id.str[:3].value_counts().to_dict())
    for c in ["business_name", "business_address", "country"]:
        s = df[c]
        print(f"  {c}: empty={(s.str.strip() == '').mean():.4f}  "
              f"len mean={s.str.len().mean():.1f} p50={s.str.len().median():.0f} "
              f"p99={s.str.len().quantile(.99):.0f} max={s.str.len().max()}")
    print("  country:", df.country.value_counts(dropna=False).to_dict())
    for ctry, g in df.groupby("country"):
        nm, ad = g.business_name, g.business_address
        print(f"   [{ctry}] n={len(g):,} name_devanagari={nm.str.contains(DEV).mean():.3f} "
              f"addr_devanagari={ad.str.contains(DEV).mean():.3f} "
              f"name_nonascii={nm.str.contains(NONASCII).mean():.3f} "
              f"addr_empty={(ad.str.strip()=='').mean():.3f} "
              f"name_upper={(nm == nm.str.upper()).mean():.3f} addr_upper={(ad == ad.str.upper()).mean():.3f}")
    print("  exact dup (name,addr,country) rows:",
          df.duplicated(["business_name", "business_address", "country"]).sum())
    print("  dup names:", df.business_name.duplicated().sum())
