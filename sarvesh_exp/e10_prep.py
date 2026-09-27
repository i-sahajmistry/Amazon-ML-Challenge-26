"""E10 step 2 prep: a sandbox whose scored pairs are cut at shortlist P >= TAU for the countries with training labels
(France keeps every pair). Filters feats2, the three cross-encoder score arrays and the extra features by the same
row mask, so every array stays aligned with feats2. Train pairs are all cut (they are US / India)."""
import os, sys, numpy as np, pandas as pd
TAU = float(sys.argv[1]); R = sys.argv[2]
O = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work"; E = "/scratch/scai/mtech/aib262045/amlc_exp/work/x"
os.makedirs(f"{R}/work/x", exist_ok=True)
test_cty = None
for split in ("train", "test"):
    P = np.load(f"{E}/shortP_{split}.npy")
    k = pd.read_parquet(f"{O}/feats2_{split}.parquet")
    keep = P >= TAU
    if split == "test":
        s1c = pd.read_parquet(f"{O}/pq/test_source1.parquet", columns=["country"]).country.values
        keep |= ~np.isin(s1c[k.sid.values], ["US", "India"])        # France (no labels) keeps its full list
    print(f"{split}: keep {keep.sum():,} of {len(keep):,} pairs ({keep.mean():.4f})", flush=True)
    k[keep].reset_index(drop=True).to_parquet(f"{R}/work/feats2_{split}.parquet")
    for t in ("_n10", "_r10", "_b10"):
        np.save(f"{R}/work/x/ce{t}_{split}.npy", np.load(f"{O}/x/ce{t}_{split}.npy")[keep])
    x = pd.read_parquet(f"{O}/archive_extra_v10/extra_{split}.parquet")
    assert len(x) == len(keep), (len(x), len(keep))
    x[keep].reset_index(drop=True).to_parquet(f"{R}/work/x/extra_{split}.parquet")
print("prepared", R, flush=True)
