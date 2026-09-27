"""Step 1 of E10 (tighter blocking for countries with labels): the text-shortlist probability P of every pair in
feats2_{split} (the 12.76M test / train pairs the matcher scores), saved in feats2 row order as x/shortP_{split}.npy.
Same computation as match.keep_mask with SHORTLIST=text (retrieval features over the full top-20 + text features)."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK
from match import normed, retrieval_feats, text_feats, SHORT_F, TXT
m = lgb.Booster(model_file=f"{WORK}/shortlist_text.txt")
for split in ("test", "train"):
    c = retrieval_feats(pd.read_parquet(f"{WORK}/cand_{split}.parquet"))
    n1 = normed(split, 1); n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    c = text_feats(c, n1, n2)
    c["P"] = m.predict(c[SHORT_F + TXT], num_threads=int(os.environ.get("NT", 8)))
    k = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"])
    P = k.merge(c[["rid", "sid", "P"]], on=["rid", "sid"], how="left").P.values.astype(np.float32)
    print(f"{split}: feats2 rows {len(k):,}, missing P {np.isnan(P).sum():,}, P<0.001 {np.nansum(P < 0.001):,}; "
          + "  ".join(f">={t}: {(P >= t).sum():,}" for t in (0.001, 0.002, 0.003, 0.005, 0.01)), flush=True)
    np.save(f"{os.environ['AMLC_ROOT']}/work/x/shortP_{split}.npy", P)
    del c
