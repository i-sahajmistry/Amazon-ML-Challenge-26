"""Does the distractor vocabulary found on train also show up in France? Share of top-1 test pairs whose added words
are strongly distractor-like (LLR < -5), by country, vs train US/India.  python x_llr_country.py"""
import numpy as np, pandas as pd
from common import WORK, load
for split in ("train", "test"):
    k = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["sid", "rank"])
    L = pd.read_parquet(f"{WORK}/x/llr_{split}.parquet", columns=["n_add_min", "a_add_min", "n_add_cnt", "n_drop_cnt"])
    top = k["rank"].values == 0
    c = load(split, 1).country.values[k.sid.values[top]]
    d = L[top].assign(c=c)
    g = d.groupby("c")
    print(split, pd.DataFrame({"pairs": g.size(), "name_word_llr<-5": g.n_add_min.apply(lambda x: (x < -5).mean()).round(4),
                               "addr_word_llr<-5": g.a_add_min.apply(lambda x: (x < -5).mean()).round(4),
                               "name_words_added": g.n_add_cnt.mean().round(3), "name_words_dropped": g.n_drop_cnt.mean().round(3)}).to_string(), flush=True)
