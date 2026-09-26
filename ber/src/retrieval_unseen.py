"""Does the bi-encoder retrieve worse for a country it never trained on?
Compares two candidate files for the same train records: the normal one (bi-encoder fine-tuned on S1 folds 0-3 of
every train country) and the leave-one-country-out one (fine-tuned without HOLDOUT). Reports, per country, the
share of matched records whose true S1 is retrieved at rank 1, within the top 5 and within the top 20, and the
median top-1 minus top-2 cosine gap (the near-tie measure). Only records of S1 folds 4-9 are counted, so the
normal bi-encoder has not seen these entities either; the difference is the unseen-country effect.
  NORMAL=<work dir> LOCO=<work dir> HOLDOUT_NAME=India python retrieval_unseen.py   (HOLDOUT itself stays unset)"""
import os
import zlib

import numpy as np
import pandas as pd

from common import load

NORMAL = os.path.expanduser(os.environ["NORMAL"])
LOCO = os.path.expanduser(os.environ["LOCO"])
pd.set_option("display.width", 200)

s1 = load("train", 1)
other = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
gt = load("train", "ground_truth")
pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m").query("m != ''")
rid_of = pd.Series(np.arange(len(other)), index=other.entity_id)
ts = np.full(len(other), -1, np.int64)
ts[rid_of.loc[pairs.m].values] = pd.Series(np.arange(len(s1)), index=s1.entity_id).loc[pairs.source1_entity_id].values
s1f = np.array([zlib.crc32(x.encode()) % 10 for x in s1.entity_id.values], dtype=np.int8)
recs = np.flatnonzero((ts >= 0) & (s1f[np.maximum(ts, 0)] >= 4))
c_rec = other.country.values
rows = []
for name, work in (("normal bi-encoder", NORMAL), ("bi-encoder without " + os.environ.get("HOLDOUT_NAME", "?"), LOCO)):
    c = pd.read_parquet(f"{work}/cand_train.parquet", columns=["rid", "sid", "score", "rank"])
    isr = np.zeros(len(ts), bool); isr[recs] = True
    c = c[isr[c.rid.values]]
    hit = c[c.sid.values == ts[c.rid.values]]
    rank = np.full(len(ts), 99, np.int16); rank[hit.rid.values] = hit["rank"].values
    top = c[c["rank"] < 2].sort_values(["rid", "rank"])
    sc = top.score.values.reshape(-1, 2); gap = pd.Series(sc[:, 0] - sc[:, 1], index=top.rid.values[::2])
    for g in sorted(set(c_rec[recs])):
        r = recs[c_rec[recs] == g]
        rows.append({"retrieval": name, "country": g, "records": len(r), "true S1 @1": (rank[r] < 1).mean(),
                     "@5": (rank[r] < 5).mean(), "@20": (rank[r] < 20).mean(),
                     "median top1-top2 gap": gap.reindex(r).median()})
print("True S1 retrieved, matched records of S1 folds 4-9:")
print(pd.DataFrame(rows).round(4).to_string(index=False))
