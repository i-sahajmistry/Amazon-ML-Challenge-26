"""Write the submission from a stage-2 model's test scores (x/test_q{TAG}.parquet, written by x_nocopy.py test):
each record goes to its argmax S1 when its stage-2 score q >= THR. One threshold and one rule for every country
label; unseen countries are handled by the country-agnostic features (mined dictionaries, words.py), not here.
  python x_final.py OUT_DIR [TAG] [THR]     THR=auto: the validation threshold saved by x_judge.py (x/judge{TAG}.json)
BLANK_COUNTRY=<label> leaves that country's S1s empty (a leaderboard probe that isolates its score)."""
import os, sys, json, numpy as np, pandas as pd
from common import WORK, load
from match import to_sets, write_tsv

OUT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
THR = sys.argv[3] if len(sys.argv) > 3 else "0.70"
THR = json.load(open(f"{WORK}/x/judge{TAG}.json"))["thr"] if THR == "auto" else float(THR)

d = pd.read_parquet(f"{WORK}/x/test_q{TAG}.parquet")
s1 = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
acc = d.q.values >= THR
if os.environ.get("BLANK_COUNTRY"):
    acc &= d.c.values != os.environ["BLANK_COUNTRY"]
print(f"thr {THR}: accepted {acc.sum()} {pd.Series(d.c.values[acc]).value_counts().to_dict()}", flush=True)
k = pd.read_parquet(f"{WORK}/x/p5_test{TAG}.parquet", columns=["rid", "sid"])
os.makedirs(OUT, exist_ok=True)
order = s1.entity_id.tolist()
write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(k, s1, other), order)
write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(d[acc][["rid", "sid"]], s1, other), order)
print("wrote", OUT, flush=True)
