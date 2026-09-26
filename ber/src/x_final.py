"""Write the submission from a stage-2 model's test scores (x/test_q{TAG}.parquet, written by x_nocopy.py test):
each record goes to its argmax S1 when its stage-2 score q >= THR. One threshold and one rule for every country
label; unseen countries are handled by the country-agnostic features (mined dictionaries, words.py), not here.
  python x_final.py OUT_DIR [TAG] [THR]     THR=auto: the validation threshold saved by x_judge.py (x/judge{TAG}.json)
BLANK_COUNTRY=<label> leaves that country's S1s empty (a leaderboard probe that isolates its score).
THR_UNSEEN=<t> uses threshold t for every country label with no training labels (absent from train), the same rule for
any such country; its value comes from the leave-one-country-out run (loco_eval.py), never from the leaderboard.
Q_UNSEEN=<tag> takes the records of every country label absent from train from x/test_q{tag}.parquet (another stage-1+2
variant, e.g. DROP="^ce" in loco_unseen.pbs) and the rest from x/test_q{TAG}.parquet; Q_MIX=<a> instead mixes the two
where both pick the same S1, (1-a)*q + a*q_variant, as loco_blend.py (where they pick different S1s the rule of loco_blend.py)."""
import os, sys, json, numpy as np, pandas as pd
from common import WORK, load
from match import to_sets, write_tsv

OUT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
THR = sys.argv[3] if len(sys.argv) > 3 else "0.70"
THR = json.load(open(f"{WORK}/x/judge{TAG}.json"))["thr"] if THR == "auto" else float(THR)

d = pd.read_parquet(f"{WORK}/x/test_q{TAG}.parquet")
s1 = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
unseen = sorted(set(d.c.values) - set(load("train", 1).country.values))
if os.environ.get("Q_UNSEEN"):   # one row per record in both files: its argmax S1 under that variant
    v = pd.read_parquet(f"{WORK}/x/test_q{os.environ['Q_UNSEEN']}.parquet").set_index("rid").reindex(d.rid.values)
    assert v.sid.notna().all(), "the variant covers different records"
    a = float(os.environ.get("Q_MIX", 1.0))
    same = v.sid.values == d.sid.values
    take_v = ~same & ((a > 0.5) | ((a == 0.5) & (v.q.values > d.q.values)))   # the same rule as loco_blend.py
    use = np.isin(d.c.values, unseen) | np.isin(v.c.values, unseen)
    q_new = np.where(same, (1 - a) * d.q.values + a * v.q.values, np.where(take_v, v.q.values, d.q.values))
    sw = use & take_v
    d.loc[sw, "sid"], d.loc[sw, "c"] = v.sid.values[sw].astype(d.sid.dtype), v.c.values[sw]
    d.loc[use, "q"] = q_new[use]
    print(f"records of {unseen} from test_q{os.environ['Q_UNSEEN']} (mix {a}): {use.sum():,}, "
          f"same S1 as the base {same[use].mean():.4f}", flush=True)
thr = np.full(len(d), THR)
if os.environ.get("THR_UNSEEN"):
    thr[np.isin(d.c.values, unseen)] = float(os.environ["THR_UNSEEN"])
    print(f"countries without training labels {unseen}: threshold {os.environ['THR_UNSEEN']}", flush=True)
acc = d.q.values >= thr
if os.environ.get("BLANK_COUNTRY"):
    acc &= d.c.values != os.environ["BLANK_COUNTRY"]
print(f"thr {THR}: accepted {acc.sum()} {pd.Series(d.c.values[acc]).value_counts().to_dict()}", flush=True)
k = pd.read_parquet(f"{WORK}/x/p5_test{TAG}.parquet", columns=["rid", "sid"])
os.makedirs(OUT, exist_ok=True)
order = s1.entity_id.tolist()
write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(k, s1, other), order)
write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(d[acc][["rid", "sid"]], s1, other), order)
print("wrote", OUT, flush=True)
