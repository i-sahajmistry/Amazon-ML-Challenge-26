"""Assemble a submission from a model's test-time stage-2 probabilities (x/test_q{TAG}.parquet, written by
x_nocopy.py test or x_thr.py): each record goes to its best S1 when q >= THR. One rule for every country.
  python x_final.py OUT_DIR [TAG] [THR]     # TAG "v5" = untagged (run.sh drops empty args)
  FROM=France:_v9sw[:min] THR_C=France:0.8 ...  # one country's q from another model (or the min of both) /
                                            # its own threshold
  NUMVETO=US,France MAXD=13 ...            # reject decoy groups at a shifted house number (see below)
  LLMVETO=0 ...                             # reject rows whose LLM judge margin (x_llmhi.py) is below 0
  BLANK=<country> python x_final.py ...    # leaderboard probe: that country's S1s left empty, so the score
                                            # difference to the full file isolates that country
  python x_final.py sweep [TAG]             # per country: S1s with no match / matches per S1, by threshold"""
import os, sys, numpy as np, pandas as pd
from common import WORK, load
from match import to_sets, write_tsv

OUT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "v5"
TAG = "" if TAG == "v5" else TAG
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70

d = pd.read_parquet(f"{WORK}/x/test_q{TAG}.parquet")
for spec in filter(None, os.environ.get("FROM", "").split(",")):   # "France:_v9sw": that country's q from another model
    c, t, *mode = spec.split(":")                                   # "France:_v9sw:min": accept only what both models
    e = pd.read_parquet(f"{WORK}/x/test_q{t}.parquet")              # accept, for the same S1
    e = e[e.c == c]
    if mode == ["min"]:
        m = d[d.c == c].merge(e[["rid", "sid", "q"]], on="rid", how="left", suffixes=("", "_b"))
        e = m.assign(q=np.where(m.sid.values == m.sid_b.values, np.minimum(m.q.values, m.q_b.values), 0.0))[["rid", "sid", "q", "c"]]
    d = pd.concat([d[d.c != c], e], ignore_index=True)
if os.environ.get("SAVE_Q"):   # keep the combined q (e.g. the France veto) as x/test_q{SAVE_Q}, a self-training seed
    d.to_parquet(f"{WORK}/x/test_q{os.environ['SAVE_Q']}.parquet")
    sys.exit()
if os.environ.get("LLMVETO"):
    # the rows the stage-2 blend never sends to the LLM judge (countries without training labels, accepted at q >= 0.99;
    # scored by x_llmhi.py): reject those whose margin logit(yes) - logit(no) is below LLMVETO (0: the judge says no)
    from x_llmhi import scored
    h = scored("test")
    no = pd.MultiIndex.from_frame(h[h.llm.values < float(os.environ["LLMVETO"])][["rid", "sid"]])
    k = pd.MultiIndex.from_arrays([d.rid.values, d.sid.values]).isin(no)
    print("LLM veto:", pd.Series(d.c.values[k & (d.q.values >= THR)]).value_counts().to_dict(), "accepted records", flush=True)
    d.loc[k, "q"] = 0.0
s1 = load("test", 1); other = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
if OUT == "sweep":
    n = s1.country.value_counts()
    for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        a = d[d.q >= t]
        has = np.bincount(a.sid.values, minlength=len(s1)) > 0
        print(f"thr {t:.2f}  " + "  ".join(f"{c} {1 - has[s1.country.values == c].mean():.4f}/{(a.c == c).sum() / n[c]:.3f}"
                                          for c in n.index), flush=True)
    sys.exit()
if os.environ.get("NUMVETO"):
    # decoy groups: test distractors come in groups at a shifted house number (VX Centre 14 -> two decoys at 15), which
    # train distractors almost never do (same-number peers: 52-78% of test records below q 0.5, 7% of train distractors).
    # Reject a record whose first number is near its S1's but not shared with it when another record claiming that S1
    # has the same number. NUMVETO=US,France lists the countries; MAXD the largest shift.
    from match import normed
    ints = lambda x: [int(t[:15]) for t in x.split()]
    R = [ints(x) for x in pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True).num.values[d.rid.values]]
    S = [ints(x) for x in normed("test", 1).num.values[d.sid.values]]
    kr = np.array([x[0] if x else -1 for x in R]); ks = np.array([x[0] if x else -1 for x in S])
    peer = d.assign(k=kr).groupby(["sid", "k"]).sid.transform("size").values > 1
    apart = np.array([bool(a) and bool(b) and a[0] not in b and b[0] not in a for a, b in zip(R, S)])
    veto = (apart & (np.abs(kr - ks) <= int(os.environ.get("MAXD", 13))) & peer & (kr >= 0)
            & d.c.isin(os.environ["NUMVETO"].split(",")).values)
    print("number veto:", pd.Series(d.c.values[veto & (d.q.values >= THR)]).value_counts().to_dict(), "accepted records", flush=True)
    d.loc[veto, "q"] = 0.0
thr = np.full(len(d), THR)
for spec in filter(None, os.environ.get("THR_C", "").split(",")):  # "France:0.8": that country's threshold
    c, t = spec.split(":")
    thr[d.c.values == c] = float(t)
acc = d.q.values >= thr
if os.environ.get("BLANK"):
    acc &= d.c.values != os.environ["BLANK"]
print(f"thr {THR}: accepted {acc.sum()}", pd.Series(d.c.values[acc]).value_counts().to_dict(), flush=True)
k = pd.read_parquet(f"{WORK}/x/p5_test{TAG}.parquet", columns=["rid", "sid"])
os.makedirs(OUT, exist_ok=True)
order = s1.entity_id.tolist()
write_tsv(f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids", to_sets(k, s1, other), order)
write_tsv(f"{OUT}/matching_results.tsv", "matched_entity_ids", to_sets(d[acc][["rid", "sid"]], s1, other), order)
print("wrote", OUT, flush=True)
