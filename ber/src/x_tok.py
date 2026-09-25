"""For chosen tokens: how often each is ADDED to the record name (vs its S1) in true pairs vs distractor pairs
(distractor vs the S1 it imitates), per country, with the log-likelihood ratio.  python x_tok.py"""
import numpy as np, pandas as pd
from collections import Counter
from common import WORK
from match import normed
from harness import truth_arrays
TOKS = ("international distribution development developpement participations participation france india america usa "
        "group groupe holdings holding services service center centre partners enterprises ventures industries exports "
        "public overseas infratech associates associes sons fils company compagnie societe trading solutions global "
        "consulting management systems technologies north south east west nord sud est ouest").split()
s1, other, ts, s1f, rf = truth_arrays()
n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
c = pd.read_parquet(f"{WORK}/feats2_train.parquet", columns=["rid", "sid", "rank"])
pos = ts[c.rid.values] == c.sid.values
neg = (ts[c.rid.values] < 0) & (c["rank"].values == 0)
ctry = s1.country.values[c.sid.values]
want = set(TOKS)
rows = []
for cn in ("US", "India"):
    for lab, m in (("pos", pos), ("neg", neg)):
        mm = m & (ctry == cn)
        cnt = Counter()
        for a, b in zip(n1.nn.values[c.sid.values[mm]], n2.nn.values[c.rid.values[mm]]):
            cnt.update(want & (set(b.split()) - set(a.split())))
        rows.append((cn, lab, mm.sum(), cnt))
out = {}
for t in TOKS:
    line = []
    for cn in ("US", "India"):
        (_, _, Np, P), (_, _, Nn, N) = [r for r in rows if r[0] == cn]
        llr = np.log((P[t] + 1) / (Np + 2)) - np.log((N[t] + 1) / (Nn + 2))
        line.append(f"{cn}: pos {P[t]:6d} neg {N[t]:6d} llr {llr:6.2f}")
    print(f"{t:15s} " + " | ".join(line), flush=True)
