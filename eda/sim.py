"""Similarity of true pairs vs random, shared-token and same-name negatives."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from load import load

from textnorm import LEGAL, NULLS, norm, name_tokens, addr_tokens, nums, is_latin


def feats(a_name, a_addr, b_name, b_addr):
    out = {}
    na, nb = norm(a_name), norm(b_name)
    out["name_exact_norm"] = na == nb
    out["name_tsr"] = fuzz.token_set_ratio(na, nb)
    out["name_ratio"] = fuzz.ratio(na, nb)
    out["addr_tsr"] = fuzz.token_set_ratio(norm(a_addr), norm(b_addr)) if b_addr.strip() else np.nan
    out["share_name_tok"] = bool(name_tokens(a_name) & name_tokens(b_name))
    out["share_addr_tok"] = bool(addr_tokens(a_addr) & addr_tokens(b_addr))
    na_, nb_ = nums(a_addr), nums(b_addr)
    out["share_number"] = bool(na_ & nb_)
    out["b_has_number"] = bool(nb_)
    out["b_latin_name"] = is_latin(b_name)
    return out


rng = np.random.default_rng(0)
j = pd.read_parquet("cache/train_pairs_joined.parquet").sample(200_000, random_state=0)
pos = pd.DataFrame([feats(*r) for r in zip(j.business_name_1, j.business_address_1, j.business_name_2, j.business_address_2)])
pos["country"], pos["src"] = j.country.values, j.src.values

# random same-country negatives
s1 = load("train_s1").sample(100_000, random_state=1)
recs = pd.concat([load("train_s2"), load("train_s3")])
negs = []
for c, g in s1.groupby("country"):
    pool = recs[recs.country == c].sample(len(g), random_state=2)
    negs.append(pd.DataFrame([feats(*r) for r in zip(g.business_name, g.business_address, pool.business_name, pool.business_address)]).assign(country=c))
rneg = pd.concat(negs)

# hard negatives: same country, share the rarest-ish (first non-legal) name token, but not in the cluster
pairs = pd.read_parquet("cache/train_pairs.parquet")
owner = dict(zip(pairs.id, pairs.s1))
def longest_tok(n):
    return max(sorted(name_tokens(n)), key=len, default="")


recs["tok0"] = [longest_tok(n) for n in recs.business_name]
rv = recs[recs.tok0 != ""].reset_index(drop=True)
idx = rv.groupby(["country", "tok0"]).indices
hn = []
for r in s1.head(60_000).itertuples():
    t = longest_tok(r.business_name)
    cand = idx.get((r.country, t))
    if cand is None:
        continue
    for k in rng.choice(cand, size=min(3, len(cand)), replace=False):
        x = rv.iloc[k]
        if owner.get(x.entity_id) != r.entity_id:
            hn.append({**feats(r.business_name, r.business_address, x.business_name, x.business_address), "country": r.country})
            break
hneg = pd.DataFrame(hn)

# hardest negatives: different entity with the exact same normalized name
rv["nn"] = [norm(n) for n in rv.business_name]
nidx = rv.groupby(["country", "nn"]).indices
sn = []
for r in s1.head(60_000).itertuples():
    cand = nidx.get((r.country, norm(r.business_name)))
    if cand is None:
        continue
    for k in cand[:20]:
        x = rv.iloc[k]
        if owner.get(x.entity_id) != r.entity_id:
            sn.append({**feats(r.business_name, r.business_address, x.business_name, x.business_address), "country": r.country})
            break
sneg = pd.DataFrame(sn)
print("S1 sampled with a same-name non-match in S2/S3:", len(sn) / 60_000)

pd.set_option("display.width", 200)
cols = ["name_exact_norm", "share_name_tok", "share_addr_tok", "share_number", "b_has_number", "b_latin_name"]
summary = pd.DataFrame({
    "pos": pos[cols].mean(), "pos_US": pos[pos.country == "US"][cols].mean(), "pos_IN": pos[pos.country == "India"][cols].mean(),
    "rand_neg": rneg[cols].mean(), "hard_neg": hneg[cols].mean(), "same_name_neg": sneg[cols].mean()})
print(summary.round(3))
print("\nmedian / p10 / p25 similarity")
for name, d in [("pos", pos), ("pos latin-name", pos[pos.b_latin_name]), ("pos non-latin-name", pos[~pos.b_latin_name]), ("rand_neg", rneg), ("hard_neg", hneg), ("same_name_neg", sneg)]:
    q = d[["name_tsr", "name_ratio", "addr_tsr"]].quantile([.1, .25, .5]).round(1)
    print(f"{name:20s}", {c: q[c].tolist() for c in q})
print("\npos: empty addr", pos.addr_tsr.isna().mean(), " neither name tok nor number shared:", (~pos.share_name_tok & ~pos.share_number).mean())
print("pos by src:\n", pos.groupby("src")[["name_exact_norm", "share_name_tok", "share_number"]].mean().round(3))
for thr in [50, 60, 70, 80, 90]:
    print(f"name_tsr>={thr}: pos {(pos.name_tsr>=thr).mean():.3f}  hard_neg {(hneg.name_tsr>=thr).mean():.3f}  | +addr_tsr>={thr}: pos {((pos.name_tsr>=thr)&(pos.addr_tsr>=thr)).mean():.3f} hard_neg {((hneg.name_tsr>=thr)&(hneg.addr_tsr>=thr)).mean():.3f} same_name_neg {((sneg.name_tsr>=thr)&(sneg.addr_tsr>=thr)).mean():.3f}")
print("hard-neg sample size", len(hneg))
