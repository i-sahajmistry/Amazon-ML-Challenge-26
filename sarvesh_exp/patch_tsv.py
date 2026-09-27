"""Build a submission = BASE (e.g. E16fr3) + a list of added pairs (and optionally removed pairs), France rows only
changed. Each added record leaves any other S1 it was on (one S1 per record). candidate_pairs.tsv is copied unchanged,
and every added pair must already be a candidate. Writes OUT/{matching_results,candidate_pairs}.tsv and
OUT/changes.tsv (every changed pair with names / addresses, for review).
  python patch_tsv.py BASE_DIR OUT_DIR ADD.parquet [REMOVE.parquet]   (parquet columns: s1_id, rec_id)"""
import sys, os, shutil, pandas as pd
from common import load
base, out, add = sys.argv[1], sys.argv[2], pd.read_parquet(sys.argv[3])
rem = pd.read_parquet(sys.argv[4]) if len(sys.argv) > 4 else pd.DataFrame(columns=["s1_id", "rec_id"])
os.makedirs(out, exist_ok=True)
mr = pd.read_csv(f"{base}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
sets = {k: set(v.split(",")) if v else set() for k, v in zip(mr.source1_entity_id, mr.matched_entity_ids)}
owner = {r: k for k, s in sets.items() for r in s}
cand = pd.read_csv(f"{base}/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
cset = {k: set(v.split(",")) if v else set() for k, v in zip(cand.iloc[:, 0], cand.iloc[:, 1])}
t1 = load("test", 1).set_index("entity_id"); o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).set_index("entity_id")
log = []
for s, r in zip(rem.s1_id, rem.rec_id):
    if r in sets[s]: sets[s].discard(r); owner.pop(r, None); log.append(("removed", s, r, ""))
moved = 0
for s, r in zip(add.s1_id, add.rec_id):
    assert r in cset[s], f"pair not in candidate_pairs: {s} {r}"
    if r in sets[s]: continue
    prev = owner.get(r, "")
    if prev: sets[prev].discard(r); moved += 1
    sets[s].add(r); owner[r] = s; log.append(("added", s, r, prev))
with open(f"{out}/matching_results.tsv", "w", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for k in mr.source1_entity_id: f.write(f"{k}\t{','.join(sorted(sets[k]))}\n")
shutil.copy(f"{base}/candidate_pairs.tsv", f"{out}/candidate_pairs.tsv")
L = pd.DataFrame(log, columns=["change", "s1_id", "rec_id", "moved_from_s1"])
L["s1_country"] = t1.country.reindex(L.s1_id).values
L["s1_name"] = t1.business_name.reindex(L.s1_id).values; L["s1_address"] = t1.business_address.reindex(L.s1_id).values
L["rec_name"] = o.business_name.reindex(L.rec_id).values; L["rec_address"] = o.business_address.reindex(L.rec_id).values
L.to_csv(f"{out}/changes.tsv", sep="\t", index=False)
print(f"{out}: added {int((L.change == 'added').sum())} (moved from another S1 {moved}), removed {int((L.change == 'removed').sum())}; "
      f"by country {L.s1_country.value_counts().to_dict()}", flush=True)
