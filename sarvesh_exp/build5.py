"""Five E16fr3 variants (US / India / France rows edited on E16fr3's matching_results.tsv):
  E17fr3      US / India from the smaller candidate set (E10, shortlist P >= 0.005) + empty-S1 rule t 0.40; France =
              E16fr3; candidate_pairs = E17's (12,020,996 pairs)
  blankFR     E16fr3 with every France S1 left empty (leaderboard probe: isolates France / US+India)
  frresc30    + France empty-S1 rule at t 0.30 (E16fr3: 0.40), same LLM gate (judge must not say no), not in reject_fr_dd
  uiresc30    + US / India empty-S1 rule at t 0.30 (E16fr3: 0.40)
  uiresc50    US / India empty-S1 rule at t 0.50 (drop E16fr3's US / India rescues with q < 0.50)"""
import os, sys, shutil, numpy as np, pandas as pd
from common import load, WORK
E = "/scratch/scai/mtech/aib262045/amlc_exp"; B = "/scratch/scai/mtech/aib262144/amlc_team/output_E16fr3"
SX = "/scratch/scai/mtech/aib262144/AmazonMLChallenge/work/x"; X = f"{WORK}/x"
t1 = load("test", 1); o = pd.concat([load("test", 2), load("test", 3)], ignore_index=True)
S1, R, C = t1.entity_id.values, o.entity_id.values, t1.country.values
sid_of = pd.Series(np.arange(len(S1)), index=S1)
mr = pd.read_csv(f"{B}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
assert (mr.source1_entity_id.values == S1).all()
base = [set(v.split(",")) if v else set() for v in mr.matched_entity_ids.values]
owner = {r: i for i, s in enumerate(base) for r in s}
def write(name, sets, cand):
    out = f"{E}/out_{name}"; os.makedirs(out, exist_ok=True)
    with open(f"{out}/matching_results.tsv", "w", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for k, s in zip(S1, sets): f.write(f"{k}\t{','.join(sorted(s))}\n")
    shutil.copy(cand, f"{out}/candidate_pairs.tsv")
    ch = sum(a != b for a, b in zip(sets, base))
    print(f"{name}: S1 rows changed vs E16fr3 {ch:,}; by country " +
          str(pd.Series(C[[i for i, (a, b) in enumerate(zip(sets, base)) if a != b]]).value_counts().to_dict()), flush=True)
def best_rows(d):   # d: rid, sid, q -> the best row per S1
    return d.sort_values("q", ascending=False).drop_duplicates("sid")
# --- 2 blankFR
s = [set() if C[i] == "France" else set(x) for i, x in enumerate(base)]; write("E16fr3_blankFR", s, f"{B}/candidate_pairs.tsv")
# --- 3 frresc30
fr = pd.read_parquet(f"{X}/test_q_frchain.parquet"); fr = fr[fr.c == "France"].reset_index(drop=True)
for f, val in (("restore_fr_dd", 1.0), ("reject_fr_dd", 0.0)):
    r = pd.read_parquet(f"{X}/{f}.parquet")
    k = pd.Series(r.sid.values, index=r.rid.values).reindex(fr.rid.values).values == fr.sid.values
    fr.loc[k, "q"] = val
rj = pd.read_parquet(f"{X}/reject_fr_dd.parquet"); rjk = set(zip(rj.rid.values, rj.sid.values))
Lm = pd.read_parquet(f"{SX}/llm_test_v10p.parquet", columns=["rid", "sid", "llm"])
fr = fr.merge(Lm, on=["rid", "sid"], how="left")
empty = np.array([len(x) == 0 for x in base])
cand = fr[empty[fr.sid.values] & (fr.q.values >= 0.30) & (fr.q.values < 0.40) & ~(fr.llm.values <= 0)]
cand = cand[[(a, b) not in rjk for a, b in zip(cand.rid.values, cand.sid.values)] if len(cand) else []]
cand = cand[[R[x] not in owner for x in cand.rid.values]] if len(cand) else cand
b = best_rows(cand); s = [set(x) for x in base]
for rr, ss in zip(b.rid.values, b.sid.values): s[ss].add(R[rr])
print(f"frresc30: France S1s filled {len(b):,}", flush=True); write("E16fr3_frresc30", s, f"{B}/candidate_pairs.tsv")
# --- 4 US / India rescue variants (q = Sahaj's bge + judge stack)
ui = pd.read_parquet(f"{SX}/test_q_v10bplw.parquet"); ui = ui[ui.c != "France"]
cand = ui[empty[ui.sid.values] & (ui.q.values >= 0.30) & (ui.q.values < 0.40)]
cand = cand[[R[x] not in owner for x in cand.rid.values]] if len(cand) else cand
b = best_rows(cand); s = [set(x) for x in base]
for rr, ss in zip(b.rid.values, b.sid.values): s[ss].add(R[rr])
print(f"uiresc30: US / India S1s filled {len(b):,}", flush=True); write("E16fr3_uiresc30", s, f"{B}/candidate_pairs.tsv")
qk = pd.Series(ui.q.values, index=pd.MultiIndex.from_arrays([ui.rid.values, ui.sid.values]))
rid_of = pd.Series(np.arange(len(R)), index=R)
s = [set(x) for x in base]; dropped = 0
for i, x in enumerate(base):
    if C[i] == "France" or len(x) != 1: continue
    r = next(iter(x)); qq = qk.get((rid_of[r], i), np.nan)
    if qq < 0.50: s[i] = set(); dropped += 1
print(f"uiresc50: US / India rescued S1s emptied (q < 0.50) {dropped:,}", flush=True); write("E16fr3_uiresc50", s, f"{B}/candidate_pairs.tsv")
# --- 1 E17fr3
q17 = pd.read_parquet(f"{X}/test_q_v10bplw.parquet"); q17 = q17[q17.c != "France"]
s = [set(x) if C[i] == "France" else set() for i, x in enumerate(base)]
a = q17[q17.q.values >= 0.70]
for rr, ss in zip(a.rid.values, a.sid.values): s[ss].add(R[rr])
has = np.array([len(x) > 0 for x in s]); acc_r = set(a.rid.values)
c2 = q17[~has[q17.sid.values] & (q17.q.values >= 0.40)]; c2 = c2[~c2.rid.isin(acc_r)]
b = best_rows(c2)
for rr, ss in zip(b.rid.values, b.sid.values): s[ss].add(R[rr])
print(f"E17fr3: US / India from the smaller set, rescued {len(b):,}", flush=True); write("E17fr3", s, f"{E}/out_E17/candidate_pairs.tsv")
