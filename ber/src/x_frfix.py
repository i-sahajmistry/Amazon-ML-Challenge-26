"""France same-address fixes from the hand review + label-free structure (review/france_review.md, x_families.py):
  suf7 restore: rule1's same-address records (strict same number / street / first + rarest name word, no pure decoy
                word; the sibling's x/test_q_rule_v10plw) whose only edit is one of the uniform 7-word suffix op
                (fils, associes, cie, services, groupe, developpement, france, +et) or no added word. Their spread over
                S1s matches unedited true records, like cie / services that the model already accepts at 97-99%.
  type reject:  accepted same-address records that add an S1 type word (club, ecole, amicale, comite...): spread like
                decoys (census), the France same-building decoy. compagnie / service (Cie / Services spellings) and legal
                forms are left out.
Writes x/restore_fr_suf7.parquet and x/reject_fr_type.parquet for x_final.py RESTORE= / REJECT=.
  python x_frfix.py"""
import pandas as pd
from x_anatomy import XD
from x_census import R1

t = pd.read_parquet(f"{XD}/fam_France.parquet")
r = pd.read_parquet(R1)[["rid", "sid"]].merge(t[["rid", "sid", "fam"]], on=["rid", "sid"], how="left")
suf = r[r.fam.isin(["SUF7", "NONE"])][["rid", "sid"]]
suf.to_parquet(f"{XD}/restore_fr_suf7.parquet")
print(f"suf7 restore: {len(suf)} of rule1's {len(r)}", flush=True)
skip = {"compagnie", "service", "eurl", "sarl", "sas", "sa", "sasu", "sci", "snc", "ei"}
typ = t[(t.pos == "SAME") & (t.fam == "TYPE") & t.acc & ~t["add"].str.split().apply(lambda a: bool(set(a) & skip))]
typ[["rid", "sid"]].to_parquet(f"{XD}/reject_fr_type.parquet")
print(f"type reject: {len(typ)} accepted same-address type-word records; by added word:\n"
      + typ["add"].value_counts().head(25).to_string(), flush=True)
