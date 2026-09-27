"""Drop from an add list the records E16fr3 already places on some S1 (restore only records left unmatched)."""
import sys, pandas as pd
B = "/scratch/scai/mtech/aib262144/amlc_team/output_E16fr3/matching_results.tsv"
mr = pd.read_csv(B, sep="\t", dtype=str, keep_default_na=False)
owned = {r for v in mr.matched_entity_ids.values if v for r in v.split(",")}
a = pd.read_parquet(sys.argv[1]); k = ~a.rec_id.isin(owned)
a[k].to_parquet(sys.argv[2]); print(f"{sys.argv[1]}: {len(a):,} -> {k.sum():,} not already placed", flush=True)
