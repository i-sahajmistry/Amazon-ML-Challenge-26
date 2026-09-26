"""Go / no-go for a new cross-encoder: on records of S1 fold 9 (seen by no cross-encoder), the share of real records
whose argmax candidate is the right S1, and the pair AUC, for the new model vs the existing e5 cross-encoders scored
on the same shortlisted pairs (x/ce_train.npy, x/ce_raw_train.npy).
  CE_DIR=... CE_TEXT=raw python g10_ce_compare.py"""
import os, numpy as np, pandas as pd
from transformers import AutoModelForSequenceClassification
from common import WORK
from harness import truth_arrays
import x_ce3 as C

N_REC = int(os.environ.get("N_REC", 300_000))


def argmax_acc(r, ok, sc, real):
    d = pd.DataFrame({"rid": r, "ok": ok, "sc": sc})
    best = d.loc[d.groupby("rid").sc.idxmax()]
    return best.ok.values[real[best.rid.values]].mean()


def main():
    _, _, ts, _, rf = truth_arrays()
    k = pd.read_parquet(f"{WORK}/feats2_train.parquet", columns=["rid", "sid"])
    r, s = k.rid.values, k.sid.values
    rng = np.random.default_rng(1)
    recs = np.unique(r[rf[r] == 9])
    v = np.flatnonzero(np.isin(r, rng.choice(recs, min(len(recs), N_REC), replace=False)))
    ok, real = ts[r[v]] == s[v], ts >= 0
    A, B = C.store("train", "s1"), C.store("train", "other")
    model = AutoModelForSequenceClassification.from_pretrained(C.CE, num_labels=1, attn_implementation="sdpa").cuda()
    new = C.logits(model, A, B, s[v], r[v])
    for name, sc in (("new model", new), ("e5 normalised", np.load(f"{WORK}/x/ce_train.npy")[v]),
                     ("e5 raw text", np.load(f"{WORK}/x/ce_raw_train.npy")[v])):
        print(f"{name:14s} fold-9 argmax right S1 {argmax_acc(r[v], ok, sc, real):.5f}   pair AUC {C.auc(ok, sc):.5f}",
              flush=True)


if __name__ == "__main__":
    main()
