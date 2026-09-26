"""Cross-encoder self-training for test countries that have no labelled training records (an open set, found from the
data; France in this test set). The cross-encoders learned from US / India text only; this adapts them to the new
country's names and addresses with its own confident predictions (allowed: self-training on the test records).

Pseudo-labels from a finished run's stage-2 test scores (x/test_q{TAG}w.parquet, each record's argmax S1):
  a record whose argmax pair has q >= HI     -> that pair 1, the record's other shortlisted pairs 0 (hard negatives)
  a record whose argmax pair has q <= LO     -> all its pairs 0
Each fine-tune continues from the trained cross-encoder (CE_DIR) for one epoch on the pseudo-labelled pairs of one
half of those records plus as many labelled training pairs (the CE's own training set: retrieved top-5 of S1 folds
0-3), so it does not forget US / India; it then re-scores the other half. No record is scored by a model that saw its
own pseudo-label. Rows of other countries keep their original scores.
  CE_TEXT=norm|raw CE_DIR=x/ce|x/ce_raw CE_TAG=|_raw TAG=_v9f python x_ce_st.py
      -> x/ce{CE_TAG}st_test.npy (x/ce{CE_TAG}_test.npy with the new country's rows re-scored)"""
import os, math, time, numpy as np, pandas as pd, torch
import torch.nn.functional as Fn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from common import WORK, load
from harness import truth_arrays
import x_ce3 as C

XD = C.XD
TAG = os.environ.get("TAG", "_v9f")
CE_TAG = os.environ.get("CE_TAG", "")
HI, LO = float(os.environ.get("ST_HI", 0.95)), float(os.environ.get("ST_LO", 0.05))
LR = float(os.environ.get("ST_LR", 2e-5))
REPLAY = float(os.environ.get("ST_REPLAY", 1.0))
key = lambda a, b: a.astype(np.int64) * (1 << 22) + b


def joined(tag):
    """train and test token stores of one side stacked: test item j sits at n_train + j."""
    fa, oa = C.store("train", tag)
    fb, ob = C.store("test", tag)
    return (np.concatenate([fa, fb]), np.concatenate([oa, ob[1:] + oa[-1]])), len(oa) - 1


def pseudo_labels():
    """pair labels for the shortlisted test pairs (feats2_test / ce_test row order): 1, 0 or -1 (unlabelled)."""
    _, _, ts, _, _ = truth_arrays()
    labelled = set(pd.concat([load("train", 2), load("train", 3)], ignore_index=True).country.values[ts >= 0])
    ct = pd.concat([load("test", 2), load("test", 3)], ignore_index=True).country.values
    unl = ~np.isin(ct, list(labelled))
    q = pd.read_parquet(f"{XD}/test_q{TAG}w.parquet", columns=["rid", "sid", "q"])
    q = q[unl[q.rid.values]]
    pos, neg = q[q.q.values >= HI], q.rid.values[q.q.values <= LO]
    k = pd.read_parquet(f"{WORK}/feats2_test.parquet", columns=["rid", "sid"])
    r, s = k.rid.values, k.sid.values
    y = np.full(len(k), -1, np.int8)
    y[np.isin(r, pos.rid.values)] = 0
    y[np.isin(key(r, s), key(pos.rid.values, pos.sid.values))] = 1
    y[np.isin(r, neg)] = 0
    print(f"self-training countries {sorted(set(ct[unl]))}: records {q.rid.nunique()}, pseudo match {len(pos)}, "
          f"none {len(neg)}; pairs labelled {(y >= 0).sum()} ({(y == 1).mean() / max((y >= 0).mean(), 1e-9):.3f} positive)",
          flush=True)
    return r, s, y, unl[r]


def finetune(A, B, a, b, y, rng):
    lens = C.plen(A, B, a, b)
    batches = C.bucketed(lens, C.BS, rng)
    model = AutoModelForSequenceClassification.from_pretrained(C.CE, num_labels=1, attn_implementation="sdpa").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * len(batches)), len(batches))
    model.train(); t0 = time.time(); run = 0.0
    for step, bt in enumerate(C.loader(C.Pairs(A, B, a, b, y, batches=batches))):
        with torch.autocast("cuda", torch.bfloat16):
            z = model(input_ids=bt["input_ids"].cuda(non_blocking=True),
                      attention_mask=bt["attention_mask"].cuda(non_blocking=True)).logits.squeeze(-1)
        loss = Fn.binary_cross_entropy_with_logits(z.float(), bt["y"].cuda(non_blocking=True))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0:
            print(f"  step {step}/{len(batches)}  loss {run:.4f}  {(step + 1) * C.BS / (time.time() - t0):.0f} pairs/s", flush=True)
    return model


def main():
    A, n1 = joined("s1")
    B, n2 = joined("other")
    r, s, y, unl_row = pseudo_labels()
    _, _, ts, _, rf = truth_arrays()
    c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
    c = c[(c["rank"].values < 5) & (rf[c.rid.values] < 4)]                  # the CE's own training pairs
    rng = np.random.default_rng(0)
    out = np.load(f"{XD}/ce{CE_TAG}_test.npy").copy()
    half = (r % 2).astype(bool)
    for h in (False, True):
        pl = (y >= 0) & (half != h)
        rep = rng.choice(len(c), min(len(c), int(REPLAY * pl.sum())), replace=False)
        a = np.r_[s[pl] + n1, c.sid.values[rep]]                             # S1 index into the joined store
        b = np.r_[r[pl] + n2, c.rid.values[rep]]
        yy = np.r_[y[pl] == 1, ts[c.rid.values[rep]] == c.sid.values[rep]].astype(np.float32)
        print(f"half {int(h)}: fine-tune on {pl.sum()} pseudo-labelled + {len(rep)} labelled pairs", flush=True)
        model = finetune(A, B, a, b, yy, rng)
        part = unl_row & (half == h)
        out[part] = C.logits(model, A, B, s[part] + n1, r[part] + n2)
        del model; torch.cuda.empty_cache()
    old = np.load(f"{XD}/ce{CE_TAG}_test.npy")
    print(f"re-scored {unl_row.sum()} pairs; mean |change| {np.abs(out[unl_row] - old[unl_row]).mean():.3f}", flush=True)
    np.save(f"{XD}/ce{CE_TAG}st_test.npy", out)


if __name__ == "__main__":
    main()
