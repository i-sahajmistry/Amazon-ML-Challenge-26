"""Cross-encoder re-scorer for candidate pairs (S1 entity text, S2/S3 record text).
Initialised from the fine-tuned e5 (MIT) and trained only on S1 folds 0-3 (records owned by a fold 0-3 entity,
distractors with hash fold 0-3) - the embedder's folds - so its scores are out-of-sample for folds 4-9 and test.
  python x_ce.py prep               # tokenise every record once
  python x_ce.py train              # fine-tune on folds 0-3 pairs, then a quick fold-9 check vs stage-1
  python x_ce.py score train|test   # logit for every top-5 pair (feats2 row order); NaN where in-sample
  SELF=x/test_q_v8w.parquet CE_INIT=x/ce_raw ... python x_ce3.py train   # continue a CE on confident test decisions
                                    # of the countries without train labels (self-training, labels never used)
"""
import os, sys, math, time, numpy as np, pandas as pd, torch
from multiprocessing import Pool
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from common import WORK
from match import normed
from harness import truth_arrays

XD = f"{WORK}/x"; os.makedirs(XD, exist_ok=True)
BASE = os.environ.get("CE_BASE", f"{WORK}/e5_ft")
CE = os.environ.get("CE_DIR", f"{XD}/ce")
L = int(os.environ.get("CE_LEN", 48))        # max tokens per side
BS = int(os.environ.get("CE_BS", 512))
LR = float(os.environ.get("CE_LR", 5e-5))
NMAX = int(os.environ.get("CE_N", 0))        # cap on training pairs, 0 = all
EPOCHS = float(os.environ.get("CE_EPOCHS", 1))
INIT = os.environ.get("CE_INIT", BASE)      # weights to start from (an already trained CE to continue it)
SELF = os.environ.get("SELF", "")            # self-training: a test q file (x_nocopy test) whose confident decisions
SELF_C = os.environ.get("SELF_C", "")        # become labels, for these countries (default: countries without train labels)
SELF_N = int(os.environ.get("SELF_N", 0))    # cap on pseudo-labelled pairs, 0 = all
HI, LO = float(os.environ.get("SELF_HI", 0.98)), float(os.environ.get("SELF_LO", 0.02))
TEXT = os.environ.get("CE_TEXT", "norm")    # norm: cached v4 normalisation | raw: transliterated, lowercased, punctuation
TOK = AutoTokenizer.from_pretrained(BASE)   # kept | orig: as given (case, accents, scripts), for a multilingual reranker
CLS, SEP, PAD = TOK.cls_token_id, TOK.sep_token_id, TOK.pad_token_id


def _raw(na):
    from anyascii import anyascii
    return f"{anyascii(na[0]).lower()} | {anyascii(na[1]).lower()}"


def _orig(na):
    return f"{na[0].strip()} | {na[1].strip()}"


def store(split, tag):
    """ragged token ids of every record: (flat int32, offsets int64). tag: s1 | other"""
    f = f"{XD}/tok{'' if TEXT == 'norm' else '_' + TEXT}_{split}_{tag}.npz"
    if not os.path.exists(f):
        if TEXT == "norm":   # cached v4 normalisation (transliterated, lowercased, abbreviations expanded)
            n = normed(split, 1) if tag == "s1" else pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
            t = (n.nn + " | " + n.na).tolist()
        else:                # raw: keeps punctuation / suffix spelling ("L.L.C." vs "LLC") that normalisation drops
            from common import load
            df = load(split, 1) if tag == "s1" else pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
            with Pool(32) as p:
                t = p.map(_orig if TEXT == "orig" else _raw, zip(df.business_name.tolist(), df.business_address.tolist()),
                          chunksize=20000)
        flat, lens = [], []
        for i in range(0, len(t), 500_000):
            ids = TOK(t[i:i + 500_000], add_special_tokens=False, truncation=True, max_length=L)["input_ids"]
            lens.append(np.fromiter(map(len, ids), np.int64, len(ids)))
            flat.append(np.fromiter((x for s in ids for x in s), np.int32, int(lens[-1].sum())))
        lens = np.concatenate(lens)
        np.savez(f, flat=np.concatenate(flat), off=np.r_[0, np.cumsum(lens)])
        print(split, tag, len(t), "tokens mean", lens.mean().round(1), "p99", np.percentile(lens, 99), flush=True)
    z = np.load(f)
    return z["flat"], z["off"]


class Pairs(torch.utils.data.IterableDataset):
    """yields whole padded batches of <s> S1 </s></s> record </s>; a indexes S1, b indexes records."""
    def __init__(self, A, B, a, b, y=None, batches=()):
        self.A, self.B, self.a, self.b, self.y, self.batches = A, B, a, b, y, batches

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        w, nw = (info.id, info.num_workers) if info else (0, 1)
        for bi in range(w, len(self.batches), nw):
            yield self.batch(self.batches[bi])

    def batch(self, idx):
        (fa, oa), (fb, ob) = self.A, self.B
        seqs = [np.concatenate(([CLS], fa[oa[i]:oa[i + 1]], [SEP, SEP], fb[ob[j]:ob[j + 1]], [SEP]))
                for i, j in zip(self.a[idx], self.b[idx])]
        ids = np.full((len(seqs), max(map(len, seqs))), PAD, np.int64)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = s
        out = {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy((ids != PAD).astype(np.int64)),
               "idx": torch.from_numpy(np.asarray(idx))}
        if self.y is not None:
            out["y"] = torch.from_numpy(self.y[idx].astype(np.float32))
        return out


def loader(ds):
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=4,
                                       persistent_workers=False)


def plen(A, B, a, b):
    return (A[1][a + 1] - A[1][a]) + (B[1][b + 1] - B[1][b])


def bucketed(lens, bs, rng, chunk=100):
    """shuffled batches of similar length: sort by length inside random chunks of `chunk` batches."""
    order = rng.permutation(len(lens))
    out = []
    for i in range(0, len(order), chunk * bs):
        o = order[i:i + chunk * bs]
        o = o[np.argsort(lens[o], kind="stable")]
        out += [o[j:j + bs] for j in range(0, len(o), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def logits(model, A, B, a, b, bs=4096):
    """scores in input order; batches sorted by length to cut padding."""
    order = np.argsort(plen(A, B, a, b), kind="stable")
    out = np.empty(len(a), np.float32)
    model.eval()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        for n, bt in enumerate(loader(Pairs(A, B, a, b, batches=batches))):
            z = model(input_ids=bt["input_ids"].cuda(non_blocking=True),
                      attention_mask=bt["attention_mask"].cuda(non_blocking=True)).logits.squeeze(-1)
            out[bt["idx"].numpy()] = z.float().cpu().numpy()
            if n % 2000 == 0:
                print(f"  scored {min((n + 1) * bs, len(a))}/{len(a)}  {(n + 1) * bs / (time.time() - t0):.0f} pairs/s", flush=True)
    return out


def auc(y, sc):
    rk = np.empty(len(sc)); rk[np.argsort(sc, kind="mergesort")] = np.arange(1, len(sc) + 1)
    npos = y.sum()
    return (rk[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def cat(X, Y):
    """one ragged store from two; ids of the second are shifted by the first's length."""
    return np.concatenate((X[0], Y[0])), np.concatenate((X[1], Y[1][1:] + X[1][-1]))


def pseudo(train_countries, rng):
    """(S1, record, label) from test: the top-5 candidates of records whose best candidate is confident. q >= HI:
    that candidate 1, the record's others 0 (a record has at most one S1); q <= LO: all 0."""
    q = pd.read_parquet(SELF)
    ctry = SELF_C.split(",") if SELF_C else sorted(set(q.c) - set(train_countries))
    q = q[q.c.isin(ctry)]
    pos = q[q.q >= HI]
    c = pd.read_parquet(f"{WORK}/cand_test.parquet", columns=["rid", "sid", "rank"])
    c = c[(c["rank"].values < 5) & np.isin(c.rid.values, np.r_[pos.rid.values, q.rid.values[q.q <= LO]])]
    c = pd.concat([c[["rid", "sid"]], pos[["rid", "sid"]]]).drop_duplicates()
    y = c.rid.map(pos.set_index("rid").sid).values == c.sid.values
    sel = np.arange(len(c)) if not SELF_N or SELF_N >= len(c) else rng.choice(len(c), SELF_N, replace=False)
    print(f"pseudo-labels {ctry}: {len(q)} records, {len(pos)} confident matches, {(q.q <= LO).sum()} confident "
          f"no-match; {len(sel)} pairs, pos {y[sel].mean():.3f}", flush=True)
    return c.sid.values[sel], c.rid.values[sel], y[sel]


def train():
    s1, other, ts, s1f, rf = truth_arrays()
    A, B = store("train", "s1"), store("train", "other")
    # training pairs: the retrieved top-5 of S1 folds 0-3, not the matcher's shortlist, so the CE keeps seeing
    # wrong-S1 candidates as negatives whatever the shortlist keeps
    c = pd.read_parquet(f"{WORK}/cand_train.parquet", columns=["rid", "sid", "rank"])
    c = c[(c["rank"].values < 5) & (rf[c.rid.values] < 4)]
    cr, cs = c.rid.values, c.sid.values
    sel = np.arange(len(cr))
    rng = np.random.default_rng(0)
    if NMAX and NMAX < len(sel):
        sel = rng.choice(sel, NMAX, replace=False)
    a, b, y = cs[sel], cr[sel], ts[cr[sel]] == cs[sel]
    if SELF:
        pa, pb, py = pseudo(s1.country.unique(), rng)
        a, b, y = np.r_[a, pa + len(A[1]) - 1], np.r_[b, pb + len(B[1]) - 1], np.r_[y, py]
        A, B = cat(A, store("test", "s1")), cat(B, store("test", "other"))
    lens = plen(A, B, a, b)
    batches = [bt for _ in range(math.ceil(EPOCHS)) for bt in bucketed(lens, BS, rng)]
    batches = batches[:int(len(batches) * EPOCHS / math.ceil(EPOCHS))]
    steps = len(batches)
    print(f"train pairs {len(a)}  pos {y.mean():.3f}  steps {steps}  lr {LR}  init {INIT}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(INIT, num_labels=1, attn_implementation="sdpa").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps), steps)
    model.train(); t0 = time.time(); run = 0.0
    for step, bt in enumerate(loader(Pairs(A, B, a, b, y, batches=batches))):
        with torch.autocast("cuda", torch.bfloat16):
            z = model(input_ids=bt["input_ids"].cuda(non_blocking=True),
                      attention_mask=bt["attention_mask"].cuda(non_blocking=True)).logits.squeeze(-1)
        loss = Fn.binary_cross_entropy_with_logits(z.float(), bt["y"].cuda(non_blocking=True))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0:
            print(f"step {step}/{steps}  loss {run:.4f}  {(step + 1) * BS / (time.time() - t0):.0f} pairs/s", flush=True)
    model.save_pretrained(CE); TOK.save_pretrained(CE)
    if SELF:   # the fold-9 check below reads the v4 feats2 / oof caches, which a rebuild may be replacing
        return
    # quick check on fold 9 (never seen by the CE or the embedder): pair AUC and argmax accuracy vs stage-1 OOF
    k = pd.read_parquet(f"{WORK}/feats2_train.parquet", columns=["rid", "sid"])
    r, s = k.rid.values, k.sid.values
    p1 = pd.read_parquet(f"{WORK}/oof_train.parquet", columns=["p"]).p.values
    recs = np.unique(r[rf[r] == 9])
    v = np.flatnonzero(np.isin(r, rng.choice(recs, min(len(recs), 600_000), replace=False)))  # whole candidate lists
    ce = logits(model, A, B, s[v], r[v])
    yv = ts[r[v]] == s[v]
    print(f"fold-9 pair AUC  CE {auc(yv, ce):.5f}  stage-1 {auc(yv, p1[v]):.5f}", flush=True)
    d = pd.DataFrame({"rid": r[v], "ok": yv, "ce": ce, "p1": p1[v]})
    for col in ("ce", "p1"):
        best = d.loc[d.groupby("rid")[col].idxmax()]
        own = ts[best.rid.values] >= 0
        print(f"  argmax by {col}: right entity for {best.ok.values[own].mean():.5f} of real records", flush=True)


def score(split):
    A, B = store(split, "s1"), store(split, "other")
    k = pd.read_parquet(f"{WORK}/feats2_{split}.parquet", columns=["rid", "sid"])
    r, s = k.rid.values, k.sid.values
    sel = np.arange(len(k))
    if split == "train":
        rf = truth_arrays()[4]
        sel = np.flatnonzero(rf[r] >= 4)
    topk = int(os.environ.get("TOPK", 0))
    if topk:   # rank of each pair within its record by v4 stage-1 probability
        p1 = pd.read_parquet(f"{WORK}/oof_train.parquet" if split == "train" else f"{WORK}/p1_test.parquet", columns=["p"]).p.values
        o = np.lexsort((-p1, r)); rk = np.empty(len(r), np.int64)
        first = np.r_[True, r[o][1:] != r[o][:-1]]; st = np.flatnonzero(first)
        rk[o] = np.arange(len(r)) - st[np.cumsum(first) - 1]
        sel = sel[rk[sel] < topk]
    model = AutoModelForSequenceClassification.from_pretrained(CE, num_labels=1, attn_implementation="sdpa").cuda()
    out = np.full(len(k), np.nan, np.float32)
    out[sel] = logits(model, A, B, s[sel], r[sel])
    np.save(f"{XD}/ce{os.environ.get('CE_TAG', '')}_{split}.npy", out)
    print("saved", split, np.isfinite(out).sum(), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        for sp in ("train", "test"):
            store(sp, "s1"); store(sp, "other")
    elif cmd == "train":
        train()
    else:
        score(sys.argv[2])
