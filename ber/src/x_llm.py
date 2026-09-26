"""LLM judge for the unsure records: Qwen3-Reranker-4B (Apache-2.0, 4.0B; trained to answer yes / no to "does the
Document meet the Query"), LoRA-fine-tuned on raw S1 / record text of each record's best candidate. Score = logit(yes)
- logit(no). A multilingual LLM knows what "& Fils", "Holding" or SARL vs SAS mean in any language, which models
trained on US / India text cannot.
  python x_llm.py train      # LoRA on train records of S1 folds 4-7 whose stage-1 p is unsure (+ some sure ones)
  python x_llm.py val        # score validation rows (x/val_q{QT}w, folds 8-9) in the unsure band; stacking check
  python x_llm.py test       # score test rows (x/test_q{QT}w) in the unsure band -> x/llm_test{QT}.parquet
  QB=_v10p python x_llm.py extend   # unsure rows of stage-2 model QB: reuse QT's scores of the same pairs, score the
                                    # rest -> x/llm_{val,test}{QB}{LLM_TAG}.parquet
  SELF=x/test_q_v10fr3l.parquet LLM_INIT=x/llm_lora LLM_DIR=x/llm_lora_fr LLM_TAG=_fr python x_llm.py train
                             # continue the LoRA on confident test decisions of the countries without train labels
env QT (stage-2 tag, default _v9p), LLM (base model dir), LLM_N (training pairs), BAND (lo,hi), LLM_DIR (adapter),
LLM_TAG (output suffix), SELF_N (pseudo-labelled pairs), SELF_C (countries; default: those without train labels)"""
import os, sys, math, time, numpy as np, pandas as pd, torch
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import WORK, load, countries
from harness import truth_arrays, wscore

XD = f"{WORK}/x"
QT = os.environ.get("QT", "_v9p")
BASE = os.environ.get("LLM", "/scratch/scai/mtech/aib262144/models/qwen3-reranker-4b")
OUTD = os.environ.get("LLM_DIR", f"{XD}/llm_lora")
LT = os.environ.get("LLM_TAG", "")
INIT = os.environ.get("LLM_INIT", "")        # an adapter to continue instead of a fresh LoRA
SELF = os.environ.get("SELF", "")            # self-training: test q file whose confident decisions become labels
SELF_N = int(os.environ.get("SELF_N", 80_000))
N = int(os.environ.get("LLM_N", 120_000))
LO, HI = map(float, os.environ.get("BAND", "0.01,0.99").split(","))
BS, LR, MAXLEN = int(os.environ.get("LLM_BS", 32)), float(os.environ.get("LLM_LR", 1e-4)), 160
PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
       "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Is the "
       "Document a record of the same business as the Query (possibly written differently), not a different business "
       "that looks similar?\n")
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOK = AutoTokenizer.from_pretrained(BASE, padding_side="left")
YES, NO = TOK.convert_tokens_to_ids("yes"), TOK.convert_tokens_to_ids("no")


def texts(split):
    s1 = load(split, 1); o = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)
    f = lambda d: (d.business_name.str.strip() + " | " + d.business_address.str.strip()).values
    return f(s1), f(o)


def prompts(T1, T2, sid, rid):
    return [f"{PRE}<Query>: {T1[s]}\n<Document>: {T2[r]}{SUF}" for s, r in zip(sid, rid)]


def model(lora=True):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if lora == "load":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, OUTD).merge_and_unload()
    elif lora and INIT:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, INIT, is_trainable=True)
    elif lora:
        from peft import LoraConfig, get_peft_model
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return m


def margin(m, P):
    b = TOK(P, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
    z = m(**b, logits_to_keep=1).logits[:, -1, :]
    return (z[:, YES] - z[:, NO]).float()


@torch.no_grad()
def score(m, P, bs=128):
    m.eval()
    order = np.argsort([len(p) for p in P])
    out = np.empty(len(P), np.float32); t0 = time.time()
    for i in range(0, len(P), bs):
        idx = order[i:i + bs]
        with torch.autocast("cuda", torch.bfloat16):
            out[idx] = margin(m, [P[j] for j in idx]).cpu().numpy()
        if i % (bs * 200) == 0:
            print(f"  scored {i + len(idx)}/{len(P)}  {(i + len(idx)) / (time.time() - t0):.0f} pairs/s", flush=True)
    return out


def train():
    s1, other, ts, s1f, rf = truth_arrays()
    k = pd.read_parquet(f"{XD}/oof5_train{QT[:-1] if QT.endswith('p') else QT}.parquet", columns=["rid", "sid", "p"])
    b = k.sort_values(["rid", "p"], ascending=[True, False]).drop_duplicates("rid")      # each record's best S1
    b = b[np.isin(rf[b.rid.values], [4, 5, 6, 7]) & np.isin(s1f[b.sid.values], [4, 5, 6, 7])]
    rng = np.random.default_rng(0)
    band = (b.p.values > LO) & (b.p.values < HI)
    pick = np.r_[rng.permutation(np.flatnonzero(band))[:int(N * 0.8)], rng.permutation(np.flatnonzero(~band))[:int(N * 0.2)]]
    b = b.iloc[rng.permutation(pick)]
    y = (ts[b.rid.values] == b.sid.values).astype(np.float32)
    T1, T2 = texts("train")
    P = prompts(T1, T2, b.sid.values, b.rid.values)
    print(f"train pairs {len(P)} (unsure band {band.sum()} records), positive {y.mean():.3f}", flush=True)
    if SELF:   # each record's best pair where the decision is confident: q >= 0.98 -> 1, q <= 0.02 -> 0
        q = pd.read_parquet(SELF)
        ctry = os.environ["SELF_C"].split(",") if os.environ.get("SELF_C") else sorted(set(q.c) - set(s1.country))
        q = q[q.c.isin(ctry) & ((q.q.values >= 0.98) | (q.q.values <= 0.02))]
        pos, neg = q[q.q.values >= 0.98], q[q.q.values <= 0.02]
        q = pd.concat([pos.sample(min(len(pos), SELF_N // 2), random_state=0), neg.sample(min(len(neg), SELF_N // 2), random_state=0)])
        U1, U2 = texts("test")
        P = P + prompts(U1, U2, q.sid.values, q.rid.values)
        y = np.r_[y, (q.q.values >= 0.98).astype(np.float32)]
        o = rng.permutation(len(P)); P = [P[i] for i in o]; y = y[o]
        print(f"+ pseudo-labelled {ctry}: {len(pos)} confident matches, {len(neg)} confident non-matches; "
              f"{len(q)} pairs used; total {len(P)}", flush=True)
    m = model(True); m.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    steps = math.ceil(len(P) / BS); warm = int(0.03 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(warm, 1)) * max(0.0, 1 - s / steps))
    m.train(); t0 = time.time(); run = None
    for st in range(steps):
        sl = slice(st * BS, (st + 1) * BS)
        with torch.autocast("cuda", torch.bfloat16):
            z = margin(m, P[sl])
        loss = Fn.binary_cross_entropy_with_logits(z, torch.from_numpy(y[sl]).cuda())
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if st % 200 == 0:
            print(f"step {st}/{steps} loss {run:.4f} {(st + 1) * BS / (time.time() - t0):.0f} pairs/s", flush=True)
    m.save_pretrained(OUTD); print("saved", OUTD, flush=True)


def auc(y, s):
    r = np.empty(len(s)); r[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    n1 = y.sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def val():
    s1, other, ts, s1f, rf = truth_arrays()
    v = pd.read_parquet(f"{XD}/val_q{QT}w.parquet")
    band = (v.q.values > LO) & (v.q.values < HI)
    T1, T2 = texts("train")
    m = model("load")
    llm = np.full(len(v), np.nan, np.float32)
    llm[band] = score(m, prompts(T1, T2, v.sid.values[band], v.rid.values[band]))
    v.assign(llm=llm).to_parquet(f"{XD}/llm_val{QT}{LT}.parquet")
    y = v.y.values.astype(bool)
    print(f"val rows {len(v)}, unsure band {band.sum()}: AUC stage-2 q {auc(y[band], v.q.values[band]):.4f}, "
          f"LLM {auc(y[band], llm[band]):.4f}", flush=True)
    stack_check(v.assign(llm=llm), ts, s1f)


def fit_stack(d):
    """logistic regression of y on [logit q, llm] over unsure rows (weighted like the validation)."""
    from sklearn.linear_model import LogisticRegression
    x = np.c_[np.log(np.clip(d.q.values, 1e-6, 1 - 1e-6) / np.clip(1 - d.q.values, 1e-6, 1)), d.llm.values]
    return LogisticRegression(C=1.0).fit(x, d.y.values, sample_weight=d.wt.values)


def apply_stack(lr, d):
    x = np.c_[np.log(np.clip(d.q.values, 1e-6, 1 - 1e-6) / np.clip(1 - d.q.values, 1e-6, 1)), d.llm.values]
    return lr.predict_proba(x)[:, 1]


def stack_check(v, ts, s1f):
    T = np.bincount(ts[ts >= 0], minlength=len(s1f)); ents = np.where(s1f >= 8)[0]
    band = np.isfinite(v.llm.values)
    half = (v.sid.values % 2).astype(bool)                      # cross-fit over two halves of the S1s
    q2 = v.q.values.copy()
    for h in (False, True):
        lr = fit_stack(v[band & (half != h)])
        m = band & (half == h)
        q2[m] = apply_stack(lr, v[m])
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8):
        a = wscore(v.sid.values, v.q.values >= thr, v.y.values, v.wt.values, T, ents)
        b = wscore(v.sid.values, q2 >= thr, v.y.values, v.wt.values, T, ents)
        print(f"thr {thr:.2f}  stage-2 {a:.5f}  + LLM {b:.5f}  ({b - a:+.5f})", flush=True)


def test():
    d = pd.read_parquet(f"{XD}/test_q{QT}w.parquet")
    band = (d.q.values > LO) & (d.q.values < HI)
    if os.environ.get("COUNTRY"):                  # e.g. COUNTRY=unlabelled: those countries' unsure rows only
        band &= d.c.isin(countries(os.environ["COUNTRY"])).values
    T1, T2 = texts("test")
    m = model("load")
    llm = np.full(len(d), np.nan, np.float32)
    llm[band] = score(m, prompts(T1, T2, d.sid.values[band], d.rid.values[band]))
    d.assign(llm=llm).to_parquet(f"{XD}/llm_test{QT}{LT}.parquet")
    print("wrote", f"{XD}/llm_test{QT}{LT}.parquet", pd.Series(d.c.values[band]).value_counts().to_dict(), flush=True)


def extend():
    """unsure rows of stage-2 model QB with an LLM score: reuse QT's score of the same (record, S1) pair, score the
    rest (COUNTRY= limits the test rows scored)."""
    QB = os.environ["QB"]
    m = None
    for split, base, src in (("val", f"val_q{QB}w", f"llm_val{QT}{LT}"), ("test", f"test_q{QB}w", f"llm_test{QT}{LT}")):
        d = pd.read_parquet(f"{XD}/{base}.parquet")
        s = pd.read_parquet(f"{XD}/{src}.parquet")
        s = s[np.isfinite(s.llm.values)][["rid", "sid", "llm"]]
        d = d.merge(s, on=["rid", "sid"], how="left")
        band = (d.q.values > LO) & (d.q.values < HI)
        if split == "test" and os.environ.get("COUNTRY"):
            band &= d.c.isin(countries(os.environ["COUNTRY"])).values
        need = band & ~np.isfinite(d.llm.values)
        print(f"{split}: unsure rows {band.sum()}, already scored {(band & np.isfinite(d.llm.values)).sum()}, "
              f"to score {need.sum()}", flush=True)
        if need.any():
            m = m or model("load")
            T1, T2 = texts("train" if split == "val" else "test")
            d.loc[need, "llm"] = score(m, prompts(T1, T2, d.sid.values[need], d.rid.values[need]))
        d.loc[~band, "llm"] = np.nan
        d.to_parquet(f"{XD}/llm_{split}{QB}{LT}.parquet")
        print("wrote", f"{XD}/llm_{split}{QB}{LT}.parquet", flush=True)


if __name__ == "__main__":
    {"train": train, "val": val, "test": test, "extend": extend}[sys.argv[1]]()
