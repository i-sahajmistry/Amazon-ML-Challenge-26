"""LLM judge: a local open-weight LLM (Apache-2.0, <= 8B parameters, run offline, not fine-tuned) reads a record and a
candidate S1 and returns the log-odds that they describe the same business. Its general language knowledge ("SCI
Dupont" is a property company, not "Dupont SARL"; "Holding", "Groupe" name another company) does not depend on which
countries have training labels, which is what an unseen country needs.
Only records the pipeline is unsure about are judged: the best stage-1 probability inside a band, or a close runner-up
S1. For each, its best candidates by stage-1 probability are judged (top 2, a third if it is also likely). The score
is a stage-2 feature (x_stage_multi.py LLM=1), so stage 2 learns from the training countries how much to trust it.
  python llm_judge.py select train|test    -> x/llm_pairs_{split}.parquet   (rid, sid)
  python llm_judge.py score train|test     -> x/llm_{split}.part{SHARD}.parquet   (SHARD=k NSHARD=n, one process per GPU)
  python llm_judge.py merge train|test     -> x/llm_{split}.parquet   (rid, sid, llm); on train also the judge's AUC
Env: LLM_MODEL (local model folder), SEL (v4: work/oof_train + p1_test | v8: x/oof5_train{TAG} + p5_test{TAG}),
     LLM_P1 (band of the best stage-1 probability, default 0.2,0.98), LLM_P2 (runner-up probability that also marks a
     record ambiguous, default 0.2), LLM_MAX (cap on judged records per split, most uncertain first),
     LLM_TOKENS (tokens per batch), LLM_SMOKE=N (score only the first N pairs and print them: a quick check)"""
import os
import sys
import time

import numpy as np
import pandas as pd

from common import WORK, load

XD = f"{WORK}/x"
os.makedirs(XD, exist_ok=True)
TAG = os.environ.get("TAG", "")
SEL = os.environ.get("SEL", "v4")
P1 = tuple(float(x) for x in os.environ.get("LLM_P1", "0.2,0.98").split(","))
P2 = float(os.environ.get("LLM_P2", 0.2))
MAXR = int(os.environ.get("LLM_MAX", 600000))
THR = 0.70
PROMPT = ("Do these two records describe the same business, i.e. the same legal entity at the same place? Records of "
          "one business can differ by typos, abbreviations, transliteration, word order, a missing or abbreviated legal "
          "form, or a partial address. Businesses that only share a name, such as a holding or property company, a "
          "franchise or another branch at a different address, are different.\n"
          "A: {n1} | {a1}\nB: {n2} | {a2}\nCountry: {c}\nAnswer Yes or No.")


def stage1(split):
    if SEL == "v8":
        f = f"{XD}/oof5_train{TAG}.parquet" if split == "train" else f"{XD}/p5_test{TAG}.parquet"
    else:
        f = f"{WORK}/oof_train.parquet" if split == "train" else f"{WORK}/p1_test.parquet"
    return pd.read_parquet(f, columns=["rid", "sid", "p"])


def others(split):
    return pd.concat([load(split, 2), load(split, 3)], ignore_index=True)


def select(split):
    k = stage1(split)
    if split == "train":   # records stage 2 trains and validates on (and a held-out country's, fold 10)
        from harness import truth_arrays
        rf = truth_arrays()[4]
        k = k[rf[k.rid.values] >= 4]
    k = k.sort_values(["rid", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = k.groupby("rid").cumcount().values
    first = g == 0
    rid1, p1 = k.rid.values[first], k.p.values[first]
    p2 = pd.Series(k.p.values[g == 1], index=k.rid.values[g == 1]).reindex(rid1).fillna(0.0).values
    unsure, ambig = (p1 >= P1[0]) & (p1 < P1[1]), p2 >= P2
    if split == "train" and os.environ.get("LLM_HELD_FOLDS"):
        # held-out country (leave-one-country-out): judge only records whose best S1 is in these crc folds, the
        # evaluation group loco_eval.py reports separately, to save GPU time
        import zlib
        from common import HELD
        folds = [int(x) for x in os.environ["LLM_HELD_FOLDS"].replace(",", ":").split(":") if x]   # "8:9"
        held = rf[rid1] == HELD
        s1ids = load("train", 1).entity_id.values
        sid1 = k.sid.values[first]
        crc = np.full(len(rid1), -1)
        crc[held] = [zlib.crc32(s1ids[s].encode()) % 10 for s in sid1[held]]
        drop = held & ~np.isin(crc, folds)
        unsure, ambig = unsure & ~drop, ambig & ~drop
        print(f"held-out records outside crc folds {folds} not judged: {drop.sum():,}", flush=True)
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    prio = -np.abs(lg(p1) - lg(THR)) + np.where(ambig, 2.0, 0.0)   # nearest the decision first, ambiguous records first
    sel = np.flatnonzero(unsure | ambig)
    capped = len(sel) > MAXR
    if capped:
        sel = sel[np.argsort(-prio[sel], kind="stable")[:MAXR]]
    keep = np.zeros(int(k.rid.max()) + 1, bool)
    keep[rid1[sel]] = True
    pairs = k[keep[k.rid.values] & ((g < 2) | ((g == 2) & (k.p.values >= 0.2)))][["rid", "sid"]].reset_index(drop=True)
    pairs.to_parquet(f"{XD}/llm_pairs_{split}.parquet")
    c = others(split).country.values[rid1]
    print(pd.DataFrame({"records": pd.Series(c).value_counts(), "unsure p1": pd.Series(c[unsure]).value_counts(),
                        "ambiguous p2": pd.Series(c[ambig]).value_counts(),
                        "judged": pd.Series(c[sel]).value_counts()}).fillna(0).astype(int).to_string())
    print(f"{split}: judged {len(sel):,} of {len(rid1):,} records ({len(sel) / len(rid1):.2%}), {len(pairs):,} pairs"
          f"{' (capped at LLM_MAX, most uncertain kept)' if capped else ''}; band p1 {P1}, runner-up p2 >= {P2}",
          flush=True)


def _clean(x):
    return " ".join(str(x).split())[:160] or "(empty)"


def _auc(score, y):
    """probability that a random true pair scores above a random other pair (ties count half)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((rankdata(score)[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def score(split):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    shard, nshard = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    smoke = int(os.environ.get("LLM_SMOKE", 0))
    out_f = f"{XD}/llm_{split}.part{shard}.parquet"
    if os.path.exists(out_f) and not smoke:
        print("exists, skip", out_f, flush=True)
        return
    pairs = pd.read_parquet(f"{XD}/llm_pairs_{split}.parquet").iloc[shard::nshard].reset_index(drop=True)
    if smoke:   # a random sample, so every country and both sources appear
        pairs = pairs.sample(n=min(smoke, len(pairs)), random_state=0).reset_index(drop=True)
    s1, o = load(split, 1), others(split)
    n1, a1, c1 = s1.business_name.values, s1.business_address.values, s1.country.values
    n2, a2 = o.business_name.values, o.business_address.values
    mp = os.environ["LLM_MODEL"]
    tok = AutoTokenizer.from_pretrained(mp)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, (yes, no)
    t0 = time.time()
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(n1=_clean(n1[s]), a1=_clean(a1[s]), n2=_clean(n2[r]),
                                                   a2=_clean(a2[r]), c=c1[s])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for s, r in zip(pairs.sid.values, pairs.rid.values)]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    lens = np.fromiter(map(len, ids), np.int64, len(ids))
    order = np.argsort(lens, kind="stable")
    print(f"shard {shard}/{nshard}: {len(pairs):,} pairs, tokens mean {lens.mean():.0f} max {lens.max()}, "
          f"prepared in {time.time() - t0:.0f}s", flush=True)
    model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    budget = int(os.environ.get("LLM_TOKENS", 24000))
    out = np.full(len(pairs), np.nan, np.float32)
    i, nb, t0 = 0, 0, time.time()
    while i < len(order):
        j = i + 1
        while j < len(order) and (j + 1 - i) * lens[order[j]] <= budget:
            j += 1
        idx = order[i:j]
        ml = int(lens[idx].max())
        inp = torch.full((len(idx), ml), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), ml), dtype=torch.long)
        for r_, k_ in enumerate(idx):
            x = ids[k_]
            inp[r_, ml - len(x):] = torch.tensor(x)
            att[r_, ml - len(x):] = 1
        pos = (att.cumsum(1) - 1).clamp(min=0)
        with torch.inference_mode():
            lo = model(input_ids=inp.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda(), use_cache=False,
                       logits_to_keep=1).logits[:, -1, :]
        out[idx] = (lo[:, yes[0]] - lo[:, no[0]]).float().cpu().numpy()
        i, nb = j, nb + 1
        if nb % 200 == 0:
            el = time.time() - t0
            print(f"  {i:,}/{len(order):,} pairs, {i / el:.0f} pairs/s, eta {(len(order) - i) / (i / el) / 60:.0f} min",
                  flush=True)
    print(f"shard {shard}: {len(order) / (time.time() - t0):.0f} pairs/s", flush=True)
    if smoke:
        for t, v in list(zip(texts, out))[:8]:
            print(f"--- log-odds(Yes) {v:+.2f}\n{t[-420:]}", flush=True)
        if split == "train":   # gate: on labelled pairs the judge must separate true pairs from the rest
            from harness import truth_arrays
            y = truth_arrays()[2][pairs.rid.values] == pairs.sid.values
            auc = _auc(out, y)
            gate = float(os.environ.get("LLM_GATE_AUC", 0.6))
            print(f"smoke check: judge AUC {auc:.4f} on {len(y):,} labelled pairs ({y.mean():.1%} true); gate {gate}",
                  flush=True)
            if not auc >= gate:
                sys.exit(3)
        return
    pairs.assign(llm=out).to_parquet(out_f)
    print("wrote", out_f, flush=True)


def merge(split):
    parts = sorted(f for f in os.listdir(XD) if f.startswith(f"llm_{split}.part"))
    L = pd.concat([pd.read_parquet(f"{XD}/{f}") for f in parts], ignore_index=True)
    n = len(pd.read_parquet(f"{XD}/llm_pairs_{split}.parquet"))
    assert len(L) == n and L.llm.notna().all(), (len(L), n, L.llm.isna().sum())
    L.to_parquet(f"{XD}/llm_{split}.parquet")
    print(f"merged {len(parts)} shards: {len(L):,} pairs -> {XD}/llm_{split}.parquet", flush=True)
    if split == "train":   # how well does the judge separate true pairs from the rest, per country (labels: train only)
        from harness import truth_arrays
        ts = truth_arrays()[2]
        y = ts[L.rid.values] == L.sid.values
        c = others("train").country.values[L.rid.values]
        for g in sorted(set(c)):
            m = c == g
            print(f"  judge AUC on {g} judged pairs: {_auc(L.llm.values[m], y[m]):.4f}  ({m.sum():,} pairs, "
                  f"{y[m].mean():.2%} true)", flush=True)


if __name__ == "__main__":
    {"select": select, "score": score, "merge": merge}[sys.argv[1]](sys.argv[2])
