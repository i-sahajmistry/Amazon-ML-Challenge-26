"""GPU nearest-neighbour retrieval with char-3-gram TF-IDF (exact cosine), per split and country.

For each (split, country):
  * sample Q_N Source-1 queries; retrieve the top-K S2+S3 records for three text views:
    name, addr, both (name + address). All text is transliterated to ASCII first.
  * reverse search for the 'both' view: for records that appear in some query's top-20, plus a
    random sample of records, find their best Source-1 entity among ALL S1 of that country.
  * train only: cosine of every true pair of the sampled queries (even when outside top-K).

Outputs (OUT_DIR/retrieval/):
  topk_{split}_{country}.parquet     s1, view, rank, rid, score
  rev_{split}_{country}.parquet      rid, best_s1, best, second, in_topk_pool, is_matched (train)
  truescore_{split}_{country}.parquet s1, id, view, score (train)
"""
import os
import time
from functools import lru_cache
from itertools import chain

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from multiprocessing import get_context
from sklearn.feature_extraction.text import TfidfVectorizer

from common import OUT_DIR, WORKERS, load, log, records, translit
from common import train_pairs as _train_pairs

Q_N = int(os.environ.get("Q_N", 50000))
K = int(os.environ.get("K", 100))
REV_SAMPLE = int(os.environ.get("REV_SAMPLE", 200000))
BATCH = int(os.environ.get("BATCH", 512))
REV_BATCH = int(os.environ.get("REV_BATCH", 2048))
OUT = os.path.join(OUT_DIR, "retrieval")
DEV = torch.device("cuda")
VIEWS = ["name", "addr", "both"]

_VEC = None  # set before forking the transform pool


@lru_cache(maxsize=1)
def train_pairs():
    return _train_pairs()[["s1", "id"]]


def _translit_many(xs):
    return [translit(x) for x in xs]


def _transform(xs):
    return _VEC.transform(xs)


def par(fn, items, workers=WORKERS):
    parts = np.array_split(np.arange(len(items)), workers * 4)
    chunks = [[items[i] for i in p] for p in parts if len(p)]
    with get_context("fork").Pool(workers) as pool:
        return pool.map(fn, chunks)


def texts(df):
    n = list(chain.from_iterable(par(_translit_many, df.business_name.tolist())))
    a = list(chain.from_iterable(par(_translit_many, df.business_address.tolist())))
    return {"name": n, "addr": a, "both": [x + " | " + y for x, y in zip(n, a)]}


def vectorize(fit_texts, *groups):
    global _VEC
    rng = np.random.default_rng(0)
    idx = rng.choice(len(fit_texts), size=min(len(fit_texts), 1_000_000), replace=False)
    _VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=3, sublinear_tf=True,
                           dtype=np.float32)
    _VEC.fit([fit_texts[i] for i in idx])
    return [sp.vstack(par(_transform, g)).tocsr() for g in groups]


def to_gpu_csr(m):
    m = m.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
        torch.from_numpy(m.data), size=m.shape, device=DEV)


def topk_search(index_csr, queries, k, batch=BATCH):
    """Exact cosine top-k of each query row against index rows. Returns (idx, score) numpy arrays."""
    ind = to_gpu_csr(index_csr)
    k = min(k, index_csr.shape[0])
    out_i = np.empty((queries.shape[0], k), dtype=np.int64)
    out_s = np.empty((queries.shape[0], k), dtype=np.float32)
    for s in range(0, queries.shape[0], batch):
        qb = torch.from_numpy(queries[s:s + batch].toarray()).to(DEV)
        scores = torch.sparse.mm(ind, qb.T.contiguous())          # (N_index, B)
        v, i = torch.topk(scores, k, dim=0)
        out_i[s:s + batch] = i.T.cpu().numpy()
        out_s[s:s + batch] = v.T.cpu().numpy()
        del scores, v, i, qb
    del ind
    torch.cuda.empty_cache()
    return out_i, out_s


def rowwise_cos(a, b):
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def run(split, country):
    t0 = time.time()
    s1 = load(f"{split}_s1")
    s1 = s1[s1.country == country].reset_index(drop=True)
    rec = records(split)
    rec = rec[rec.country == country].reset_index(drop=True)
    q = s1.sample(min(Q_N, len(s1)), random_state=42).reset_index(drop=True)
    log(split, country, "S1", len(s1), "records", len(rec), "queries", len(q))

    t_rec, t_s1 = texts(rec), texts(s1)
    s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    qpos = s1_pos.loc[q.entity_id].values
    log("texts ready", round(time.time() - t0))

    rows, true_rows, rev = [], [], None
    for view in VIEWS:
        R, S = vectorize(t_rec[view] + t_s1[view], t_rec[view], t_s1[view])
        Q = S[qpos]
        log(view, "vectorized: vocab", R.shape[1], "nnz/rec", round(R.nnz / R.shape[0], 1), round(time.time() - t0))
        idx, sc = topk_search(R, Q, K)
        log(view, "topk done", round(time.time() - t0))
        rows.append(pd.DataFrame({
            "s1": np.repeat(q.entity_id.values, idx.shape[1]), "view": view,
            "rank": np.tile(np.arange(1, idx.shape[1] + 1), len(q)),
            "rid": rec.entity_id.values[idx.ravel()], "score": sc.ravel()}))

        if split == "train":
            tp = train_pairs()[["s1", "id"]]
            tp = tp[tp.s1.isin(set(q.entity_id))]
            rpos = pd.Series(np.arange(len(rec)), index=rec.entity_id)
            cos = rowwise_cos(S[s1_pos.loc[tp.s1].values], R[rpos.loc[tp.id].values])
            true_rows.append(tp.assign(view=view, score=cos))

        if view == "both":
            # reverse search: records -> best S1 over ALL S1 of this country
            # candidate pool only matters for the mutual-best analysis on train
            pool = set(rows[-1].loc[rows[-1]["rank"] <= 20, "rid"]) if split == "train" else set()
            rng = np.random.default_rng(1)
            extra = rec.entity_id.values[rng.choice(len(rec), size=min(REV_SAMPLE, len(rec)), replace=False)]
            extra_set = set(extra)
            rids = np.array(sorted(pool | extra_set))
            rpos = pd.Series(np.arange(len(rec)), index=rec.entity_id).loc[rids].values
            ri, rs = topk_search(S, R[rpos], 2, batch=REV_BATCH)
            rev = pd.DataFrame({"rid": rids, "best_s1": s1.entity_id.values[ri[:, 0]],
                                "best": rs[:, 0], "second": rs[:, 1],
                                # set lookups: np.isin on string arrays is quadratic
                                "in_topk_pool": [r in pool for r in rids],
                                "in_random_sample": [r in extra_set for r in rids]})
            log("reverse search done", len(rids), round(time.time() - t0))

    os.makedirs(OUT, exist_ok=True)
    tag = f"{split}_{country}"
    pd.concat(rows).to_parquet(f"{OUT}/topk_{tag}.parquet")
    if split == "train":
        matched = set(train_pairs().id)
        rev["is_matched"] = rev.rid.isin(matched)
        pd.concat(true_rows).to_parquet(f"{OUT}/truescore_{tag}.parquet")
    rev.to_parquet(f"{OUT}/rev_{tag}.parquet")
    log(tag, "saved", round(time.time() - t0))


def main():
    log("GPU", torch.cuda.get_device_name(0), "workers", WORKERS)
    todo = os.environ.get("ONLY", "train:US,train:India,test:US,test:India,test:France").split(",")
    for item in todo:
        split, country = item.split(":")
        run(split, country)


if __name__ == "__main__":
    main()
