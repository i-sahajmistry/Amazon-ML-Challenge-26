"""Candidate generation: embed every record with the fine-tuned e5, then for each S2/S3 record take the top-K most
similar S1 records with the same country label. Search is a FAISS HNSW graph index per country (ANN=hnsw, default):
each query walks a small-world graph over the S1 embeddings, so its cost grows roughly with log(#S1) instead of #S1,
and the index can be sharded / product-quantised for billions of records. On train India (x_ann.py) M=32 and
efSearch=512 find the true S1 in the top-20 for 99.49% of records vs 99.54% for exact search (IVF with 256 of 939
lists probed: 98.10%). ANN=exact is the brute-force GPU matmul it replaced, ANN=ivf an inverted-file index.
match.py then shortlists these top-K (SHORTLIST).
usage: python retrieve.py train|test"""
import sys, os, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from common import load, embed_text, s1_fold, WORK

split = sys.argv[1]
K = int(os.environ.get("K", 20))
ANN, NPROBE, EF = os.environ.get("ANN", "hnsw"), int(os.environ.get("NPROBE", 32)), int(os.environ.get("HNSW_EF", 512))
s1 = load(split, 1)
other = pd.concat([load(split, 2), load(split, 3)], ignore_index=True)


def emb(df, tag):
    f = f"{WORK}/emb_{split}_{tag}.npy"
    if os.path.exists(f):
        return np.load(f, mmap_mode="r")
    model = SentenceTransformer(f"{WORK}/e5_ft", device="cuda").half()
    model.max_seq_length = 64
    e = model.encode(embed_text(df), batch_size=4096, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float16)
    np.save(f, e)
    print("embedded", tag, e.shape, flush=True)
    return e


E1, E2 = emb(s1, "s1"), emb(other, "other")
out = []
# country is treated as an open set of labels: block on exact label equality, whatever the labels are
for c in sorted(set(other.country)):
    qi = np.where(other.country.values == c)[0]
    di = np.where(s1.country.values == c)[0]
    if len(di) == 0:
        continue
    k = min(K, len(di))
    if ANN in ("hnsw", "ivf"):
        import faiss
        faiss.omp_set_num_threads(32)
        X = np.ascontiguousarray(E1[di], dtype=np.float32)
        if ANN == "hnsw":
            index = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(X)
            index.hnsw.efSearch = max(EF, K)
        else:
            nlist = max(1, int(4 * np.sqrt(len(di))))
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(X.shape[1]), X.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(X[np.random.default_rng(0).permutation(len(X))[:256 * nlist]])
            index.add(X)
            index.nprobe = min(NPROBE, nlist)
        for st in range(0, len(qi), 65536):
            q = qi[st:st + 65536]
            s_, j = index.search(np.ascontiguousarray(E2[q], dtype=np.float32), k)
            ok = j >= 0                                            # fewer than k S1s in the probed lists
            out.append(pd.DataFrame({"rid": np.repeat(q, k).astype(np.int32)[ok.ravel()],
                                     "sid": di[j[ok]].astype(np.int32), "score": s_[ok].astype(np.float32),
                                     "rank": np.tile(np.arange(k, dtype=np.int8), len(q))[ok.ravel()]}))
    else:
        D = torch.from_numpy(np.ascontiguousarray(E1[di])).cuda()
        for st in range(0, len(qi), 4096):
            q = qi[st:st + 4096]
            s, j = (torch.from_numpy(np.ascontiguousarray(E2[q])).cuda() @ D.T).topk(k, dim=1)
            out.append(pd.DataFrame({"rid": np.repeat(q, k).astype(np.int32),
                                     "sid": di[j.cpu().numpy().ravel()].astype(np.int32),
                                     "score": s.float().cpu().numpy().ravel(),
                                     "rank": np.tile(np.arange(k, dtype=np.int8), len(q))}))
        del D; torch.cuda.empty_cache()
    print("country", c, "queries", len(qi), "s1", len(di), flush=True)
cand = pd.concat(out, ignore_index=True)
cand.to_parquet(f"{WORK}/cand_{split}.parquet")
print("candidates", len(cand), flush=True)

if split == "train":  # blocking recall on the folds the embedder never saw
    gt = load("train", "ground_truth")
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m").query("m != ''")
    rid = pd.Series(np.arange(len(other)), index=other.entity_id).loc[pairs.m].values
    sid = pd.Series(np.arange(len(s1)), index=s1.entity_id).loc[pairs.source1_entity_id].values
    fold = s1_fold(pairs.source1_entity_id.tolist())
    t = pd.DataFrame({"rid": rid, "sid": sid, "fold": fold})
    m = t.merge(cand, on=["rid", "sid"], how="left")
    for name, msk in [("embed folds 0-3", m.fold < 4), ("unseen folds 4-9", m.fold >= 4)]:
        r = m[msk]["rank"]
        print(name, {k: round(float((r < k).mean()), 4) for k in [1, 2, 3, 5, 10, 20] if k <= K})
