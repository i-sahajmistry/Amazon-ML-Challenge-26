"""Pick the ANN index for retrieve.py: on one train country (COUNTRY, default India) and 100k of its records, true-S1
recall @1 / @20 and overlap with the exact top-20, build and query time, for exact GPU search and FAISS HNSW / IVF
settings. HNSW "centered" indexes S1 embeddings minus their mean (queries unchanged, so each query's ranking is the
same inner product minus a constant): these embeddings are all very similar, which makes the graph hard to navigate.
  COUNTRY=US ANN_SET=hnsw2 python x_ann.py"""
import os, time, numpy as np, pandas as pd, torch, faiss
from common import WORK
from harness import truth_arrays

faiss.omp_set_num_threads(32)
s1, other, ts, s1f, rf = truth_arrays()
E1, E2 = np.load(f"{WORK}/emb_train_s1.npy", mmap_mode="r"), np.load(f"{WORK}/emb_train_other.npy", mmap_mode="r")
CTRY = os.environ.get("COUNTRY", "India")
di = np.where(s1.country.values == CTRY)[0]
qi = np.random.default_rng(0).choice(np.where(other.country.values == CTRY)[0], 100_000, replace=False)
X, Q = np.ascontiguousarray(E1[di], dtype=np.float32), np.ascontiguousarray(E2[qi], dtype=np.float32)
real = ts[qi] >= 0
t0 = time.time()
D = torch.from_numpy(X).cuda().half()
ex = torch.cat([(torch.from_numpy(Q[i:i + 8192]).cuda().half() @ D.T).topk(20, dim=1).indices.cpu()
                for i in range(0, len(Q), 8192)]).numpy()
print(f"exact GPU        query {time.time() - t0:6.1f}s", flush=True)


def report(name, I, tb, tq):
    sid = np.where(I >= 0, di[np.maximum(I, 0)], -1)
    hit1 = (sid[:, 0] == ts[qi])[real].mean()
    hit20 = (sid == ts[qi][:, None]).any(1)[real].mean()
    ov = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(I[:20000], ex[:20000])])
    print(f"{name:26s} build {tb:6.1f}s query {tq:6.1f}s  true@1 {hit1:.4f} true@20 {hit20:.4f}  overlap-exact {ov:.4f}",
          flush=True)


sid_ex = di[ex]
print(f"{'exact':26s} true@1 {(sid_ex[:, 0] == ts[qi])[real].mean():.4f} true@20 {(sid_ex == ts[qi][:, None]).any(1)[real].mean():.4f}")
def hnsw(name, Xb, M, efs):
    t0 = time.time(); ix = faiss.IndexHNSWFlat(Xb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = 200; ix.add(Xb); tb = time.time() - t0
    for ef in efs:
        ix.hnsw.efSearch = ef
        t0 = time.time(); _, I = ix.search(Q, 20); report(f"{name} efS{ef}", I, tb, time.time() - t0)


if os.environ.get("ANN_SET") == "hnsw2":
    hnsw("HNSW M32", X, 32, (512, 768))
    hnsw("HNSW M32 centered", np.ascontiguousarray(X - X.mean(0)), 32, (256, 512))
    hnsw("HNSW M48", X, 48, (512,))
else:
    hnsw("HNSW M32", X, 32, (64, 128, 256, 512))
    for nl_mult in (1, 4):
        nlist = int(nl_mult * np.sqrt(len(X)))
        t0 = time.time(); ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(X.shape[1]), X.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        ix.train(X[:256 * nlist]); ix.add(X); tb = time.time() - t0
        for npb in (32, 128, 256):
            ix.nprobe = npb
            t0 = time.time(); _, I = ix.search(Q, 20); report(f"IVF nlist{nlist} nprobe{npb}", I, tb, time.time() - t0)
