"""Per-entity decision that maximises expected macro F0.5.
Each S2/S3 record points to its best S1 candidate with probability p. For one S1 entity, sort its records by p
and choose the k (top-k) that maximises E[F0.5], computed exactly: TP ~ PoissonBinomial(chosen p),
missed ~ PoissonBinomial(unchosen p). `lam` rescales the odds for the test prior (more distractors than train)."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _pb(q):
    """Poisson-binomial pmf of a sum of Bernoulli(q)."""
    d = np.zeros(len(q) + 1); d[0] = 1.0
    for i in range(len(q)):
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1 - q[i]) + d[j - 1] * q[i]
        d[0] *= 1 - q[i]
    return d


@njit(cache=True)
def _best_k(q, extra):
    """q sorted descending. extra = expected true matches that are not among the candidates (blocking misses)."""
    n = len(q)
    best_k, best_v = 0, -1.0
    for k in range(n + 1):
        a = _pb(q[:k]); b = _pb(q[k:])
        v = 0.0
        for x in range(k + 1):
            if a[x] == 0.0:
                continue
            for y in range(n - k + 1):
                pxy = a[x] * b[y]
                if pxy == 0.0:
                    continue
                t = x + y + extra
                if t == 0.0:
                    f = 1.0 if k == 0 else 0.0
                elif x == 0:
                    f = 0.0
                else:
                    pr = x / k; rc = x / t
                    f = 1.25 * pr * rc / (0.25 * pr + rc)
                v += pxy * f
        if v > best_v:
            best_v, best_k = v, k
    return best_k


@njit(parallel=True, cache=True)
def choose(starts, q, extra, cap):
    """starts: CSR offsets of entities over q (each entity's q sorted descending). Returns chosen k per entity."""
    m = len(starts) - 1
    out = np.zeros(m, np.int64)
    for e in prange(m):
        s, t = starts[e], min(starts[e + 1], starts[e] + cap)
        if t > s:
            out[e] = _best_k(q[s:t], extra)
    return out


def adjust(p, lam):
    """prior shift: divide the match odds by lam."""
    return p / (p + (1 - p) * lam)


def select(sid, p, lam=1.0, extra=0.0, cap=24):
    """sid, p: one row per record (its best S1 and probability). Returns boolean mask of accepted rows."""
    q = adjust(np.asarray(p, np.float64), lam)
    order = np.lexsort((-q, sid))
    s_sorted, q_sorted = sid[order], q[order]
    bounds = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1], True])
    k = choose(bounds.astype(np.int64), q_sorted, extra, cap)
    rank = np.arange(len(order)) - np.repeat(bounds[:-1], np.diff(bounds))
    acc = np.zeros(len(p), bool)
    acc[order] = rank < np.repeat(k, np.diff(bounds))
    return acc


if __name__ == "__main__":
    # sanity: one confident + one coin-flip on an entity; a lone 0.6 on a likely singleton
    import numpy as np
    sid = np.array([0, 0, 1, 2, 2, 2, 2]); p = np.array([0.99, 0.5, 0.6, 0.99, 0.98, 0.97, 0.7])
    print(select(sid, p))  # expect: 0.5 rejected, 0.6 accepted, 0.7 rejected next to 3 confident
