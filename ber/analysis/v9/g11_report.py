"""Score report for a finished run (TAG): macro F0.5 with distractors weighted to the 39% test share (copy-free),
micro precision / recall, per country.
  train  stage 1, out of fold: records of S1 folds 4-7 (each scored by the stage-1 model that did not see them),
         every record at its argmax pair, best threshold of the stage-1 probability
  val    S1 folds 8-9: stage 1 as above, and stage 2 (x/val_q{TAG}w.parquet: fitted on folds 4-7) at 0.65 / 0.70
  TAG=_v9fS python g11_report.py"""
import os, numpy as np, pandas as pd
from common import WORK, load
from harness import truth_arrays, wscore

TAG = os.environ.get("TAG", "_v9fS")
XD = f"{WORK}/x"


def stats(sid, acc, y, wt, T, ents):
    f = wscore(sid, acc, y, wt, T, ents)
    tp = (acc & y).sum(); fp = (wt * (acc & ~y)).sum()
    return f, tp / max(tp + fp, 1e-9), tp / max(T[ents].sum(), 1)


def main():
    s1, other, ts, s1f, rf = truth_arrays()
    matched = ts >= 0
    W = int(0.39 / 0.61 * matched.sum()) / np.sum(~matched & (rf >= 4))
    T = np.bincount(ts[matched], minlength=len(s1))
    c1 = s1.country.values
    k = pd.read_parquet(f"{XD}/oof5_train{TAG}.parquet")
    k = k[rf[k.rid.values] >= 4]
    o = np.lexsort((-k.p.values, k.rid.values))
    b = k.iloc[o[np.r_[True, k.rid.values[o][1:] != k.rid.values[o][:-1]]]]
    b = b[s1f[b.sid.values] >= 4]
    y = ts[b.rid.values] == b.sid.values
    wt = np.where(matched[b.rid.values], 1.0, W)
    for name, folds in (("train (S1 folds 4-7)", (4, 5, 6, 7)), ("val (S1 folds 8-9)", (8, 9))):
        m = np.isin(s1f[b.sid.values], folds)
        for cc in ("all", "US", "India"):
            ents = np.flatnonzero(np.isin(s1f, folds) & ((c1 == cc) if cc != "all" else True))
            mm = m & (np.isin(c1[b.sid.values], [cc]) if cc != "all" else m)
            res = {t: stats(b.sid.values[mm], b.p.values[mm] >= t, y[mm], wt[mm], T, ents) for t in (0.4, 0.5, 0.6, 0.7)}
            t = max(res, key=lambda x: res[x][0])
            f, p, r = res[t]
            print(f"stage 1  {name:22s} {cc:6s} F0.5 {f:.5f} (thr {t})  precision {p:.5f}  recall {r:.5f}", flush=True)
    v = pd.read_parquet(f"{XD}/val_q{TAG}w.parquet")
    ents_all = np.flatnonzero(s1f >= 8)
    for thr in (0.65, 0.70):
        for cc in ("all", "US", "India"):
            ents = ents_all if cc == "all" else np.flatnonzero((s1f >= 8) & (c1 == cc))
            mm = np.ones(len(v), bool) if cc == "all" else (c1[v.sid.values] == cc)
            f, p, r = stats(v.sid.values[mm], v.q.values[mm] >= thr, v.y.values[mm].astype(bool), v.wt.values[mm], T, ents)
            print(f"stage 2  val (S1 folds 8-9) thr {thr:.2f} {cc:6s} F0.5 {f:.5f}  precision {p:.5f}  recall {r:.5f}", flush=True)


if __name__ == "__main__":
    main()
