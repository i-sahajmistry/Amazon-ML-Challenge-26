"""Candidate-set size vs F0.5 for the text shortlist (candidate_pairs.tsv counts toward the final ranking).
Scores every retrieved train pair of records of S1 folds 4-9 with the saved text shortlist model
(work/shortlist_text.txt), then filters the copy-free stage-2 validation predictions of the merged model
(x/val_q{TAG}w.parquet, written by x_nocopy.py test) to pairs with P >= tau, as a stricter shortlist would: a record
whose argmax pair drops out counts as rejected. Test side: pairs and candidates per S1 at the same tau
(x/shortlist_p_test.parquet from g8_shortlist_combo.py).
  TAG=_v9f python g9_text_tau.py"""
import os, time, json, numpy as np, pandas as pd, lightgbm as lgb
from common import WORK, load
from harness import truth_arrays, wscore
from match import retrieval_feats, text_feats, normed, TEXT_F

NT = int(os.environ.get("NT", 8))
TAG = os.environ.get("TAG", "_v9f")
TAUS = (0.001, 0.002, 0.003, 0.005, 0.01, 0.02)
key = lambda r, s: r.astype(np.int64) * (1 << 22) + s


def train_p():
    f = f"{WORK}/x/shortlist_p_train.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    _, _, ts, s1f, rf = truth_arrays()
    c = pd.read_parquet(f"{WORK}/cand_train.parquet")
    c = retrieval_feats(c[rf[c.rid.values] >= 4].sort_values(["rid", "rank"]).reset_index(drop=True))
    c = text_feats(c, normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True))
    p = lgb.Booster(model_file=f"{WORK}/shortlist_text.txt").predict(c[TEXT_F], num_threads=NT)
    o = pd.DataFrame({"k": key(c.rid.values, c.sid.values), "p": p.astype(np.float32)})
    o.to_parquet(f)
    return o


def main():
    P = train_p().set_index("k").p
    vf = f"{WORK}/x/val_q{TAG}w.parquet"
    while not os.path.exists(vf):
        time.sleep(30)
    time.sleep(10)
    v = pd.read_parquet(vf)
    s1, _, ts, s1f, rf = truth_arrays()
    T = np.bincount(ts[ts >= 0], minlength=len(s1))
    ents = np.where(s1f >= 8)[0]
    pv = P.reindex(key(v.rid.values, v.sid.values)).values
    sid, y, w, q = v.sid.values, v.y.values.astype(bool), v.wt.values, v.q.values
    t = pd.read_parquet(f"{WORK}/x/shortlist_p_test.parquet", columns=["sid", "p_sahaj_v9"])
    c1 = load("test", 1).country.values
    base = {thr: wscore(sid, q >= thr, y, w, T, ents) for thr in (0.65, 0.7)}
    print(f"merged {TAG} validation: thr 0.65 {base[0.65]:.5f}, 0.70 {base[0.7]:.5f} (shortlist tau 0.001)", flush=True)
    rows = []
    print(f"{'tau':>6} {'F 0.65':>8} {'F 0.70':>8} {'delta':>9} | {'test pairs':>11} {'per S1':>7} | US / India / France",
          flush=True)
    for tau in TAUS:
        f = {thr: wscore(sid, (q >= thr) & (pv >= tau), y, w, T, ents) for thr in (0.65, 0.7)}
        k = t.p_sahaj_v9.values >= tau
        per = np.bincount(t.sid.values[k], minlength=len(c1))
        byc = [per[c1 == c].mean() for c in ("US", "India", "France")]
        rows.append(dict(tau=tau, f065=f[0.65], f070=f[0.7], pairs=int(k.sum()), per_s1=float(per.mean()),
                         us=byc[0], india=byc[1], france=byc[2]))
        print(f"{tau:>6} {f[0.65]:>8.5f} {f[0.7]:>8.5f} {f[0.7] - base[0.7]:>+9.5f} | {k.sum():>11,} {per.mean():>7.2f} | "
              f"{byc[0]:.2f} / {byc[1]:.2f} / {byc[2]:.2f}", flush=True)
    json.dump({"base": base, "rows": rows}, open(f"{WORK}/x/text_tau{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
