"""Shortlist study for the merged model (candidate-set size now counts toward the final ranking).

Train (records of S1 folds 4-6 fit, 7-9 evaluate, as x_shortlist2.py): three calibrated shortlist models on the
retrieved top-20 pairs, compared on true S1 kept vs candidates per record / per S1:
  retrieval       Sahaj's v8 features (cosine, rank, gaps, near-ties, softmax share)
  +text           plus cheap text similarities of each pair and their gap to the record's best (Sahaj's v9)
  +text+S1        plus the S1 side of the retrieved pairs (how many records claim the S1, this record's rank and
                  gap among them), from the v8-generalize branch
Test: applies Sahaj's saved v9 text model (work/shortlist_text.txt) and finds the threshold that reproduces his v9
pair set (work/feats2_test.parquet), so the merged code can regenerate it; also saves every test pair's probability
under each model (x/shortlist_p_test.parquet) for choosing the final threshold.
  python g8_shortlist_combo.py"""
import os, json, numpy as np, pandas as pd, lightgbm as lgb
from rapidfuzz import process, fuzz
from common import WORK, load
from harness import truth_arrays
from match import retrieval_feats, normed

NT = int(os.environ.get("NT", 8))
BASE_F = ["score", "rank", "gap_top1", "gap_next", "top1", "n02", "n05", "n10", "soft"]
S1_F = ["r20_s1_rank", "r20_s1_gap", "r20_s1_n", "r20_s1_top"]
TXT = ["t_nn", "t_cn", "t_na", "t_nn_gap", "t_cn_gap", "t_na_gap"]
MODELS = {"retrieval": BASE_F, "+text": BASE_F + TXT, "+text+S1": BASE_F + TXT + S1_F}
TAUS = (0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005)
PRM = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, verbose=-1, num_threads=NT)


def text_feats(c, split):
    """Sahaj's x_shortlist2 text similarities of each retrieved pair (normalised name, core name, address)."""
    n1 = normed(split, 1)
    n2 = pd.concat([normed(split, 2), normed(split, 3)], ignore_index=True)
    P = dict(workers=NT, dtype=np.float32)
    for name, col, scorer in (("t_nn", "nn", fuzz.token_set_ratio), ("t_cn", "cn", fuzz.ratio),
                              ("t_na", "na", fuzz.token_set_ratio)):
        c[name] = process.cpdist(n1[col].values[c.sid.values].tolist(), n2[col].values[c.rid.values].tolist(),
                                 scorer=scorer, **P)
        c[name + "_gap"] = c.groupby("rid")[name].transform("max").values - c[name].values
    return c


def per_country(split):
    """retrieval + S1-side + text features, country by country (S1s never cross countries)."""
    c = pd.read_parquet(f"{WORK}/cand_{split}.parquet")
    ctry = pd.concat([load(split, 2), load(split, 3)], ignore_index=True).country.values
    for cc in sorted(set(ctry[c.rid.values])):
        part = c[ctry[c.rid.values] == cc].sort_values(["rid", "rank"]).reset_index(drop=True)
        yield cc, part


def main():
    s1, other, ts, s1f, rf = truth_arrays()
    tr_parts, ev_parts = [], []
    for cc, c in per_country("train"):
        c = retrieval_feats(c)                                       # S1-side features need every pair
        c = c[rf[c.rid.values] >= 4].reset_index(drop=True)
        c = text_feats(c, "train")
        keep = ["rid", "sid"] + BASE_F + TXT + S1_F
        r = c.rid.values
        tr_parts.append(c.loc[np.isin(rf[r], [4, 5, 6]) & (r % 4 == 0), keep])
        ev_parts.append(c.loc[np.isin(rf[r], [7, 8, 9]), keep])
        print("train", cc, len(c), flush=True)
    T, E = pd.concat(tr_parts, ignore_index=True), pd.concat(ev_parts, ignore_index=True)
    del tr_parts, ev_parts
    yT, yE = ts[T.rid.values] == T.sid.values, ts[E.rid.values] == E.sid.values
    real = np.unique(E.rid.values[ts[E.rid.values] >= 0])
    n_rec, n_s1 = E.rid.nunique(), int((s1f >= 7).sum())
    res, models = {}, {}
    for name, F in MODELS.items():
        m = lgb.train(PRM, lgb.Dataset(T[F], yT), 300)
        models[name] = m
        m.save_model(f"{WORK}/shortlist_{name.replace('+', 'p')}.txt")
        p = m.predict(E[F], num_threads=NT)
        for tau in TAUS:
            k = p >= tau
            hit = float(np.isin(real, E.rid.values[k & yE]).mean())
            res[f"{name} {tau}"] = dict(kept=hit, per_record=float(k.sum() / n_rec), per_s1=float(k.sum() / n_s1))
            print(f"{name:10s} tau {tau:<7} true S1 kept {hit:.4%}   per record {k.sum() / n_rec:5.3f}   "
                  f"per S1 {k.sum() / n_s1:6.2f}", flush=True)
    del T, E
    # test: Sahaj's v9 text model, and every model's probability per test pair
    sahaj = lgb.Booster(model_file=f"{WORK}/shortlist_text.txt")
    outs = []
    for cc, c in per_country("test"):
        c = text_feats(retrieval_feats(c), "test")
        o = c[["rid", "sid"]].copy()
        o["p_sahaj_v9"] = sahaj.predict(c[BASE_F + TXT], num_threads=NT).astype(np.float32)
        for name, F in MODELS.items():
            o[f"p_{name}"] = models[name].predict(c[F], num_threads=NT).astype(np.float32)
        outs.append(o)
        print("test", cc, len(c), flush=True)
    o = pd.concat(outs, ignore_index=True)
    o.to_parquet(f"{WORK}/x/shortlist_p_test.parquet")
    k2 = pd.read_parquet(f"{WORK}/feats2_test.parquet", columns=["rid", "sid"])
    target = len(k2)
    ps = np.sort(o.p_sahaj_v9.values)[::-1]
    tau_star = float(ps[target - 1])
    kept = o[o.p_sahaj_v9.values >= tau_star]
    key = lambda d: d.rid.values.astype(np.int64) * (1 << 22) + d.sid.values
    same = len(np.intersect1d(key(kept), key(k2)))
    print(f"Sahaj v9 test pairs {target}: threshold reproducing that count {tau_star:.6f}; kept {len(kept)}, "
          f"identical pairs {same} ({same / target:.4%})", flush=True)
    for tau in (0.001, 0.002, 0.003, 0.005, 0.01):
        print(f"  sahaj model tau {tau}: {int((o.p_sahaj_v9.values >= tau).sum())} test pairs", flush=True)
    json.dump({"train": res, "sahaj_tau_star": tau_star, "sahaj_pairs": target, "identical": int(same)},
              open(f"{WORK}/x/shortlist_combo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
