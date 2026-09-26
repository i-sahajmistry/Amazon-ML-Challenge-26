"""Entity-group anatomy: where true matches and decoys sit relative to their S1's house number, labelled (US / India
validation, x/val_q_v10pw) vs France test (the best file's decisions, x/test_q_v10fr3l).
Per S1 group (records whose best candidate is the S1), D = the most common first house number among the group's
records that differ from the S1's; each record is at SAME (S1's number), D, OTHER (another number) or NONE (no number).
Prints label / acceptance rates by position and the words records add at SAME, and dumps labelled groups to read.
  python x_anatomy.py -> x/anat_{US,India}_val.txt"""
import numpy as np, pandas as pd
from collections import Counter
from common import WORK, load
from match import normed

XD = f"{WORK}/x"
first = lambda s: int(s.split()[0][:15]) if s else -1


def positions(d, n1, n2):
    a = np.array([first(x) for x in n1.num.values[d.sid.values]])
    b = np.array([first(x) for x in n2.num.values[d.rid.values]])
    d = d.assign(a=a, b=b)
    other = d[(d.b >= 0) & (d.a >= 0) & (d.b != d.a)]
    D = other.groupby(["sid", "b"]).size().reset_index().sort_values(0).drop_duplicates("sid", keep="last").set_index("sid").b
    dd = D.reindex(d.sid.values).fillna(-2).values
    pos = np.where((a < 0) | (b < 0), "NONE", np.where(b == a, "SAME", np.where(b == dd, "D", "OTHER")))
    s1w = [set(x.split()) for x in n1.cn.values[d.sid.values]]
    rw = [set(x.split()) for x in n2.cn.values[d.rid.values]]
    return d.assign(pos=pos, up=np.sign(b - a), add=[" ".join(sorted(y - x)) for x, y in zip(s1w, rw)],
                    drop=[" ".join(sorted(x - y)) for x, y in zip(s1w, rw)])


def words(d, title, k=25):
    c = Counter(w for s in d["add"].values for w in s.split())
    n = len(d)
    print(f"  {title} ({n} rows; {np.mean(d['add'].values != ''):.3f} add a word): "
          + ", ".join(f"{w} {v / n:.3f}" for w, v in c.most_common(k)), flush=True)


if __name__ == "__main__":
    n1, n2 = normed("train", 1), pd.concat([normed("train", 2), normed("train", 3)], ignore_index=True)
    s1 = load("train", 1); o = pd.concat([load("train", 2), load("train", 3)], ignore_index=True)
    v = pd.read_parquet(f"{XD}/val_q_v10pw.parquet")
    v = positions(v.assign(c=s1.country.values[v.sid.values]), n1, n2)
    for c in ("US", "India"):
        d = v[v.c.values == c]
        print(f"\n== {c} validation: {len(d)} rows over {d.sid.nunique()} S1s; true share {d.y.mean():.3f}", flush=True)
        for p in ("SAME", "D", "OTHER", "NONE"):
            m = d.pos.values == p
            print(f"  {p:5s} rows/S1 {m.sum() / d.sid.nunique():.3f}  true {d.y.values[m].mean():.4f}  "
                  f"accepted {np.mean(d.q.values[m] >= 0.7):.4f}  decoys here / all decoys {(~d.y.values[m]).sum() / (~d.y.values).sum():.3f}",
                  flush=True)
        words(d[(d.pos == "SAME") & d.y], "true @SAME adds")
        words(d[(d.pos == "SAME") & ~d.y], "decoy @SAME adds")
        words(d[(d.pos == "D") & ~d.y], "decoy @D adds")
        rng = np.random.default_rng(0)
        pick = rng.choice(d.sid.unique(), 40, replace=False)
        with open(f"{XD}/anat_{c}_val.txt", "w") as f:
            for s in pick:
                r = s1.iloc[s]
                f.write(f"S1 {r.business_name} | {r.business_address}\n")
                for x in d[d.sid == s].sort_values("q", ascending=False).itertuples():
                    t = o.iloc[x.rid]
                    f.write(f"  {'TRUE ' if x.y else 'decoy'} {x.pos:5s} q={x.q:.2f} | {t.business_name} | {t.business_address}\n")
                f.write("\n")

    t1, t2 = normed("test", 1), pd.concat([normed("test", 2), normed("test", 3)], ignore_index=True)
    t = pd.read_parquet(f"{XD}/test_q_v10fr3l.parquet")
    t = positions(t, t1, t2)
    for c in ("US", "India", "France"):
        d = t[t.c.values == c]
        acc = d.q.values >= 0.7
        print(f"\n== {c} test: {len(d)} rows over {d.sid.nunique()} S1s; accepted {acc.mean():.3f}", flush=True)
        for p in ("SAME", "D", "OTHER", "NONE"):
            m = d.pos.values == p
            print(f"  {p:5s} rows/S1 {m.sum() / d.sid.nunique():.3f}  accepted {acc[m].mean():.4f}", flush=True)
        words(d[(d.pos == "SAME") & acc], "accepted @SAME adds")
        words(d[(d.pos == "SAME") & ~acc], "rejected @SAME adds")
        words(d[(d.pos == "D")], "@D adds")
