"""House-number positions in an S1's group (the records whose best candidate is that S1), for the same-address fixes
(x_ddfix.py): SAME (the S1's first house number), D (the most common other first number in the group: its shifted
decoy cluster), OTHER (another number), NONE (no number); and the words a record adds to / drops from the S1's core
name."""
import numpy as np

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
