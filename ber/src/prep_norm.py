import time
from match import normed
for split in ["train", "test"]:
    for src in [1, 2, 3]:
        t = time.time(); d = normed(split, src); print(split, src, len(d), round(time.time() - t), "s", flush=True)
print(d.head(3).to_string())
