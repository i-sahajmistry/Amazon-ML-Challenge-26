"""Load the challenge TSVs with pyarrow and cache them as parquet in eda/cache/."""
import os
import pyarrow.csv as pc
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))
# Repo layout: student_resource/ at the repo root. Override with DATA_DIR.
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "..", "student_resource", "dataset"))
CACHE = os.path.join(ROOT, "cache")
FILES = {
    "train_s1": "train/train_source1.tsv", "train_s2": "train/train_source2.tsv",
    "train_s3": "train/train_source3.tsv", "train_gt": "train/train_ground_truth.tsv",
    "test_s1": "test/test_source1.tsv", "test_s2": "test/test_source2.tsv",
    "test_s3": "test/test_source3.tsv",
}


def read_tsv(path):
    # quote_char=False: fields are raw text and may contain quote characters.
    return pc.read_csv(
        path,
        parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pc.ConvertOptions(strings_can_be_null=False,
                                          column_types=None),
    )


def load(name):
    """Return a pandas DataFrame for one file, using the parquet cache if present."""
    cached = os.path.join(CACHE, name + ".parquet")
    if os.path.exists(cached):
        return pq.read_table(cached).to_pandas()
    table = read_tsv(os.path.join(DATA, FILES[name]))
    os.makedirs(CACHE, exist_ok=True)
    pq.write_table(table, cached)
    return table.to_pandas()


if __name__ == "__main__":
    for n in FILES:
        df = load(n)
        print(n, df.shape, list(df.columns), dict(df.dtypes.astype(str)))
