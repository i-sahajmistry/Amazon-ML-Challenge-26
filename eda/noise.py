import re
import unicodedata
import pandas as pd
from load import load

SCRIPTS = {"Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
           "Gujarati": (0x0A80, 0x0AFF), "Odia": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
           "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F)}


def script_of(s):
    for ch in s:
        o = ord(ch)
        for k, (a, b) in SCRIPTS.items():
            if a <= o <= b:
                return k
    return "Latin/other"


PAT = {
    "domain_name": r"(?i)^[a-z0-9\-]+\.(com|in|net|org|co|co\.in|biz|info|fr)$",
    "junk_prefix": r"^[^\w\s]{2,}\s",
    "leet_digit_in_word": r"(?i)[a-z][0-9][a-z]",
    "double_space": r"  ",
    "bracket_suffix": r"[\[\(][^\]\)]*[\]\)]\s*$",
}
APAT = {
    "null_token": r"(?i)\b(null|n/a|none)\b",
    "po_box/pmb": r"(?i)\b(po box|p\.o\. box|pmb)\b",
    "us_zip5": r"\b\d{5}(-\d{4})?\b",
    "in_pin6": r"\b\d{6}\b|\b\d{3} \d{3}\b",
    "fr_cp5": r"\b\d{5}\b",
    "near_landmark": r"(?i)\b(near|opp|opposite|behind|nr\.?)\b",
}

rows = []
for split in ["train", "test"]:
    for s in ["s1", "s2", "s3"]:
        df = load(f"{split}_{s}")
        for c, g in df.groupby("country"):
            smp = g.sample(min(len(g), 200_000), random_state=0)
            r = {"file": f"{split}_{s}", "country": c, "n": len(g)}
            for k, p in PAT.items():
                r["name:" + k] = smp.business_name.str.contains(p, regex=True).mean()
            for k, p in APAT.items():
                if (k.startswith("us_") and c != "US") or (k.startswith("in_") and c != "India") or (k.startswith("fr_") and c != "France"):
                    continue
                r["addr:" + k] = smp.business_address.str.contains(p, regex=True).mean()
            if c != "US":
                vc = smp.business_name.map(script_of).value_counts(normalize=True)
                r["name_scripts"] = ", ".join(f"{k} {v:.3f}" for k, v in vc.items())
                vc = smp.business_address.map(script_of).value_counts(normalize=True)
                r["addr_scripts"] = ", ".join(f"{k} {v:.3f}" for k, v in vc.items() if v > 0.001)
            rows.append(r)
out = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200); pd.set_option("display.max_columns", 50)
num = out.drop(columns=["name_scripts", "addr_scripts"])
print(num.set_index(["file", "country"]).T.round(3).to_string())
print()
for r in out.dropna(subset=["name_scripts"]).itertuples():
    print(f"{r.file:9s} {r.country:7s} NAME: {r.name_scripts}\n{'':17s} ADDR: {r.addr_scripts}")
