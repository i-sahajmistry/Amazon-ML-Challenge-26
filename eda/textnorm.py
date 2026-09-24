"""Text normalization helpers shared by the EDA scripts."""
import re
import unicodedata

LEGAL = set("""inc incorporated llc l l c ltd limited pvt private corp corporation co company plc llp lp pc pllc
the and of services service group center centre sarl sas eurl sa ei ets fils cie""".split())
NULLS = {"null", "n/a", "na", "none"}


def norm(s):
    """Lowercase, strip accents, replace punctuation with spaces, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_tokens(s):
    return {t for t in norm(s).split() if t not in LEGAL and len(t) > 1}


def addr_tokens(s):
    return {t for t in norm(s).split() if t not in NULLS}


def nums(s):
    return set(re.findall(r"\d+", s))


def is_latin(s):
    return not re.search(r"[ऀ-෿]", s)
