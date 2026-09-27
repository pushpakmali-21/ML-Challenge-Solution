"""
Normalization utilities for business names and addresses.

Goal: collapse superficial noise (abbreviations, punctuation, casing,
legal-suffix variants, common address abbreviations) into a canonical
form so that downstream string-similarity and blocking features are
comparing "like with like" as much as possible.

This is intentionally rule-based and language/script agnostic where
possible -- it does NOT call any external service (per challenge rules).
"""
import re
import unicodedata

# --- Legal-suffix / business-name abbreviation normalization ---------------
# Map noisy variant -> canonical short token. Order matters: longer phrases first.
LEGAL_SUFFIX_MAP = {
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\.?\s*ltd\.?\b": "pvt ltd",
    r"\bpvt\.?\b": "pvt",
    r"\bltd\.?\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\bcompany\b": "co",
    r"\bco\.?\b": "co",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bl\.l\.c\.?\b": "llc",
    r"\bl\.l\.p\.?\b": "llp",
    r"\band\b": "&",
    r"\bbros\.?\b": "brothers",
    r"\bindustries\b": "inds",
    r"\benterprises?\b": "ent",
}

# --- Address abbreviation normalization -------------------------------------
ADDRESS_ABBR_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bstr\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bapt\.?\b": "apartment",
    r"\bbldg\.?\b": "building",
    r"\bfl\.?\b": "floor",
    r"\bno\.?\b": "number",
    r"\bnr\.?\b": "near",
    r"\bopp\.?\b": "opposite",
    r"\bsec\.?\b": "sector",
    r"\bcolny\b": "colony",
    r"\bnagar\b": "nagar",
    r"\bdist\.?\b": "district",
    r"\bp\.?o\.?\b": "post office",
    r"\bpin\.?\b": "pincode",
    r"\bzip\.?\b": "zipcode",
}

_PUNCT_RE = re.compile(r"[^\w\s&]")
_MULTISPACE_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Remove diacritics (transliteration variance) without external services."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _apply_map(text: str, mapping: dict) -> str:
    for pattern, repl in mapping.items():
        text = re.sub(pattern, repl, text)
    return text


def basic_clean(text) -> str:
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text)
    text = _strip_accents(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(text) -> str:
    text = basic_clean(text)
    text = _apply_map(text, LEGAL_SUFFIX_MAP)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_address(text) -> str:
    text = basic_clean(text)
    text = _apply_map(text, ADDRESS_ABBR_MAP)
    # drop pure "landmark filler" words that add noise, not signal
    text = re.sub(r"\b(near|opposite|behind|beside)\b", "", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def token_set(text: str) -> set:
    return set(t for t in text.split() if t)
