"""
text_preprocessing.py — Scientific text preprocessing.

Normalises whitespace, removes noise, and produces clean_title,
clean_abstract, and clean_document fields while preserving medical
terminology.
"""

import re
import pandas as pd

from .config_loader import load_config, ensure_dirs


# ================================================================
# TEXT CLEANING FUNCTIONS
# ================================================================

def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r"\S+@\S+\.\S+", "", text)


def _remove_html_tags(text: str) -> str:
    """Remove HTML/XML tags."""
    return re.sub(r"<[^>]+>", " ", text)


def _remove_formatting_noise(text: str) -> str:
    """Remove common formatting noise from scientific text."""
    # Remove copyright notices
    text = re.sub(
        r"(?:©|copyright)\s*\d{4}[^.]*\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove "Published by..." patterns
    text = re.sub(
        r"published\s+by\s+[^.]+\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove ClinicalTrials.gov identifiers
    text = re.sub(r"NCT\d{8}", "", text)
    # Remove ISRCTN identifiers
    text = re.sub(r"ISRCTN\d+", "", text)
    # Remove reference markers like [1], [2,3]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    return text


def _normalize_punctuation(text: str) -> str:
    """Normalise punctuation without removing meaningful characters."""
    # Replace multiple periods with single
    text = re.sub(r"\.{2,}", ".", text)
    # Replace multiple hyphens with single
    text = re.sub(r"-{2,}", "-", text)
    # Remove leading/trailing hyphens from words
    text = re.sub(r"\b-|-\b", " ", text)
    # Fix multiple spaces from removals
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str, cfg: dict | None = None) -> str:
    """Apply full cleaning pipeline to a single text string.

    Parameters
    ----------
    text : raw text
    cfg : preprocessing config dict (optional)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if cfg is None:
        cfg = load_config().get("preprocessing", {})

    result = text

    # HTML tags
    result = _remove_html_tags(result)

    # URLs
    if cfg.get("remove_urls", True):
        result = _remove_urls(result)

    # Emails
    if cfg.get("remove_emails", True):
        result = _remove_emails(result)

    # Formatting noise
    result = _remove_formatting_noise(result)

    # Punctuation normalisation
    result = _normalize_punctuation(result)

    # Whitespace
    if cfg.get("normalize_whitespace", True):
        result = _normalize_whitespace(result)

    return result


# ================================================================
# CORPUS PREPROCESSING
# ================================================================

def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the entire corpus.

    Adds clean_title, clean_abstract, clean_document columns.
    Filters out extremely short documents.
    """
    cfg = load_config()
    prep_cfg = cfg.get("preprocessing", {})
    min_len = prep_cfg.get("min_clean_doc_length", 50)

    result = df.copy()

    print("\n[PREPROCESS] Cleaning text fields...")
    print(f"  Input records: {len(result):,}")

    # Clean individual fields
    result["clean_title"] = result["title"].fillna("").apply(
        lambda t: clean_text(t, prep_cfg)
    )
    result["clean_abstract"] = result["abstract"].fillna("").apply(
        lambda t: clean_text(t, prep_cfg)
    )

    # Build clean_document from clean_title + clean_abstract
    result["clean_document"] = (
        result["clean_title"] + ". " + result["clean_abstract"]
    ).str.strip()

    # Remove records with extremely short cleaned documents
    before = len(result)
    mask = result["clean_document"].str.len() >= min_len
    removed_short = result[~mask]
    result = result[mask].reset_index(drop=True)

    print(f"  Removed {before - len(result):,} records with "
          f"clean_document < {min_len} chars")
    print(f"  Final preprocessed corpus: {len(result):,} records")

    # Show before/after examples
    print("\n--- Before/After Text Examples ---")
    for i in range(min(3, len(result))):
        row = result.iloc[i]
        print(f"\n  PMID {row['pmid']}:")
        orig = str(row.get("document", ""))[:200]
        clean = str(row["clean_document"])[:200]
        print(f"    ORIGINAL: {orig}...")
        print(f"    CLEANED:  {clean}...")

    return result


def save_preprocessed(df: pd.DataFrame) -> dict:
    """Save preprocessed corpus."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "mhealth_preprocessed.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["preprocessed"] = p
    print(f"[SAVE] Preprocessed corpus -> {p}  ({len(df):,} records)")

    return saved
