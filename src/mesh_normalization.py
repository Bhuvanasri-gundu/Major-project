"""
mesh_normalization.py — MeSH term normalisation and
topic-concept mapping.

Normalises formatting variations, maps concepts to MeSH descriptors,
and builds document -> concept -> MeSH -> topic mappings.
"""

import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

from .config_loader import load_config, ensure_dirs


# ================================================================
# NORMALISATION RULES
# ================================================================

# Common MeSH formatting variations to normalise
_NORMALISATION_MAP = {
    # Plural variants
    "mobile applications": "mobile applications",
    "mobile application": "mobile applications",
    # Age-related
    "aged, 80 and over": "aged, 80 and over",
    "child, preschool": "child, preschool",
    # Hyphens vs spaces
    "self-care": "self care",
    "self care": "self care",
    "follow-up": "follow-up studies",
    "cross-sectional studies": "cross-sectional studies",
    # Common equivalences
    "cell phone": "cell phone",
    "cell phones": "cell phone",
    "smartphone": "smartphone",
    "smart phone": "smartphone",
    "telemedicine": "telemedicine",
    "tele-medicine": "telemedicine",
}


def normalize_mesh_term(term: str) -> str:
    """Normalise a single MeSH term.

    Steps:
    1. Strip whitespace
    2. Lowercase
    3. Remove extra spaces
    4. Apply known normalisation mappings
    """
    if not isinstance(term, str):
        return ""

    result = term.strip().lower()
    result = re.sub(r"\s+", " ", result)

    # Apply known mappings
    if result in _NORMALISATION_MAP:
        result = _NORMALISATION_MAP[result]

    return result


def normalize_mesh_terms(df_concepts: pd.DataFrame) -> pd.DataFrame:
    """Normalise all concepts that originate from MeSH terms.

    Parameters
    ----------
    df_concepts : document_concepts dataframe with columns
                  [pmid, topic_id, concept]

    Returns
    -------
    DataFrame with original_concept and normalized_concept columns
    """
    print("\n[MESH] Normalising MeSH concepts...")

    result = df_concepts.copy()
    result["original_concept"] = result["concept"]
    result["normalized_concept"] = result["concept"].apply(normalize_mesh_term)

    # Remove empty after normalisation
    result = result[result["normalized_concept"].str.len() > 0].reset_index(
        drop=True
    )

    n_orig = df_concepts["concept"].nunique()
    n_norm = result["normalized_concept"].nunique()
    print(f"  Original unique concepts: {n_orig:,}")
    print(f"  After normalisation: {n_norm:,}")
    print(f"  Merged {n_orig - n_norm:,} duplicates")

    return result


# ================================================================
# TOPIC-CONCEPT MAPPING
# ================================================================

def build_topic_concept_mapping(
    normalized_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a topic <-> concept mapping table.

    Shows which concepts appear in which topics, and with what frequency.
    """
    print("\n[MESH] Building topic-concept mapping...")

    mapping = (
        normalized_df
        .groupby(["topic_id", "normalized_concept"])
        .agg(
            document_count=("pmid", "nunique"),
            mention_count=("pmid", "size"),
        )
        .reset_index()
        .sort_values(["topic_id", "mention_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    n_topics = mapping["topic_id"].nunique()
    n_concepts = mapping["normalized_concept"].nunique()
    print(f"  Topics: {n_topics}")
    print(f"  Concepts: {n_concepts:,}")
    print(f"  Topic-concept pairs: {len(mapping):,}")

    return mapping


# ================================================================
# SAVE OUTPUTS
# ================================================================

def save_mesh_outputs(
    normalized_df: pd.DataFrame,
    topic_concept_mapping: pd.DataFrame,
) -> dict:
    """Save MeSH normalisation outputs."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "mesh_normalized.csv"
    normalized_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["mesh_normalized"] = p
    print(f"[SAVE] Normalized MeSH -> {p}  ({len(normalized_df):,} rows)")

    p = out / "topic_concept_mapping.csv"
    topic_concept_mapping.to_csv(p, index=False, encoding="utf-8-sig")
    saved["topic_concept_mapping"] = p
    print(f"[SAVE] Topic-concept mapping -> {p}  ({len(topic_concept_mapping):,} rows)")

    return saved
