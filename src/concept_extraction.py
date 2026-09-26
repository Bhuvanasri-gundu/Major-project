"""
concept_extraction.py — Biomedical concept extraction from
MeSH terms, author keywords, and title/abstract.

Builds document-to-concept mappings and concept statistics.
"""

import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from .config_loader import load_config, ensure_dirs


# ================================================================
# CONCEPT EXTRACTION
# ================================================================

def _parse_semicolon_field(value) -> list[str]:
    """Parse a semicolon-separated field into a list of trimmed strings."""
    if pd.isna(value) or not str(value).strip():
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


def _extract_ngrams_from_text(text: str, min_len: int = 3) -> list[str]:
    """Extract simple noun-phrase-like n-grams from text.

    This is a lightweight alternative to spaCy NER.
    Extracts capitalised multi-word phrases and known medical patterns.
    """
    if not isinstance(text, str) or len(text) < min_len:
        return []

    concepts = []

    # Pattern: sequences of capitalised words (potential named entities)
    # e.g., "Type 2 Diabetes", "Blood Pressure"
    cap_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[0-9]+|of|and|the|in|for))*"
        r"(?:\s+[A-Z][a-z]+))\b"
    )
    for match in cap_pattern.finditer(text):
        phrase = match.group(1).strip()
        # Filter too-short or too-long
        if 3 <= len(phrase) <= 80 and len(phrase.split()) <= 5:
            concepts.append(phrase.lower())

    return concepts


def extract_concepts_from_record(row: pd.Series, use_text: bool = False) -> list[str]:
    """Extract biomedical concepts from a single record.

    Sources (in priority order):
    1. MeSH terms (normalised)
    2. Author keywords
    3. Optionally, text-based extraction
    """
    concepts = []

    # MeSH terms
    mesh = _parse_semicolon_field(row.get("mesh_terms"))
    for term in mesh:
        normalised = term.strip().lower()
        if len(normalised) >= 3:
            concepts.append(normalised)

    # Keywords
    kw = _parse_semicolon_field(row.get("keywords"))
    for term in kw:
        normalised = term.strip().lower()
        if len(normalised) >= 3:
            concepts.append(normalised)

    # Text-based extraction (lightweight)
    if use_text:
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        text_concepts = _extract_ngrams_from_text(title + " " + abstract)
        concepts.extend(text_concepts)

    # Deduplicate within record
    seen = set()
    unique = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


def extract_all_concepts(
    df: pd.DataFrame,
    topic_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract concepts from entire corpus.

    Parameters
    ----------
    df : preprocessed dataframe
    topic_labels : optional topic assignments (same length as df)

    Returns
    -------
    doc_concepts : long-format PMID | Topic | Concept
    concept_summary : concept frequency statistics
    """
    cfg = load_config()
    concept_cfg = cfg.get("concepts", {})
    min_freq = concept_cfg.get("min_concept_frequency", 5)
    max_concepts = concept_cfg.get("max_concepts", 5000)
    use_text = concept_cfg.get("use_spacy_ner", False)

    print("\n[CONCEPT] Extracting biomedical concepts...")
    print(f"  Records: {len(df):,}")
    print(f"  Min frequency threshold: {min_freq}")

    # Extract concepts per document
    rows = []
    global_counter = Counter()

    for idx, (_, row) in enumerate(df.iterrows()):
        concepts = extract_concepts_from_record(row, use_text=use_text)
        topic = int(topic_labels[idx]) if topic_labels is not None else -1

        for concept in concepts:
            rows.append({
                "pmid": row["pmid"],
                "topic_id": topic,
                "concept": concept,
            })
            global_counter[concept] += 1

    doc_concepts = pd.DataFrame(rows)
    print(f"  Total concept mentions: {len(doc_concepts):,}")
    print(f"  Unique concepts (before filter): {len(global_counter):,}")

    # Filter by minimum frequency
    valid_concepts = {
        c for c, count in global_counter.items() if count >= min_freq
    }

    # Cap at max_concepts (keep most frequent)
    if len(valid_concepts) > max_concepts:
        top_concepts = global_counter.most_common(max_concepts)
        valid_concepts = {c for c, _ in top_concepts}

    doc_concepts = doc_concepts[
        doc_concepts["concept"].isin(valid_concepts)
    ].reset_index(drop=True)

    print(f"  Concepts after frequency filter (>={min_freq}): "
          f"{len(valid_concepts):,}")

    # Build concept summary
    concept_freq = doc_concepts["concept"].value_counts()
    concept_doc_count = doc_concepts.groupby("concept")["pmid"].nunique()

    # Concept by topic
    if topic_labels is not None:
        topic_concept_freq = (
            doc_concepts
            .groupby(["concept", "topic_id"])
            .size()
            .reset_index(name="count")
        )
    else:
        topic_concept_freq = pd.DataFrame()

    # Concept by year
    if "year" in df.columns:
        # Merge year into doc_concepts
        pmid_year = df[["pmid", "year"]].drop_duplicates()
        dc_with_year = doc_concepts.merge(pmid_year, on="pmid", how="left")
        concept_year = (
            dc_with_year
            .groupby(["concept", "year"])
            .size()
            .reset_index(name="count")
        )
    else:
        concept_year = pd.DataFrame()

    # Summary table
    summary_rows = []
    for concept in concept_freq.index:
        freq = int(concept_freq[concept])
        n_docs = int(concept_doc_count.get(concept, 0))
        summary_rows.append({
            "concept": concept,
            "total_mentions": freq,
            "document_count": n_docs,
        })
    concept_summary = pd.DataFrame(summary_rows).sort_values(
        "total_mentions", ascending=False
    ).reset_index(drop=True)

    print(f"\n[CONCEPT] Top 20 concepts:")
    for _, r in concept_summary.head(20).iterrows():
        print(f"  {r['concept']}: {r['total_mentions']:,} mentions, "
              f"{r['document_count']:,} docs")

    return doc_concepts, concept_summary


def save_concept_outputs(
    doc_concepts: pd.DataFrame,
    concept_summary: pd.DataFrame,
) -> dict:
    """Save concept extraction outputs."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "document_concepts.csv"
    doc_concepts.to_csv(p, index=False, encoding="utf-8-sig")
    saved["doc_concepts"] = p
    print(f"[SAVE] Document concepts -> {p}  ({len(doc_concepts):,} rows)")

    p = out / "concept_summary.csv"
    concept_summary.to_csv(p, index=False, encoding="utf-8-sig")
    saved["concept_summary"] = p
    print(f"[SAVE] Concept summary -> {p}  "
          f"({len(concept_summary):,} unique concepts)")

    return saved
