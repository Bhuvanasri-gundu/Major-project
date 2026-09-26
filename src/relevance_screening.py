"""
relevance_screening.py — mHealth relevance screening.

Uses configurable keyword/phrase matching on title, abstract, and document
to classify records as relevant / borderline / low_relevance.
"""

import re
import pandas as pd
import numpy as np

from .config_loader import load_config, ensure_dirs


def _count_term_matches(text: str, terms: list[str]) -> int:
    """Count distinct term matches in text (case-insensitive)."""
    if not isinstance(text, str) or not text.strip():
        return 0
    text_lower = text.lower()
    count = 0
    for term in terms:
        # Use word-boundary-aware matching
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, text_lower):
            count += 1
    return count


def compute_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relevance scores for each record.

    Returns a copy of df with new columns:
      - relevance_score (float 0-1)
      - relevance_label (relevant / borderline / low_relevance)
      - relevance_reason (human-readable explanation)
    """
    cfg = load_config()
    scr = cfg["relevance_screening"]

    primary = scr["primary_terms"]
    secondary = scr["secondary_terms"]
    pw = scr["primary_weight"]
    sw = scr["secondary_weight"]
    tw = scr["title_weight"]
    aw = scr["abstract_weight"]
    rel_thresh = scr["relevant_threshold"]
    bord_thresh = scr["borderline_threshold"]

    # Maximum possible score for normalisation
    max_primary = len(primary) * pw
    max_secondary = len(secondary) * sw
    max_per_field = max_primary + max_secondary
    max_total = max_per_field * tw + max_per_field * aw
    if max_total == 0:
        max_total = 1.0

    result = df.copy()

    scores = []
    labels = []
    reasons = []

    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))

        # Primary matches
        tp = _count_term_matches(title, primary)
        ap = _count_term_matches(abstract, primary)

        # Secondary matches
        ts = _count_term_matches(title, secondary)
        as_ = _count_term_matches(abstract, secondary)

        # Relevance logic:
        # If title or abstract has ANY primary term -> relevant (score 0.7 - 1.0 based on count)
        # If only secondary terms -> borderline (score 0.3 - 0.6)
        # If no terms matched -> low_relevance (score 0.0)
        
        has_primary = (tp > 0) or (ap > 0)
        has_secondary = (ts > 0) or (as_ > 0)
        
        primary_count = tp * 2 + ap  # title matches carry 2x weight
        secondary_count = ts * 2 + as_
        
        if has_primary:
            # Score scaled between 0.5 and 1.0
            norm_score = min(0.5 + 0.1 * primary_count + 0.02 * secondary_count, 1.0)
            label = "relevant"
        elif has_secondary:
            # Score scaled between 0.1 and 0.49
            norm_score = min(0.1 + 0.05 * secondary_count, 0.49)
            label = "borderline"
        else:
            norm_score = 0.0
            label = "low_relevance"

        reason_parts = []
        text_combined = (title + " " + abstract).lower()
        matched_primary = [t for t in primary if re.search(r"\b" + re.escape(t.lower()) + r"\b", text_combined)]
        matched_secondary = [t for t in secondary if re.search(r"\b" + re.escape(t.lower()) + r"\b", text_combined)]
        if matched_primary:
            reason_parts.append(f"primary: {', '.join(matched_primary[:5])}")
        if matched_secondary:
            reason_parts.append(f"secondary: {', '.join(matched_secondary[:5])}")

        if not reason_parts:
            reason = "no mHealth keyword matches"
        else:
            reason = "; ".join(reason_parts)

        scores.append(round(norm_score, 4))
        labels.append(label)
        reasons.append(reason)

    result["relevance_score"] = scores
    result["relevance_label"] = labels
    result["relevance_reason"] = reasons

    return result


def screen_corpus(
    df: pd.DataFrame,
    keep_borderline: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen the corpus for mHealth relevance.

    Parameters
    ----------
    df : cleaned dataset
    keep_borderline : if True, include borderline records in screened corpus

    Returns
    -------
    screened_df : filtered corpus (relevant + optionally borderline)
    stats_df : screening statistics
    """
    print("\n[SCREEN] Computing relevance scores...")
    scored = compute_relevance_scores(df)

    # Statistics
    label_counts = scored["relevance_label"].value_counts()
    total = len(scored)

    stats_rows = []
    for label in ["relevant", "borderline", "low_relevance"]:
        count = label_counts.get(label, 0)
        stats_rows.append({
            "label": label,
            "count": count,
            "percentage": round(100 * count / total, 2),
        })
    stats_rows.append({"label": "total", "count": total, "percentage": 100.0})

    stats_df = pd.DataFrame(stats_rows)

    print("\n[SCREEN] Relevance screening results:")
    for _, row in stats_df.iterrows():
        print(f"  {row['label']}: {row['count']:,} ({row['percentage']:.1f}%)")

    # Filter
    if keep_borderline:
        mask = scored["relevance_label"].isin(["relevant", "borderline"])
        action = "Keeping relevant + borderline"
    else:
        mask = scored["relevance_label"] == "relevant"
        action = "Keeping relevant only"

    screened = scored[mask].reset_index(drop=True)
    removed = total - len(screened)

    print(f"\n[SCREEN] {action}")
    print(f"  Before: {total:,}  Removed: {removed:,}  After: {len(screened):,}")

    # Show examples
    print("\n--- Example RELEVANT articles ---")
    rel = scored[scored["relevance_label"] == "relevant"]
    if len(rel) > 0:
        for _, r in rel.head(3).iterrows():
            print(f"  PMID {r['pmid']}: {str(r['title'])[:100]}")
            print(f"    Score: {r['relevance_score']}, Reason: {r['relevance_reason']}")

    print("\n--- Example BORDERLINE articles ---")
    bord = scored[scored["relevance_label"] == "borderline"]
    if len(bord) > 0:
        for _, r in bord.head(3).iterrows():
            print(f"  PMID {r['pmid']}: {str(r['title'])[:100]}")
            print(f"    Score: {r['relevance_score']}, Reason: {r['relevance_reason']}")

    print("\n--- Example LOW-RELEVANCE articles ---")
    low = scored[scored["relevance_label"] == "low_relevance"]
    if len(low) > 0:
        for _, r in low.head(3).iterrows():
            print(f"  PMID {r['pmid']}: {str(r['title'])[:100]}")
            print(f"    Score: {r['relevance_score']}, Reason: {r['relevance_reason']}")

    return screened, stats_df


def save_screening_outputs(
    screened_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict:
    """Save screened corpus and statistics."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "mhealth_screened_corpus.csv"
    screened_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["screened_corpus"] = p
    print(f"[SAVE] Screened corpus -> {p}  ({len(screened_df):,} records)")

    p = out / "screening_statistics.csv"
    stats_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["screening_stats"] = p
    print(f"[SAVE] Screening statistics -> {p}")

    return saved
