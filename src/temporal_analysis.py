"""
temporal_analysis.py — Temporal research trend analysis.

Computes frequency-by-year matrices for topics and concepts,
identifies emerging and declining trends.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from .config_loader import load_config, ensure_dirs


# ================================================================
# TREND COMPUTATION
# ================================================================

def compute_yearly_growth(series: pd.Series, window: int = 3) -> float:
    """Compute average yearly growth rate over a recent window.

    Uses simple linear slope normalised by mean.
    Returns growth rate: positive = increasing, negative = declining.
    """
    if len(series) < 2:
        return 0.0

    recent = series.tail(window)
    if recent.sum() == 0:
        return 0.0

    # Simple linear regression slope
    x = np.arange(len(recent), dtype=float)
    y = recent.values.astype(float)

    mean_x = x.mean()
    mean_y = y.mean()
    if mean_y == 0:
        return 0.0

    numerator = ((x - mean_x) * (y - mean_y)).sum()
    denominator = ((x - mean_x) ** 2).sum()

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    # Normalise by mean to get relative growth
    return slope / mean_y


def build_topic_year_matrix(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Build topic x year frequency matrix.

    Parameters
    ----------
    assignments : topic_assignments.csv with columns [pmid, topic_id, year]

    Returns
    -------
    pivot table with years as rows, topic_ids as columns
    """
    cfg = load_config()
    year_range = cfg["temporal"]["year_range"]

    valid = assignments[
        (assignments["topic_id"] != -1) &
        (assignments["year"].notna())
    ].copy()
    valid["year"] = valid["year"].astype(int)

    pivot = pd.crosstab(valid["year"], valid["topic_id"])

    # Ensure all years in range are present
    all_years = range(year_range[0], year_range[1] + 1)
    for y in all_years:
        if y not in pivot.index:
            pivot.loc[y] = 0
    pivot = pivot.sort_index()

    return pivot


def build_concept_year_matrix(
    doc_concepts: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 300,
) -> pd.DataFrame:
    """Build concept x year frequency matrix.

    Parameters
    ----------
    doc_concepts : document_concepts or mesh_normalized dataframe
    df : main dataframe with pmid and year columns
    top_n : maximum number of concepts to include

    Returns
    -------
    pivot table with years as rows, concepts as columns
    """
    cfg = load_config()

    # Determine concept column
    concept_col = (
        "normalized_concept"
        if "normalized_concept" in doc_concepts.columns
        else "concept"
    )

    # Get top N concepts by frequency
    top_concepts = (
        doc_concepts[concept_col]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )

    # Merge year
    pmid_year = df[["pmid", "year"]].drop_duplicates()
    merged = doc_concepts.merge(pmid_year, on="pmid", how="left")
    merged = merged[merged[concept_col].isin(top_concepts)]
    merged = merged[merged["year"].notna()]
    merged["year"] = merged["year"].astype(int)

    pivot = pd.crosstab(merged["year"], merged[concept_col])
    pivot = pivot.sort_index()

    return pivot


# ================================================================
# TREND ANALYSIS
# ================================================================

def analyze_trends(
    topic_year_matrix: pd.DataFrame,
    concept_year_matrix: pd.DataFrame,
) -> dict:
    """Analyze temporal trends for topics and concepts.

    Returns dict with:
      - topic_trends: DataFrame of topic growth stats
      - concept_trends: DataFrame of concept growth stats
      - emerging_concepts: list of concepts with high recent growth
    """
    cfg = load_config()
    tcfg = cfg["temporal"]
    recent_window = tcfg.get("recent_window", 3)
    growth_window = tcfg.get("growth_window", 5)

    print("\n[TEMPORAL] Analyzing research trends...")

    # --- Topic trends ---
    topic_trends = []
    for tid in topic_year_matrix.columns:
        series = topic_year_matrix[tid]
        total = int(series.sum())
        recent = int(series.tail(recent_window).sum())
        growth = compute_yearly_growth(series, window=growth_window)

        topic_trends.append({
            "topic_id": tid,
            "total_papers": total,
            "recent_papers": recent,
            "growth_rate": round(growth, 4),
            "trend": (
                "increasing" if growth > 0.05
                else "declining" if growth < -0.05
                else "stable"
            ),
        })

    topic_trends_df = pd.DataFrame(topic_trends).sort_values(
        "growth_rate", ascending=False
    ).reset_index(drop=True)

    # --- Concept trends ---
    concept_trends = []
    for concept in concept_year_matrix.columns:
        series = concept_year_matrix[concept]
        total = int(series.sum())
        recent = int(series.tail(recent_window).sum())
        growth = compute_yearly_growth(series, window=growth_window)

        concept_trends.append({
            "concept": concept,
            "total_mentions": total,
            "recent_mentions": recent,
            "growth_rate": round(growth, 4),
            "trend": (
                "increasing" if growth > 0.05
                else "declining" if growth < -0.05
                else "stable"
            ),
        })

    concept_trends_df = pd.DataFrame(concept_trends).sort_values(
        "growth_rate", ascending=False
    ).reset_index(drop=True)

    # --- Emerging concepts ---
    emerging = concept_trends_df[
        (concept_trends_df["trend"] == "increasing") &
        (concept_trends_df["recent_mentions"] >= tcfg.get("min_yearly_count", 2))
    ].head(50)

    print(f"\n  Topics analyzed: {len(topic_trends_df)}")
    print(f"  Concepts analyzed: {len(concept_trends_df)}")
    print(f"  Emerging concepts: {len(emerging)}")

    print("\n  Top 10 increasing topics:")
    for _, r in topic_trends_df.head(10).iterrows():
        print(f"    Topic {r['topic_id']}: growth={r['growth_rate']:.3f}, "
              f"recent={r['recent_papers']}")

    print("\n  Top 10 emerging concepts:")
    for _, r in emerging.head(10).iterrows():
        print(f"    {r['concept']}: growth={r['growth_rate']:.3f}, "
              f"recent={r['recent_mentions']}")

    return {
        "topic_trends": topic_trends_df,
        "concept_trends": concept_trends_df,
        "emerging_concepts": emerging,
    }


def save_temporal_outputs(
    topic_year_matrix: pd.DataFrame,
    concept_year_matrix: pd.DataFrame,
    trend_results: dict,
) -> dict:
    """Save temporal analysis outputs."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "topic_year_matrix.csv"
    topic_year_matrix.to_csv(p, encoding="utf-8-sig")
    saved["topic_year_matrix"] = p
    print(f"[SAVE] Topic-year matrix -> {p}")

    p = out / "concept_year_matrix.csv"
    concept_year_matrix.to_csv(p, encoding="utf-8-sig")
    saved["concept_year_matrix"] = p
    print(f"[SAVE] Concept-year matrix -> {p}")

    p = out / "topic_trends.csv"
    trend_results["topic_trends"].to_csv(p, index=False, encoding="utf-8-sig")
    saved["topic_trends"] = p

    p = out / "concept_trends.csv"
    trend_results["concept_trends"].to_csv(p, index=False, encoding="utf-8-sig")
    saved["concept_trends"] = p

    p = out / "emerging_concepts.csv"
    trend_results["emerging_concepts"].to_csv(p, index=False, encoding="utf-8-sig")
    saved["emerging_concepts"] = p

    print(f"[SAVE] Trend analysis outputs saved")
    return saved
