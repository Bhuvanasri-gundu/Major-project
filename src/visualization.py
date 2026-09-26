"""
visualization.py — Presentation-ready figures for the project.

Generates all required plots using matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from .config_loader import load_config, ensure_dirs


def _savefig(fig, name: str) -> str:
    """Save a figure to the figures directory."""
    cfg = load_config()
    dirs = ensure_dirs()
    dpi = cfg.get("visualization", {}).get("figure_dpi", 150)
    fmt = cfg.get("visualization", {}).get("figure_format", "png")
    path = dirs["figures"] / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[FIG] Saved -> {path}")
    return str(path)


# ================================================================
# 1. Papers by Year
# ================================================================

def plot_papers_by_year(df: pd.DataFrame) -> str:
    """Bar chart of number of papers per year."""
    year_counts = df["year"].astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_counts.index, year_counts.values, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("mHealth Research Papers by Year (2015–2025)", fontsize=14)
    for i, (year, count) in enumerate(year_counts.items()):
        ax.text(year, count + 10, str(count), ha="center", fontsize=9)
    ax.set_xticks(year_counts.index)
    plt.tight_layout()
    return _savefig(fig, "papers_by_year")


# ================================================================
# 2. Top Topics
# ================================================================

def plot_topic_frequency(summary_df: pd.DataFrame, top_n: int = 25) -> str:
    """Horizontal bar chart of top BERTopic topics."""
    data = summary_df[summary_df["topic_id"] != -1].head(top_n).copy()
    data = data.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.35)))
    labels = [
        f"Topic {row['topic_id']}: {str(row['top_terms'])[:50]}"
        for _, row in data.iterrows()
    ]
    ax.barh(labels, data["count"], color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Number of Documents", fontsize=12)
    ax.set_title(f"Top {top_n} Topics by Document Count", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "topic_frequency")


# ================================================================
# 3. Topic Distribution
# ================================================================

def plot_topic_distribution(summary_df: pd.DataFrame) -> str:
    """Pie/donut chart of topic distribution."""
    data = summary_df.copy()
    # Group small topics
    threshold = 1.0  # percent
    major = data[data["percentage"] >= threshold].copy()
    minor_sum = data[data["percentage"] < threshold]["count"].sum()

    if minor_sum > 0:
        minor_row = pd.DataFrame([{
            "topic_id": "Other",
            "count": minor_sum,
            "percentage": round(100 * minor_sum / data["count"].sum(), 2),
            "top_terms": "other small topics",
        }])
        major = pd.concat([major, minor_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [
        f"T{row['topic_id']}" if row["topic_id"] != "Other" else "Other"
        for _, row in major.iterrows()
    ]
    colors = plt.cm.Set3(np.linspace(0, 1, len(major)))
    wedges, texts, autotexts = ax.pie(
        major["count"],
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        pctdistance=0.85,
    )
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("Topic Distribution", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "topic_distribution")


# ================================================================
# 4. Topic Trends Over Time
# ================================================================

def plot_topic_trends(topic_year_matrix: pd.DataFrame, top_n: int = 10) -> str:
    """Line chart of topic trends over time."""
    # Select top N topics by total count
    totals = topic_year_matrix.sum().sort_values(ascending=False)
    top_topics = totals.head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    for tid in top_topics:
        ax.plot(
            topic_year_matrix.index,
            topic_year_matrix[tid],
            marker="o",
            linewidth=2,
            label=f"Topic {tid}",
        )
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title(f"Top {top_n} Topic Trends Over Time", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    return _savefig(fig, "topic_trends")


# ================================================================
# 5. Top Biomedical Concepts
# ================================================================

def plot_top_concepts(concept_summary: pd.DataFrame, top_n: int = 30) -> str:
    """Horizontal bar chart of top biomedical concepts."""
    data = concept_summary.head(top_n).sort_values(
        "total_mentions", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.3)))
    ax.barh(data["concept"], data["total_mentions"], color="#FF9800", edgecolor="white")
    ax.set_xlabel("Total Mentions", fontsize=12)
    ax.set_title(f"Top {top_n} Biomedical Concepts", fontsize=14)
    plt.tight_layout()
    return _savefig(fig, "top_concepts")


# ================================================================
# 6. Concept Trends Over Time
# ================================================================

def plot_concept_trends(concept_year_matrix: pd.DataFrame, top_n: int = 10) -> str:
    """Line chart of concept trends over time."""
    totals = concept_year_matrix.sum().sort_values(ascending=False)
    top_concepts = totals.head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    for concept in top_concepts:
        ax.plot(
            concept_year_matrix.index,
            concept_year_matrix[concept],
            marker="o",
            linewidth=2,
            label=concept[:30],
        )
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mentions", fontsize=12)
    ax.set_title(f"Top {top_n} Concept Trends Over Time", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    return _savefig(fig, "concept_trends")


# ================================================================
# 7. Co-occurrence Heatmap
# ================================================================

def plot_cooccurrence_heatmap(
    cooccurrence_df: pd.DataFrame,
    top_n: int = 40,
) -> str:
    """Heatmap of concept co-occurrence (Jaccard similarity)."""
    # Get top concepts by co-occurrence
    all_concepts = set(cooccurrence_df["concept_a"]) | set(cooccurrence_df["concept_b"])
    concept_freq = pd.concat([
        cooccurrence_df["concept_a"],
        cooccurrence_df["concept_b"],
    ]).value_counts()
    top_concepts = concept_freq.head(top_n).index.tolist()

    # Build matrix
    filtered = cooccurrence_df[
        cooccurrence_df["concept_a"].isin(top_concepts) &
        cooccurrence_df["concept_b"].isin(top_concepts)
    ]

    matrix = pd.DataFrame(0.0, index=top_concepts, columns=top_concepts)
    for _, row in filtered.iterrows():
        ca, cb = row["concept_a"], row["concept_b"]
        val = row["normalized_cooccurrence"]
        if ca in matrix.index and cb in matrix.columns:
            matrix.loc[ca, cb] = val
            matrix.loc[cb, ca] = val

    # Fill diagonal
    mat_arr = matrix.to_numpy(copy=True)
    np.fill_diagonal(mat_arr, 1.0)
    matrix = pd.DataFrame(mat_arr, index=top_concepts, columns=top_concepts)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
        cmap="YlOrRd",
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Jaccard Similarity"},
    )
    ax.set_title("Concept Co-occurrence Heatmap (Jaccard)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    return _savefig(fig, "concept_cooccurrence_heatmap")


# ================================================================
# 8. Semantic Relationship Visualisation
# ================================================================

def plot_semantic_relationships(
    semantic_pairs: pd.DataFrame,
    top_n: int = 30,
) -> str:
    """Network-like scatter of top semantic relationships."""
    data = semantic_pairs.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (_, row) in enumerate(data.iterrows()):
        y = top_n - i
        ax.barh(
            y,
            row["semantic_similarity"],
            color=plt.cm.viridis(row["semantic_similarity"]),
            edgecolor="white",
            height=0.7,
        )
        label = f"{row['concept_a'][:20]} <-> {row['concept_b'][:20]}"
        ax.text(0.01, y, label, va="center", fontsize=8)

    ax.set_xlabel("Semantic Similarity", fontsize=12)
    ax.set_title(f"Top {top_n} Semantic Concept Relationships", fontsize=14)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return _savefig(fig, "semantic_relationships")


# ================================================================
# 9. Candidate Gap Ranking
# ================================================================

def plot_gap_ranking(gaps_df: pd.DataFrame, top_n: int = 20) -> str:
    """Bar chart of top candidate research gaps by GapScore."""
    data = gaps_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.4)))

    labels = [
        f"{row['concept_a'][:18]} <-> {row['concept_b'][:18]}"
        for _, row in data.iterrows()
    ]
    labels.reverse()

    scores = data["gap_score"].values[::-1]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(scores)))
    ax.barh(range(len(scores)), scores, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Gap Score", fontsize=12)
    ax.set_title(
        "Top Candidate Research Gaps\n"
        "(GapScore = SemanticSim x (1-CoOccurrence) x TemporalGrowth)",
        fontsize=13,
    )
    plt.tight_layout()
    return _savefig(fig, "candidate_gap_ranking")


# ================================================================
# RUN ALL
# ================================================================

def generate_all_figures(data: dict) -> list[str]:
    """Generate all required figures.

    Parameters
    ----------
    data : dict with keys matching required dataframes

    Returns
    -------
    list of saved file paths
    """
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    paths = []

    if "df" in data:
        paths.append(plot_papers_by_year(data["df"]))

    if "topic_summary" in data:
        paths.append(plot_topic_frequency(data["topic_summary"]))
        paths.append(plot_topic_distribution(data["topic_summary"]))

    if "topic_year_matrix" in data:
        paths.append(plot_topic_trends(data["topic_year_matrix"]))

    if "concept_summary" in data:
        paths.append(plot_top_concepts(data["concept_summary"]))

    if "concept_year_matrix" in data:
        paths.append(plot_concept_trends(data["concept_year_matrix"]))

    if "cooccurrence" in data:
        paths.append(plot_cooccurrence_heatmap(data["cooccurrence"]))

    if "semantic_pairs" in data:
        paths.append(plot_semantic_relationships(data["semantic_pairs"]))

    if "gaps" in data:
        paths.append(plot_gap_ranking(data["gaps"]))

    print(f"\n[FIG] Generated {len(paths)} figures")
    return paths
