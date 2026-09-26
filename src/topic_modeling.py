"""
topic_modeling.py — BERTopic topic discovery using the official BERTopic package.

Pipeline:
    SPECTER2 embeddings (precomputed, 12,642 x 768)
        |
        v
    UMAP (dimensionality reduction)
        |
        v
    HDBSCAN (density-based clustering)
        |
        v
    c-TF-IDF (BERTopic's own implementation — class-based TF-IDF)
        |
        v
    Topic assignments + top terms + summary

BERTopic is used with precomputed embeddings so SPECTER2 does NOT run again.
The embedding_model=None argument tells BERTopic to skip its own embedding step.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

from .config_loader import load_config, ensure_dirs, set_seed


# ================================================================
# MAIN PIPELINE
# ================================================================

def run_topic_modeling(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_column: str = "clean_document",
) -> dict:
    """Run the official BERTopic pipeline on precomputed SPECTER2 embeddings.

    Parameters
    ----------
    df : preprocessed dataframe (12,642 rows) with pmid, year, title columns
    embeddings : numpy array of shape (12,642, 768) — SPECTER2 embeddings
    text_column : column in df to use as document text for c-TF-IDF

    Returns
    -------
    dict with all results and saved file paths
    """
    dirs = ensure_dirs()
    out = dirs["outputs"]
    cfg = load_config()
    bt_cfg = cfg["bertopic"]
    umap_cfg = bt_cfg["umap"]
    hdb_cfg = bt_cfg["hdbscan"]
    seed = cfg.get("random_seed", 42)

    set_seed(seed)

    docs = df[text_column].fillna("").tolist()

    print("\n" + "=" * 60)
    print("OFFICIAL BERTopic TOPIC MODELING")
    print("=" * 60)
    print(f"\n  Documents   : {len(docs):,}")
    print(f"  Embeddings  : {embeddings.shape}")
    print(f"  UMAP        : n_components={umap_cfg['n_components']}  "
          f"n_neighbors={umap_cfg['n_neighbors']}  metric={umap_cfg['metric']}")
    print(f"  HDBSCAN     : min_cluster_size={hdb_cfg['min_cluster_size']}  "
          f"min_samples={hdb_cfg['min_samples']}")
    print(f"  n-gram      : {tuple(bt_cfg.get('n_gram_range', [1, 3]))}")
    print(f"  Random seed : {seed}")

    # ----------------------------------------------------------------
    # Configure UMAP, HDBSCAN, and Vectorizer — passed into BERTopic
    # ----------------------------------------------------------------
    umap_model = UMAP(
        n_components=umap_cfg["n_components"],
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg.get("min_dist", 0.0),
        metric=umap_cfg["metric"],
        random_state=seed,
        verbose=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        metric=hdb_cfg.get("metric", "euclidean"),
        prediction_data=hdb_cfg.get("prediction_data", True),
    )

    ngram_range = tuple(bt_cfg.get("n_gram_range", [1, 3]))
    vectorizer_model = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        max_features=10000,
        min_df=2,
    )

    # ----------------------------------------------------------------
    # BERTopic — embedding_model=None because we pass precomputed embeddings
    # ----------------------------------------------------------------
    topic_model = BERTopic(
        embedding_model=None,          # Use our precomputed SPECTER2 embeddings
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=bt_cfg.get("top_n_words", 15),
        calculate_probabilities=False,  # Faster; not needed for gap detection
        verbose=True,
    )

    print("\n[BERTopic] Fitting model on precomputed embeddings...")
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    # topics is a list of integer topic IDs (−1 = outlier)
    labels = np.array(topics)

    n_topics = len([t for t in topic_model.get_topics() if t != -1])
    n_outliers = int((labels == -1).sum())

    print(f"\n[BERTopic] === Topic Discovery Results ===")
    print(f"  Topics found         : {n_topics}")
    print(f"  Outlier docs (id=-1) : {n_outliers:,}  ({100*n_outliers/len(labels):.1f}%)")
    print(f"  Docs in topics       : {len(labels) - n_outliers:,}")

    # ----------------------------------------------------------------
    # Extract topic information from BERTopic's own data structures
    # ----------------------------------------------------------------
    topic_info = topic_model.get_topic_info()

    # Top words per topic
    tw_rows = []
    for tid in sorted(set(labels)):
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)  # list of (word, score)
        if not words:
            continue
        for rank, (word, score) in enumerate(words):
            tw_rows.append({
                "topic_id": int(tid),
                "rank": rank + 1,
                "term": word,
                "ctfidf_score": round(float(score), 6),
            })
    tw_df = pd.DataFrame(tw_rows)

    # ----------------------------------------------------------------
    # Build assignments dataframe (same schema as before)
    # ----------------------------------------------------------------
    assignments = df[["pmid"]].copy()
    assignments["topic_id"] = labels
    if "year" in df.columns:
        assignments["year"] = df["year"].values
    title_col = "clean_title" if "clean_title" in df.columns else "title"
    if title_col in df.columns:
        assignments["title"] = df[title_col].values

    # ----------------------------------------------------------------
    # Build topic summary (same schema as before)
    # ----------------------------------------------------------------
    label_counts = Counter(labels)
    summary_rows = []

    for tid in sorted(t for t in set(labels) if t != -1):
        words = topic_model.get_topic(tid) or []
        top_terms = ", ".join([w for w, _ in words[:10]])
        count = label_counts.get(tid, 0)

        # Representative titles (first 3 docs in this topic)
        topic_docs = df[labels == tid]
        rep_titles = "; ".join(
            topic_docs[title_col].fillna("").head(3).tolist()
        )

        summary_rows.append({
            "topic_id": int(tid),
            "count": count,
            "percentage": round(100 * count / len(df), 2),
            "top_terms": top_terms,
            "representative_titles": rep_titles,
        })

    # Outlier row
    n_out = label_counts.get(-1, 0)
    summary_rows.append({
        "topic_id": -1,
        "count": n_out,
        "percentage": round(100 * n_out / len(df), 2),
        "top_terms": "(outlier / unassigned)",
        "representative_titles": "",
    })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "count", ascending=False
    ).reset_index(drop=True)

    # ----------------------------------------------------------------
    # Topic-year frequency matrix
    # ----------------------------------------------------------------
    year_trends = _build_topic_year_matrix(assignments)

    # ----------------------------------------------------------------
    # Print top 10 topics
    # ----------------------------------------------------------------
    print(f"\n  Top 10 topics by document count:")
    for _, row in summary_df[summary_df["topic_id"] != -1].head(10).iterrows():
        print(f"    Topic {int(row['topic_id']):3d}: {int(row['count']):4d} docs"
              f" — {str(row['top_terms'])[:70]}")

    # ----------------------------------------------------------------
    # Save all outputs
    # ----------------------------------------------------------------
    p = out / "topic_assignments.csv"
    assignments.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] Topic assignments -> {p}  ({len(assignments):,} rows)")

    p = out / "topic_summary.csv"
    summary_df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Topic summary     -> {p}  ({len(summary_df)} topics)")

    p = out / "topic_top_terms.csv"
    tw_df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Topic top terms   -> {p}  ({len(tw_df)} rows)")

    p = out / "topic_year_matrix.csv"
    year_trends.to_csv(p, encoding="utf-8-sig")
    print(f"[SAVE] Topic-year matrix -> {p}  {year_trends.shape}")

    # Save the trained official BERTopic model
    models_dir = dirs.get("models", out.parent / "models")
    model_save_path = models_dir / "bertopic_model"
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Ensure vocabulary indices are standard Python ints for JSON serialization
    if hasattr(topic_model, "vectorizer_model") and hasattr(topic_model.vectorizer_model, "vocabulary_"):
        if topic_model.vectorizer_model.vocabulary_:
            topic_model.vectorizer_model.vocabulary_ = {
                k: int(v) for k, v in topic_model.vectorizer_model.vocabulary_.items()
            }

    topic_model.save(str(model_save_path), serialization="safetensors", save_ctfidf=True)
    print(f"[SAVE] BERTopic model    -> {model_save_path}")

    return {
        "topic_model": topic_model,
        "labels": labels,
        "assignments": assignments,
        "summary": summary_df,
        "top_terms_df": tw_df,
        "year_trends": year_trends,
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "model_path": model_save_path,
    }


def load_saved_bertopic_model(model_path: str | Path | None = None) -> BERTopic:
    """Load a previously saved official BERTopic model from disk."""
    if model_path is None:
        dirs = ensure_dirs()
        model_path = dirs.get("models", dirs["outputs"].parent / "models") / "bertopic_model"
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Saved BERTopic model not found at {model_path}")
    print(f"[LOAD] Loading BERTopic model from: {model_path}")
    return BERTopic.load(str(model_path))


# ================================================================
# HELPER
# ================================================================

def _build_topic_year_matrix(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build year x topic_id frequency pivot table (genuine topics only)."""
    cfg = load_config()
    year_range = cfg["temporal"]["year_range"]

    if "year" not in assignments.columns:
        return pd.DataFrame()

    valid = assignments[assignments["topic_id"] != -1].copy()
    valid["year"] = valid["year"].astype(int)

    pivot = pd.crosstab(valid["year"], valid["topic_id"])

    # Ensure all years in range are present as index rows
    for y in range(year_range[0], year_range[1] + 1):
        if y not in pivot.index:
            pivot.loc[y] = 0

    return pivot.sort_index()
