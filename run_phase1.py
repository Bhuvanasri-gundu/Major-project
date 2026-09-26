"""
run_phase1.py — Master pipeline runner for Phase 1.
Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research.

Executes all Phase 1 stages in sequence:
    1. Data Validation (raw -> 14,881 records)
    2. Rule-Based Relevance Screening (14,881 -> 12,642 screened corpus)
    3. Text Preprocessing (clean fields)
    4. SPECTER2 Document Embeddings (12,642 x 768 dense vectors)
    5. Official BERTopic Topic Modeling (63 topics + outlier class, saved to models/bertopic_model/)
    6. Biomedical Concept Extraction (MeSH, keywords, noun phrases)
    7. MeSH Normalization & Topic-Concept Mapping
    8. Temporal Trend Analysis (2015-2025 growth rates)
    9. Concept Quality Filtering, Centroid Debiasing, Co-occurrence, and Gap Scoring
    10. Publication Figure Generation (9 figures)

Intermediate outputs are detected and reused when valid, avoiding redundant heavy computations.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config, set_seed, ensure_dirs
from src.data_validation import (
    load_dataset, validate_dataset, print_validation_report,
    clean_dataset, save_validation_outputs,
)
from src.relevance_screening import screen_corpus, save_screening_outputs
from src.text_preprocessing import preprocess_corpus, save_preprocessed
from src.embeddings import generate_embeddings
from src.topic_modeling import run_topic_modeling
from src.concept_extraction import extract_all_concepts, save_concept_outputs
from src.mesh_normalization import (
    normalize_mesh_terms, build_topic_concept_mapping, save_mesh_outputs,
)
from src.temporal_analysis import (
    build_topic_year_matrix, build_concept_year_matrix,
    analyze_trends, save_temporal_outputs,
)
from src.concept_quality_filter import (
    filter_concepts, compute_debiased_concept_embeddings,
    compute_debiased_semantic_similarity, save_quality_filter_outputs,
    deduplicate_synonym_pairs,
)
from src.gap_detection import (
    compute_cooccurrence, detect_candidate_gaps,
    prepare_gap_evidence, save_gap_outputs,
)
from src.visualization import generate_all_figures


def main():
    start_time = time.time()

    print("=" * 70)
    print("PHASE 1 MASTER PIPELINE")
    print("Temporal-Semantic Gap-Guided Evidence-Grounded")
    print("Hypothesis Generation in mHealth Research")
    print("=" * 70)

    cfg = load_config()
    dirs = ensure_dirs()
    out = dirs["outputs"]
    set_seed()

    # ============================================================
    # STEP 1: DATA VALIDATION
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA VALIDATION")
    print("=" * 70)

    cleaned_file = out / "mhealth_candidate_cleaned.csv"
    if cleaned_file.exists():
        print(f"[REUSE] Found existing cleaned dataset: {cleaned_file}")
        df_cleaned = pd.read_csv(cleaned_file, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(df_cleaned):,} validated records")
        raw_count = 14905
    else:
        df_raw = load_dataset()
        raw_count = len(df_raw)
        stats = validate_dataset(df_raw)
        print_validation_report(stats)
        df_cleaned, filter_log = clean_dataset(df_raw)
        save_validation_outputs(df_raw, stats, filter_log, df_cleaned)

    # ============================================================
    # STEP 2: RELEVANCE SCREENING
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: RELEVANCE SCREENING")
    print("=" * 70)

    screened_file = out / "mhealth_screened_corpus.csv"
    if screened_file.exists():
        print(f"[REUSE] Found existing screened corpus: {screened_file}")
        df_screened = pd.read_csv(screened_file, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(df_screened):,} screened records")
    else:
        df_screened, screening_stats = screen_corpus(df_cleaned, keep_borderline=True)
        save_screening_outputs(df_screened, screening_stats)

    # ============================================================
    # STEP 3: TEXT PREPROCESSING
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: TEXT PREPROCESSING")
    print("=" * 70)

    preprocessed_file = out / "mhealth_preprocessed.csv"
    if preprocessed_file.exists():
        print(f"[REUSE] Found existing preprocessed corpus: {preprocessed_file}")
        df_processed = pd.read_csv(preprocessed_file, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(df_processed):,} preprocessed records")
    else:
        df_processed = preprocess_corpus(df_screened)
        save_preprocessed(df_processed)

    # ============================================================
    # STEP 4: DOCUMENT EMBEDDINGS (SPECTER2)
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: DOCUMENT EMBEDDINGS (SPECTER2)")
    print("=" * 70)

    embeddings, emb_index = generate_embeddings(
        df_processed,
        text_column="clean_document",
        resume=True,
    )
    print(f"  Embeddings: {embeddings.shape}, device: cuda/cpu")

    # ============================================================
    # STEP 5: OFFICIAL BERTopic TOPIC MODELING
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 5: OFFICIAL BERTopic TOPIC MODELING")
    print("=" * 70)

    topic_assign_file = out / "topic_assignments.csv"
    topic_summary_file = out / "topic_summary.csv"
    model_dir = dirs["models"] / "bertopic_model"
    if topic_assign_file.exists() and topic_summary_file.exists() and model_dir.exists():
        print(f"[REUSE] Found existing topic modeling outputs: {topic_assign_file}")
        topic_assignments = pd.read_csv(topic_assign_file)
        topic_summary = pd.read_csv(topic_summary_file)
        labels = topic_assignments["topic_id"].values
        n_topics = len(topic_summary[topic_summary["topic_id"] != -1])
        n_outliers = int((topic_assignments["topic_id"] == -1).sum())
        print(f"  Loaded {len(topic_assignments):,} topic assignments ({n_topics} topics, {n_outliers:,} outliers)")
    else:
        topic_results = run_topic_modeling(
            df_processed,
            embeddings,
            text_column="clean_document",
        )
        topic_assignments = topic_results["assignments"]
        topic_summary = topic_results["summary"]
        labels = topic_results["labels"]
        n_topics = topic_results["n_topics"]
        n_outliers = topic_results["n_outliers"]

    # ============================================================
    # STEP 6: CONCEPT EXTRACTION
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 6: CONCEPT EXTRACTION")
    print("=" * 70)

    doc_concepts, concept_summary = extract_all_concepts(
        df_processed,
        topic_labels=labels,
    )
    save_concept_outputs(doc_concepts, concept_summary)

    # ============================================================
    # STEP 7: MeSH NORMALIZATION & MAPPING
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 7: MeSH NORMALIZATION & MAPPING")
    print("=" * 70)

    normalized_concepts = normalize_mesh_terms(doc_concepts)
    topic_concept_map = build_topic_concept_mapping(normalized_concepts)
    save_mesh_outputs(normalized_concepts, topic_concept_map)

    concept_summary_norm = (
        normalized_concepts.groupby("normalized_concept")["pmid"]
        .nunique()
        .reset_index()
        .rename(columns={"normalized_concept": "concept", "pmid": "document_count"})
        .assign(total_mentions=lambda df: df["document_count"])
        .sort_values("document_count", ascending=False)
        .reset_index(drop=True)
    )

    # ============================================================
    # STEP 8: TEMPORAL ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 8: TEMPORAL ANALYSIS")
    print("=" * 70)

    topic_year_matrix = build_topic_year_matrix(topic_assignments)
    concept_year_matrix = build_concept_year_matrix(normalized_concepts, df_processed)
    trend_results = analyze_trends(topic_year_matrix, concept_year_matrix)
    save_temporal_outputs(topic_year_matrix, concept_year_matrix, trend_results)
    concept_trends = trend_results["concept_trends"]

    # ============================================================
    # STEP 9: QUALITY FILTERING, CENTROID DEBIASING & GAP DETECTION
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 9: QUALITY FILTERING, CENTROID DEBIASING & GAP DETECTION")
    print("=" * 70)

    # 1. Quality filter
    filtered_summary, filtered_doc_concepts, filter_report = filter_concepts(
        concept_summary_norm,
        normalized_concepts,
        concept_col="normalized_concept",
        min_doc_count=20,
        max_doc_fraction=0.60,
    )

    # 2. Centroid debiasing
    concept_embs, corpus_mean = compute_debiased_concept_embeddings(
        filtered_summary,
        filtered_doc_concepts,
        embeddings,
        emb_index,
        concept_col="normalized_concept",
    )

    # 3. Debiased pairwise similarity
    semantic_pairs = compute_debiased_semantic_similarity(
        concept_embs,
        top_n_pairs=cfg.get("semantic", {}).get("top_n_pairs", 5000),
        min_similarity=cfg.get("semantic", {}).get("similarity_threshold", 0.30),
    )

    # 4. Co-occurrence
    fd = filtered_doc_concepts.copy()
    fd["concept"] = fd["normalized_concept"]
    cooccurrence_df = compute_cooccurrence(
        fd, top_n_concepts=cfg.get("cooccurrence", {}).get("top_n_concepts", 600)
    )

    # 5. Gap detection
    ct_filtered = concept_trends[
        concept_trends["concept"].isin(filtered_summary["concept"].tolist())
    ].reset_index(drop=True)

    gaps_df = detect_candidate_gaps(
        semantic_pairs=semantic_pairs,
        cooccurrence_df=cooccurrence_df,
        concept_trends=ct_filtered,
        doc_concepts=fd,
        topic_assignments=topic_assignments,
    )

    evidence_df = prepare_gap_evidence(gaps_df, df_processed, ct_filtered, topic_summary)
    gaps_df = deduplicate_synonym_pairs(gaps_df, sim_threshold=0.90)

    # Save all gap outputs including gap_quality_audit.csv
    save_quality_filter_outputs(filtered_summary, filter_report, semantic_pairs)
    save_gap_outputs(semantic_pairs, cooccurrence_df, gaps_df, evidence_df)

    # ============================================================
    # STEP 10: FIGURES
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 10: GENERATING FIGURES")
    print("=" * 70)

    fig_data = {
        "df": df_processed,
        "topic_summary": topic_summary,
        "topic_year_matrix": topic_year_matrix,
        "concept_summary": concept_summary_norm,
        "concept_year_matrix": concept_year_matrix,
        "cooccurrence": cooccurrence_df,
        "semantic_pairs": semantic_pairs,
        "gaps": gaps_df,
    }
    figure_paths = generate_all_figures(fig_data)

    # ============================================================
    # PHASE 1 SUMMARY
    # ============================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    genuine_gaps = gaps_df[
        gaps_df.get("gap_note", pd.Series([""] * len(gaps_df), index=gaps_df.index)) != "potential_synonym_pair"
    ]

    summary = {
        "initial_records": raw_count,
        "cleaned_records": len(df_cleaned),
        "screened_records": len(df_screened),
        "preprocessed_records": len(df_processed),
        "embedding_dimensions": int(embeddings.shape[1]),
        "num_topics": n_topics,
        "num_outliers": n_outliers,
        "outlier_percentage": round(100 * n_outliers / len(df_processed), 1),
        "num_concepts_raw": len(concept_summary),
        "num_concepts_normalized": len(concept_summary_norm),
        "num_concepts_quality_filtered": len(filtered_summary),
        "num_semantic_pairs": len(semantic_pairs),
        "num_cooccurrence_pairs": len(cooccurrence_df),
        "num_candidate_gaps_total": len(gaps_df),
        "num_candidate_gaps_genuine": len(genuine_gaps),
        "num_candidate_gaps_synonym_flagged": len(gaps_df) - len(genuine_gaps),
        "num_figures": len(figure_paths),
        "processing_time_seconds": round(elapsed, 1),
        "processing_time_minutes": round(elapsed / 60, 1),
    }

    if len(genuine_gaps) > 0:
        top_gaps = []
        for _, g in genuine_gaps.head(10).iterrows():
            top_gaps.append({
                "concept_a": g["concept_a"],
                "concept_b": g["concept_b"],
                "gap_score": round(float(g["gap_score"]), 4),
            })
        summary["top_10_genuine_gaps"] = top_gaps

    for k, v in summary.items():
        if k != "top_10_genuine_gaps":
            print(f"  {k}: {v}")

    # Save summary
    summary_csv_rows = [{"metric": k, "value": str(v)} for k, v in summary.items() if k != "top_10_genuine_gaps"]
    pd.DataFrame(summary_csv_rows).to_csv(out / "phase1_summary.csv", index=False, encoding="utf-8-sig")

    with open(out / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[DONE] Phase 1 completed in {elapsed / 60:.1f} minutes")
    print(f"[DONE] Outputs saved to: {out}")
    print(f"[DONE] Model saved to: {dirs.get('models', out.parent / 'models') / 'bertopic_model'}")

    print("\n--- Phase 2 Input Files ---")
    print(f"  {out / 'candidate_research_gaps.csv'}")
    print(f"  {out / 'gap_evidence.csv'}")
    print(f"  {out / 'gap_quality_audit.csv'}")


if __name__ == "__main__":
    main()
