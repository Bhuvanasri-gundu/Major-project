"""
run_evaluation.py — Master Execution Suite for Phase 3 Evaluation & Benchmarking.

Executes all evaluation experiments:
1. Retrieval Evaluation (P@K, R@K, MRR, nDCG@K against PubMed MeSH ground truth)
2. Evidence Grounding Evaluation (validity, traceability, coverage, NLI distributions)
3. Baseline Comparisons (GapScore vs Co-occurrence, Diversity vs Standard, RAG vs Direct)
4. Ablation Study (Full vs -RAG vs -NLI vs -Judge vs -Temporal)
5. Saves structured evaluation reports to outputs/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import ensure_dirs, get_project_root, load_config
from src.phase3.ablation_study import run_ablation_experiments
from src.phase3.baseline_comparison import (
    compare_gap_ranking_baselines,
    compare_generation_baselines,
    compare_retrieval_diversity,
)
from src.phase3.grounding_eval import evaluate_grounding
from src.phase3.retrieval_eval import evaluate_retrieval_suite

logger = logging.getLogger("Phase3Evaluation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_full_evaluation():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 3: EVALUATION & COMPARATIVE EXPERIMENTS")
    logger.info("=" * 70)

    dirs = ensure_dirs()
    outputs_dir = dirs["outputs"]

    # 1. Load Data
    logger.info("Loading Phase 1 & Phase 2 data...")
    gaps_csv = outputs_dir / "candidate_research_gaps.csv"
    cooc_csv = outputs_dir / "concept_cooccurrence.csv"
    doc_concepts_csv = outputs_dir / "document_concepts.csv"

    if not doc_concepts_csv.exists() or not gaps_csv.exists():
        logger.error("Required Phase 1 artifacts missing. Run Phase 1 first.")
        return 1

    gaps_df = pd.read_csv(gaps_csv)
    cooc_df = pd.read_csv(cooc_csv) if cooc_csv.exists() else pd.DataFrame()
    doc_concepts_df = pd.read_csv(doc_concepts_csv)

    # Phase 2 JSON outputs
    prep_gaps_file = outputs_dir / "phase2_prepared_gaps.json"
    ev_file = outputs_dir / "retrieved_gap_evidence.json"
    hyp_file = outputs_dir / "generated_hypotheses.json"
    verif_file = outputs_dir / "verification_results.json"
    nli_file = outputs_dir / "nli_results.json"
    judge_file = outputs_dir / "judge_results.json"
    rank_file = outputs_dir / "hypothesis_rankings.json"

    def _load_j(p):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    prepared_gaps = _load_j(prep_gaps_file)
    retrieved_ev = _load_j(ev_file)
    hypotheses = _load_j(hyp_file)
    verifications = _load_j(verif_file)
    nli_results = _load_j(nli_file)
    judge_results = _load_j(judge_file)
    rankings = _load_j(rank_file)

    logger.info(
        "Loaded: %d gaps, %d hypotheses, %d verification records, %d rankings.",
        len(prepared_gaps),
        len(hypotheses),
        len(verifications),
        len(rankings),
    )

    evaluation_master: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_title": "Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation Benchmarks",
    }

    # ----------------------------------------------------
    # 2. Retrieval Evaluation
    # ----------------------------------------------------
    logger.info("\nExecuting Retrieval Evaluation (P@K, R@K, MRR, nDCG@K)...")
    retrieval_report = evaluate_retrieval_suite(
        prepared_gaps=prepared_gaps,
        retrieved_evidence=retrieved_ev,
        doc_concepts_df=doc_concepts_df,
        k_values=[5, 10, 20],
    )
    evaluation_master["retrieval_evaluation"] = retrieval_report.to_dict()
    logger.info(
        "Retrieval Results: P@5=%.4f, P@10=%.4f, MRR=%.4f, nDCG@10=%.4f across %d queries.",
        retrieval_report.mean_precision_at_k.get(5, 0.0),
        retrieval_report.mean_precision_at_k.get(10, 0.0),
        retrieval_report.mean_mrr,
        retrieval_report.mean_ndcg_at_k.get(10, 0.0),
        retrieval_report.num_queries,
    )

    # ----------------------------------------------------
    # 3. Grounding & Verification Evaluation
    # ----------------------------------------------------
    logger.info("\nExecuting Grounding & Traceability Evaluation...")
    grounding_report = evaluate_grounding(
        hypotheses=hypotheses,
        verifications=verifications,
        nli_results=nli_results,
    )
    evaluation_master["grounding_evaluation"] = grounding_report.to_dict()
    logger.info(
        "Grounding Results: Citation Validity=%.1f%%, Full Traceability=%.1f%%, Flagged Insufficient=%.1f%%.",
        grounding_report.citation_validity_rate * 100.0,
        grounding_report.traceability_rate * 100.0,
        grounding_report.flagged_insufficient_support_rate * 100.0,
    )

    # ----------------------------------------------------
    # 4. Baseline Comparisons
    # ----------------------------------------------------
    logger.info("\nExecuting Baseline Comparisons...")
    # Baseline 1: Gap Ranking
    gap_comp = compare_gap_ranking_baselines(gaps_df, cooc_df)
    logger.info("  Gap Ranking Comparison: Spearman corr=%.4f", gap_comp.rank_correlation_spearman)

    # Baseline 2: Retrieval Diversity
    from src.phase2.retrieval import EvidenceRetriever
    index_dir = outputs_dir / "faiss_index"
    retriever_instance = EvidenceRetriever(index_path=index_dir)
    retrieval_div_comp = compare_retrieval_diversity(retriever_instance, prepared_gaps, top_k=20)
    logger.info("  Retrieval Diversity: %+.1f%% unique papers", retrieval_div_comp.paper_diversity_gain_pct)

    # Baseline 3: Generation Grounding
    gen_comp = compare_generation_baselines(hypotheses, verifications, judge_results)
    logger.info("  Generation Grounding: Validity %.1f%% (Proposed) vs %.1f%% (No-RAG)",
                gen_comp.proposed_citation_validity * 100.0, gen_comp.no_rag_citation_validity * 100.0)

    evaluation_master["baseline_comparisons"] = {
        "gap_ranking_baseline": gap_comp.to_dict(),
        "retrieval_diversity_baseline": retrieval_div_comp.to_dict(),
        "generation_grounding_baseline": gen_comp.to_dict(),
    }

    # ----------------------------------------------------
    # 5. Ablation Study
    # ----------------------------------------------------
    logger.info("\nExecuting Ablation Study...")
    ablation_report = run_ablation_experiments(
        hypotheses=hypotheses,
        rankings=rankings,
        verifications=verifications,
        nli_results=nli_results,
        judge_results=judge_results,
        rule_results={},
        gaps_df=gaps_df,
    )
    evaluation_master["ablation_study"] = ablation_report.to_dict()
    for abl in ablation_report.ablations:
        logger.info("  Ablation '%s': Utility Score=%.4f", abl.configuration_name, abl.composite_utility_score)

    # ----------------------------------------------------
    # 6. Save Outputs
    # ----------------------------------------------------
    eval_json_path = outputs_dir / "phase3_evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_master, f, indent=2, ensure_ascii=False)
    logger.info("\nSaved full evaluation JSON to: %s", eval_json_path)

    # Save summary metrics table CSV
    summary_rows = [
        {"metric_category": "Retrieval", "metric_name": "Precision@5", "value": round(retrieval_report.mean_precision_at_k.get(5, 0.0), 4), "benchmark_note": "PubMed MeSH ground truth"},
        {"metric_category": "Retrieval", "metric_name": "Precision@10", "value": round(retrieval_report.mean_precision_at_k.get(10, 0.0), 4), "benchmark_note": "PubMed MeSH ground truth"},
        {"metric_category": "Retrieval", "metric_name": "Recall@10", "value": round(retrieval_report.mean_recall_at_k.get(10, 0.0), 4), "benchmark_note": "Corpus relevant pool"},
        {"metric_category": "Retrieval", "metric_name": "Mean Reciprocal Rank (MRR)", "value": round(retrieval_report.mean_mrr, 4), "benchmark_note": "First relevant hit rank"},
        {"metric_category": "Retrieval", "metric_name": "nDCG@10", "value": round(retrieval_report.mean_ndcg_at_k.get(10, 0.0), 4), "benchmark_note": "Graded relevance 2=joint, 1=single"},
        {"metric_category": "Grounding", "metric_name": "Citation Validity Rate", "value": round(grounding_report.citation_validity_rate, 4), "benchmark_note": "Cited IDs matching candidate pool"},
        {"metric_category": "Grounding", "metric_name": "Traceability Rate", "value": round(grounding_report.traceability_rate, 4), "benchmark_note": "Hypotheses with 100% valid citations"},
        {"metric_category": "Grounding", "metric_name": "Evidence Coverage", "value": round(grounding_report.mean_evidence_coverage, 4), "benchmark_note": "Fraction of retrieved chunks cited"},
        {"metric_category": "Grounding", "metric_name": "NLI Entailment Rate", "value": round(grounding_report.nli_percentages.get("supported", 0.0), 2), "benchmark_note": "DeBERTa-v3 cross-encoder"},
        {"metric_category": "Grounding", "metric_name": "Fabricated Citations", "value": grounding_report.fabricated_citation_count, "benchmark_note": "Hallucinated reference count"},
        {"metric_category": "Baseline", "metric_name": "Paper Diversity Gain", "value": f"{retrieval_div_comp.paper_diversity_gain_pct:+.1f}%", "benchmark_note": "vs Standard Dense (no diversity)"},
        {"metric_category": "Baseline", "metric_name": "Hallucination Reduction", "value": f"{(gen_comp.no_rag_hallucination_rate - gen_comp.proposed_hallucination_rate)*100:.1f}%", "benchmark_note": "RAG vs Direct Prompting (No RAG)"},
        {"metric_category": "Ablation", "metric_name": "Full Pipeline Utility", "value": round(ablation_report.baseline_full_pipeline.composite_utility_score, 4), "benchmark_note": "Composite metric"},
        {"metric_category": "Ablation", "metric_name": "Without RAG Utility", "value": round(ablation_report.ablations[0].composite_utility_score, 4), "benchmark_note": "Direct Prompting ablation"},
        {"metric_category": "Ablation", "metric_name": "Without NLI Utility", "value": round(ablation_report.ablations[1].composite_utility_score, 4), "benchmark_note": "No cross-encoder ablation"},
        {"metric_category": "Ablation", "metric_name": "Without Temporal Utility", "value": round(ablation_report.ablations[3].composite_utility_score, 4), "benchmark_note": "Static gap score ablation"},
    ]
    summary_df = pd.DataFrame(summary_rows)
    eval_csv_path = outputs_dir / "phase3_evaluation_summary.csv"
    summary_df.to_csv(eval_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved evaluation summary CSV to: %s", eval_csv_path)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 EVALUATION COMPLETE in %.2f seconds", elapsed)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_full_evaluation())
