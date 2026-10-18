"""
run_phase2.py — Master orchestrator for Phase 2 pipeline.

Executes the complete evidence-grounded hypothesis generation pipeline:
1. Load and validate Phase 1 artifacts
2. Prepare and prioritize research gaps
3. Chunk documents for retrieval
4. Build/load FAISS index
5. Retrieve evidence for gaps
6. Generate hypotheses using LLM
7. Verify evidence references
8. Run NLI validation (if enabled)
9. Run LLM-as-a-judge evaluation (if enabled)
10. Apply rule-based checks
11. Rank hypotheses
12. Deduplicate hypotheses
13. Save all outputs

Usage:
    python run_phase2.py                    # full pipeline (requires GEMINI_API_KEY)
    python run_phase2.py --mock             # use mock LLM responses (no API key needed)
    python run_phase2.py --max-gaps 5       # process only top 5 gaps
    python run_phase2.py --skip-nli         # skip NLI validation
    python run_phase2.py --rebuild-index    # force rebuild FAISS index
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from src.config_loader import load_config, ensure_dirs, get_project_root

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger("Phase2Pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Evidence-Grounded Hypothesis Generation")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM responses (no API key needed)")
    parser.add_argument("--max-gaps", type=int, default=None,
                        help="Maximum number of gaps to process (overrides config)")
    parser.add_argument("--skip-nli", action="store_true",
                        help="Skip NLI validation stage")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-a-judge stage")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild FAISS index")
    parser.add_argument("--gaps-only", action="store_true",
                        help="Only prepare gaps, skip generation")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    dirs = ensure_dirs()
    p2_cfg = cfg.get("phase2", {})

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 2: EVIDENCE-GROUNDED HYPOTHESIS GENERATION")
    logger.info("=" * 70)

    use_mock = args.mock or os.environ.get("USE_MOCK_LLM", "").lower() == "true" or p2_cfg.get("llm", {}).get("use_mock", False)
    if use_mock:
        logger.info(">>> MOCK MODE: Using mock LLM responses (no API calls)")

    outputs_dir = dirs["outputs"]
    results = {
        "started_at": datetime.now().isoformat(),
        "mock_mode": use_mock,
        "stages_completed": [],
        "stages_failed": [],
        "warnings": [],
    }

    # ================================================================
    # STAGE 1: Load Phase 1 artifacts
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Loading Phase 1 artifacts")
    logger.info("=" * 60)

    try:
        from src.phase2.gap_preparation import load_phase1_gaps, load_phase1_artifacts

        gaps_df = load_phase1_gaps(outputs_dir)
        artifacts = load_phase1_artifacts(get_project_root())

        logger.info(f"  Loaded {len(gaps_df)} candidate research gaps")
        logger.info(f"  Loaded {len(artifacts['papers'])} preprocessed papers")
        logger.info(f"  Loaded {len(artifacts['topic_assignments'])} topic assignments")
        results["stages_completed"].append("load_phase1_artifacts")
    except Exception as e:
        logger.error(f"FATAL: Failed to load Phase 1 artifacts: {e}")
        results["stages_failed"].append(("load_phase1_artifacts", str(e)))
        _save_results(results, outputs_dir, start_time)
        return 1

    # ================================================================
    # STAGE 2: Prepare and prioritize gaps
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Preparing and prioritizing research gaps")
    logger.info("=" * 60)

    try:
        from src.phase2.gap_preparation import prioritize_gaps, save_prepared_gaps

        max_gaps = args.max_gaps or p2_cfg.get("gap_preparation", {}).get("max_gaps_to_process", 20)
        gaps_df = prioritize_gaps(gaps_df, max_gaps=max_gaps)
        logger.info(f"  Prioritized top {len(gaps_df)} gaps")

        save_prepared_gaps(gaps_df, outputs_dir / "phase2_prepared_gaps.json")
        results["stages_completed"].append("gap_preparation")
        results["num_gaps_prepared"] = len(gaps_df)
    except Exception as e:
        logger.error(f"Failed to prepare gaps: {e}", exc_info=True)
        results["stages_failed"].append(("gap_preparation", str(e)))
        _save_results(results, outputs_dir, start_time)
        return 1

    if args.gaps_only:
        logger.info(">>> --gaps-only flag: Stopping after gap preparation")
        _save_results(results, outputs_dir, start_time)
        return 0

    # ================================================================
    # STAGE 3: Chunk documents for retrieval
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Chunking documents for retrieval")
    logger.info("=" * 60)

    chunks_path = outputs_dir / "chunks.json"
    try:
        from src.phase2.chunking import create_chunks, save_chunks, load_chunks, get_chunk_statistics

        if chunks_path.exists() and not args.rebuild_index:
            logger.info("  Loading cached chunks...")
            chunks = load_chunks(chunks_path)
            logger.info(f"  Loaded {len(chunks)} cached chunks")
        else:
            chunks = create_chunks(
                artifacts["papers"],
                chunk_size=p2_cfg.get("chunking", {}).get("chunk_size", 512),
                chunk_overlap=p2_cfg.get("chunking", {}).get("chunk_overlap", 128),
                min_chunk_length=p2_cfg.get("chunking", {}).get("min_chunk_length", 50),
                topic_assignments_df=artifacts.get("topic_assignments"),
                document_concepts_df=artifacts.get("document_concepts"),
            )
            save_chunks(chunks, chunks_path)
            logger.info(f"  Created and saved {len(chunks)} chunks")

        stats = get_chunk_statistics(chunks)
        logger.info(f"  Chunk statistics: {json.dumps(stats, indent=2)}")
        results["stages_completed"].append("chunking")
        results["num_chunks"] = len(chunks)
    except Exception as e:
        logger.error(f"Failed to chunk documents: {e}", exc_info=True)
        results["stages_failed"].append(("chunking", str(e)))
        _save_results(results, outputs_dir, start_time)
        return 1

    # ================================================================
    # STAGE 4: Build/load FAISS index and retrieve evidence
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Building retrieval index and retrieving evidence")
    logger.info("=" * 60)

    try:
        from src.phase2.retrieval import EvidenceRetriever, save_retrieved_evidence

        retrieval_cfg = p2_cfg.get("retrieval", {})
        index_dir = outputs_dir / "faiss_index"

        retriever = EvidenceRetriever(
            embedding_model_name=retrieval_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            index_path=index_dir if index_dir.exists() and not args.rebuild_index else None,
        )

        if not retriever.index_loaded:
            logger.info("  Building new FAISS index...")
            retriever.build_index(chunks, batch_size=retrieval_cfg.get("batch_size", 64))
            retriever.save_index(index_dir)
            logger.info("  Index built and saved")
        else:
            logger.info("  Loaded existing FAISS index")

        # Retrieve evidence for each gap
        top_k = retrieval_cfg.get("top_k", 20)
        min_score = retrieval_cfg.get("min_relevance_score", 0.2)
        all_evidence = {}

        for idx, (_, gap) in enumerate(gaps_df.iterrows()):
            gap_id = gap.get("gap_id", f"GAP-{idx+1:03d}")
            logger.info(f"  Retrieving evidence for {gap_id}: "
                       f"[{gap['concept_a']}] <-> [{gap['concept_b']}]")
            evidence = retriever.retrieve_for_gap(gap, top_k=top_k, min_score=min_score)
            all_evidence[gap_id] = evidence
            logger.info(f"    Retrieved {len(evidence)} passages")

        save_retrieved_evidence(all_evidence, outputs_dir / "retrieved_gap_evidence.json")
        results["stages_completed"].append("evidence_retrieval")
        results["total_evidence_passages"] = sum(len(v) for v in all_evidence.values())
    except Exception as e:
        logger.error(f"Failed in retrieval stage: {e}", exc_info=True)
        results["stages_failed"].append(("evidence_retrieval", str(e)))
        _save_results(results, outputs_dir, start_time)
        return 1

    # ================================================================
    # STAGE 5: Generate hypotheses
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: Generating hypotheses")
    logger.info("=" * 60)

    try:
        from src.phase2.hypothesis_generation import HypothesisGenerator, save_hypotheses

        llm_cfg = p2_cfg.get("llm", {})
        generator = HypothesisGenerator(
            model_name=llm_cfg.get("model_name"),
            max_hypotheses_per_gap=llm_cfg.get("max_hypotheses_per_gap", 3),
            use_mock=use_mock,
        )

        all_hypotheses = []
        for idx, (_, gap) in enumerate(gaps_df.iterrows()):
            gap_id = gap.get("gap_id", f"GAP-{idx+1:03d}")
            evidence = all_evidence.get(gap_id, [])
            logger.info(f"  Generating hypotheses for {gap_id} "
                       f"({len(evidence)} evidence passages)...")

            try:
                hypotheses = generator.generate_hypotheses(
                    gap=gap,
                    evidence_passages=evidence,
                    max_hypotheses=llm_cfg.get("max_hypotheses_per_gap", 3),
                )
                all_hypotheses.extend(hypotheses)
                logger.info(f"    Generated {len(hypotheses)} hypotheses")
            except Exception as e:
                logger.warning(f"    Failed to generate for {gap_id}: {e}")
                results["warnings"].append(f"Generation failed for {gap_id}: {str(e)}")

        save_hypotheses(all_hypotheses, outputs_dir / "generated_hypotheses.json")
        results["stages_completed"].append("hypothesis_generation")
        results["num_hypotheses_generated"] = len(all_hypotheses)
        logger.info(f"  Total hypotheses generated: {len(all_hypotheses)}")
    except Exception as e:
        logger.error(f"Failed in hypothesis generation: {e}", exc_info=True)
        results["stages_failed"].append(("hypothesis_generation", str(e)))
        _save_results(results, outputs_dir, start_time)
        return 1

    if not all_hypotheses:
        logger.warning("No hypotheses generated. Stopping pipeline.")
        _save_results(results, outputs_dir, start_time)
        return 1

    # ================================================================
    # STAGE 6: Evidence verification
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 6: Evidence verification")
    logger.info("=" * 60)

    verification_results = {}
    try:
        from src.phase2.evidence_verification import aggregate_verification, save_verification_results

        # Build evidence pool lookup
        evidence_pool = {}
        for gap_id, ev_list in all_evidence.items():
            for ev in ev_list:
                eid = ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)
                if eid:
                    evidence_pool[eid] = ev

        for hyp in all_hypotheses:
            hyp_id = hyp.get("hypothesis_id") if isinstance(hyp, dict) else getattr(hyp, "hypothesis_id", None)
            gap_id = hyp.get("gap_id") if isinstance(hyp, dict) else getattr(hyp, "gap_id", None)
            if hyp_id:
                gap_ev = all_evidence.get(gap_id, list(evidence_pool.values()))
                result = aggregate_verification(hyp, gap_ev, artifacts["papers"])
                verification_results[hyp_id] = result

        save_verification_results(verification_results, outputs_dir / "verification_results.json")
        results["stages_completed"].append("evidence_verification")
        logger.info(f"  Verified {len(verification_results)} hypotheses")
    except Exception as e:
        logger.error(f"Evidence verification failed: {e}", exc_info=True)
        results["stages_failed"].append(("evidence_verification", str(e)))
        results["warnings"].append("Evidence verification failed - continuing with remaining stages")

    # ================================================================
    # STAGE 7: NLI validation (optional)
    # ================================================================
    nli_results = {}
    if not args.skip_nli and p2_cfg.get("nli", {}).get("enabled", True):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 7: NLI validation")
        logger.info("=" * 60)

        try:
            from src.phase2.nli_validation import NLIValidator, save_nli_results

            nli_validator = NLIValidator()

            if nli_validator.is_available():
                for hyp in all_hypotheses:
                    hyp_id = hyp.get("hypothesis_id") if isinstance(hyp, dict) else getattr(hyp, "hypothesis_id", None)
                    hyp_text = hyp.get("hypothesis") if isinstance(hyp, dict) else getattr(hyp, "hypothesis", None)
                    gap_id = hyp.get("gap_id") if isinstance(hyp, dict) else getattr(hyp, "gap_id", None)

                    if hyp_id and hyp_text and gap_id:
                        ev_list = all_evidence.get(gap_id, [])
                        ev_texts = []
                        ev_ids = []
                        for ev in ev_list[:10]:  # Limit to avoid excessive computation
                            t = ev.get("text_content") if isinstance(ev, dict) else getattr(ev, "text_content", None)
                            eid = ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)
                            if t and eid:
                                ev_texts.append(t)
                                ev_ids.append(eid)

                        if ev_texts:
                            nli_result = nli_validator.validate_hypothesis(hyp_text, ev_texts, ev_ids)
                            nli_results[hyp_id] = nli_result

                save_nli_results(nli_results, outputs_dir / "nli_results.json")
                results["stages_completed"].append("nli_validation")
                logger.info(f"  NLI validation completed for {len(nli_results)} hypotheses")
            else:
                logger.warning("  NLI model not available - skipping")
                results["warnings"].append("NLI model could not be loaded - stage skipped")
        except Exception as e:
            logger.error(f"NLI validation failed: {e}", exc_info=True)
            results["stages_failed"].append(("nli_validation", str(e)))
            results["warnings"].append("NLI validation failed - continuing without NLI")
    else:
        logger.info("\nSTAGE 7: NLI validation — SKIPPED (disabled)")
        results["warnings"].append("NLI validation skipped by user/config")

    # ================================================================
    # STAGE 8: LLM-as-a-judge (optional)
    # ================================================================
    judge_results = {}
    if not args.skip_judge and p2_cfg.get("judge", {}).get("enabled", True):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 8: LLM-as-a-judge evaluation")
        logger.info("=" * 60)

        try:
            from src.phase2.llm_judge import LLMJudge, save_judge_results

            judge = LLMJudge(use_mock=use_mock)

            for idx, (_, gap) in enumerate(gaps_df.iterrows()):
                gap_id = gap.get("gap_id", f"GAP-{idx+1:03d}")
                gap_hypotheses = [h for h in all_hypotheses if
                                  (h.get("gap_id") if isinstance(h, dict) else getattr(h, "gap_id", None)) == gap_id]

                for hyp in gap_hypotheses:
                    hyp_id = hyp.get("hypothesis_id") if isinstance(hyp, dict) else getattr(hyp, "hypothesis_id", None)
                    evidence = all_evidence.get(gap_id, [])
                    try:
                        result = judge.evaluate_hypothesis(hyp, gap, evidence)
                        judge_results[hyp_id] = result
                    except Exception as e:
                        logger.warning(f"  Judge failed for {hyp_id}: {e}")

            save_judge_results(judge_results, outputs_dir / "judge_results.json")
            results["stages_completed"].append("llm_judge")
            logger.info(f"  Judge evaluation completed for {len(judge_results)} hypotheses")
        except Exception as e:
            logger.error(f"LLM judge failed: {e}", exc_info=True)
            results["stages_failed"].append(("llm_judge", str(e)))
    else:
        logger.info("\nSTAGE 8: LLM-as-a-judge — SKIPPED (disabled)")

    # ================================================================
    # STAGE 9: Rule-based checks
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 9: Rule-based validation checks")
    logger.info("=" * 60)

    rule_results = {}
    try:
        from src.phase2.rule_checks import run_all_checks

        evidence_pool_for_rules = {}
        for gap_id, ev_list in all_evidence.items():
            for ev in ev_list:
                eid = ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)
                if eid:
                    evidence_pool_for_rules[eid] = ev

        for hyp in all_hypotheses:
            hyp_id = hyp.get("hypothesis_id") if isinstance(hyp, dict) else getattr(hyp, "hypothesis_id", None)
            if hyp_id:
                result = run_all_checks(hyp, evidence_pool=evidence_pool_for_rules)
                rule_results[hyp_id] = result

        # Save rule check results
        rule_output = {}
        for hyp_id, res in rule_results.items():
            rule_output[hyp_id] = res if isinstance(res, dict) else (
                res.to_dict() if hasattr(res, "to_dict") else str(res)
            )
        with open(outputs_dir / "rule_check_results.json", "w", encoding="utf-8") as f:
            json.dump(rule_output, f, indent=2, ensure_ascii=False, default=str)

        results["stages_completed"].append("rule_checks")
        logger.info(f"  Rule checks completed for {len(rule_results)} hypotheses")
    except Exception as e:
        logger.error(f"Rule checks failed: {e}", exc_info=True)
        results["stages_failed"].append(("rule_checks", str(e)))

    # ================================================================
    # STAGE 10: Ranking
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 10: Hypothesis ranking")
    logger.info("=" * 60)

    rankings = []
    try:
        from src.phase2.ranking import rank_hypotheses, save_rankings

        rankings = rank_hypotheses(
            all_hypotheses,
            verification_results,
            nli_results,
            judge_results,
            rule_results,
        )
        save_rankings(rankings, outputs_dir / "hypothesis_rankings.json")
        results["stages_completed"].append("ranking")
        logger.info(f"  Ranked {len(rankings)} hypotheses")
    except Exception as e:
        logger.error(f"Ranking failed: {e}", exc_info=True)
        results["stages_failed"].append(("ranking", str(e)))

    # ================================================================
    # STAGE 11: Deduplication
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 11: Semantic deduplication")
    logger.info("=" * 60)

    try:
        from src.phase2.deduplication import SemanticDeduplicator, save_deduplication_results

        dedup_cfg = p2_cfg.get("deduplication", {})
        deduplicator = SemanticDeduplicator(
            threshold=dedup_cfg.get("similarity_threshold", 0.85),
        )

        dedup_result = deduplicator.deduplicate(all_hypotheses, rankings)
        save_deduplication_results(dedup_result, outputs_dir / "deduplication_results.json")

        results["stages_completed"].append("deduplication")
        results["num_hypotheses_after_dedup"] = dedup_result.get("total_output") if isinstance(dedup_result, dict) else getattr(dedup_result, "total_output", len(all_hypotheses))
        logger.info(f"  Deduplication: {len(all_hypotheses)} -> {results['num_hypotheses_after_dedup']} hypotheses")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}", exc_info=True)
        results["stages_failed"].append(("deduplication", str(e)))

    # ================================================================
    # STAGE 12: Save final outputs
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 12: Saving final outputs")
    logger.info("=" * 60)

    try:
        # Create final summary combining everything
        final_output = {
            "project": "Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation",
            "phase": "Phase 2",
            "generated_at": datetime.now().isoformat(),
            "mock_mode": use_mock,
            "gaps_processed": len(gaps_df),
            "hypotheses_generated": len(all_hypotheses),
            "hypotheses_verified": len(verification_results),
            "nli_validated": len(nli_results),
            "judge_evaluated": len(judge_results),
            "rule_checked": len(rule_results),
            "ranked": len(rankings),
            "pipeline_stages": results["stages_completed"],
            "warnings": results["warnings"],
            "failures": results["stages_failed"],
        }

        with open(outputs_dir / "phase2_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False, default=str)

        # Create CSV export of final hypotheses
        _export_hypotheses_csv(all_hypotheses, rankings, verification_results,
                               judge_results, nli_results, rule_results,
                               outputs_dir / "final_hypotheses.csv")

        results["stages_completed"].append("save_outputs")
        logger.info("  All outputs saved successfully")
    except Exception as e:
        logger.error(f"Failed to save final outputs: {e}", exc_info=True)
        results["stages_failed"].append(("save_outputs", str(e)))

    # Final summary
    _save_results(results, outputs_dir, start_time)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 PIPELINE COMPLETE")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Stages completed: {len(results['stages_completed'])}")
    logger.info(f"  Stages failed: {len(results['stages_failed'])}")
    logger.info(f"  Warnings: {len(results['warnings'])}")
    logger.info(f"  Hypotheses generated: {results.get('num_hypotheses_generated', 0)}")
    logger.info("=" * 70)

    return 0 if not results["stages_failed"] else 1


def _save_results(results, outputs_dir, start_time):
    """Save pipeline execution results."""
    results["completed_at"] = datetime.now().isoformat()
    results["elapsed_seconds"] = round(time.time() - start_time, 1)
    try:
        with open(outputs_dir / "phase2_pipeline_log.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass


def _export_hypotheses_csv(hypotheses, rankings, verifications, judge_results,
                           nli_results, rule_results, output_path):
    """Export final hypotheses to CSV with all quality indicators."""
    import pandas as pd

    rows = []
    # Build ranking lookup
    ranking_lookup = {}
    for r in rankings:
        rid = r.get("hypothesis_id") if isinstance(r, dict) else getattr(r, "hypothesis_id", None)
        if rid:
            ranking_lookup[rid] = r

    for hyp in hypotheses:
        h = hyp if isinstance(hyp, dict) else hyp.__dict__ if hasattr(hyp, "__dict__") else {}
        hyp_id = h.get("hypothesis_id", "")

        row = {
            "hypothesis_id": hyp_id,
            "gap_id": h.get("gap_id", ""),
            "hypothesis": h.get("hypothesis", ""),
            "research_question": h.get("research_question", ""),
            "independent_variable": h.get("independent_variable", ""),
            "dependent_variable": h.get("dependent_variable", ""),
            "target_population": h.get("target_population", ""),
            "proposed_relationship": h.get("proposed_relationship", ""),
            "potential_contribution": h.get("potential_contribution", ""),
            "supporting_evidence_count": len(h.get("supporting_evidence_ids", [])),
            "contradictory_evidence_count": len(h.get("contradictory_evidence_ids", [])),
        }

        # Add ranking info
        rank = ranking_lookup.get(hyp_id, {})
        if isinstance(rank, dict):
            row["rank_position"] = rank.get("rank_position", "")
            row["rank_score"] = rank.get("rank_score", "")
            row["quality_tier"] = rank.get("quality_tier", "")
        elif hasattr(rank, "rank_position"):
            row["rank_position"] = rank.rank_position
            row["rank_score"] = rank.rank_score
            row["quality_tier"] = rank.quality_tier

        # Add judge scores
        judge = judge_results.get(hyp_id, {})
        if isinstance(judge, dict):
            for dim in ["relevance_to_gap", "clarity", "logical_consistency",
                        "testability", "evidence_alignment", "novelty_potential"]:
                row[f"judge_{dim}"] = judge.get(dim, "")
        elif hasattr(judge, "relevance_to_gap"):
            for dim in ["relevance_to_gap", "clarity", "logical_consistency",
                        "testability", "evidence_alignment", "novelty_potential"]:
                row[f"judge_{dim}"] = getattr(judge, dim, "")

        rows.append(row)

    df = pd.DataFrame(rows)
    if "rank_position" in df.columns:
        df = df.sort_values("rank_position", na_position="last")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"  Exported {len(df)} hypotheses to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
