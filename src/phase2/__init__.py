"""
Phase 2 — Evidence-Grounded Hypothesis Generation Pipeline.

Modules:
    llm_judge: LLM-as-a-judge quality assessment
    rule_checks: Rule-based validation checks
    ranking: Hypothesis ranking and scoring
    deduplication: Semantic deduplication of hypotheses
"""
from .gap_preparation import (
    load_phase1_gaps,
    load_phase1_artifacts,
    validate_gap_record,
    prioritize_gaps,
    get_gap_context,
    save_prepared_gaps,
)
from .chunking import (
    ChunkMetadata,
    construct_source_url,
    create_chunks,
    get_chunks_for_paper,
    save_chunks,
    load_chunks,
    get_chunk_statistics,
)
from .retrieval import (
    RetrievedEvidence,
    EvidenceRetriever,
    save_retrieved_evidence,
    load_retrieved_evidence,
)
from .llm_judge import LLMJudge, JudgeResult, save_judge_results
from .rule_checks import (
    CheckDetail,
    RuleCheckResult,
    run_all_checks,
    save_rule_results,
    check_has_variables,
    check_has_population,
    check_falsifiability,
    check_causal_language,
    check_evidence_references,
    check_evidence_coverage,
    check_specificity,
    check_hypothesis_length,
    check_conflicting_evidence_acknowledged,
)
from .ranking import (
    RankedHypothesis,
    rank_hypotheses,
    apply_ranking,
    get_ranking_summary,
    save_rankings,
)
from .deduplication import (
    DuplicateGroup,
    DeduplicationResult,
    SemanticDeduplicator,
    get_deduplicated_hypotheses,
    save_deduplication_results,
)

__all__ = [
    # Gap Preparation
    "load_phase1_gaps",
    "load_phase1_artifacts",
    "validate_gap_record",
    "prioritize_gaps",
    "get_gap_context",
    "save_prepared_gaps",
    # Chunking
    "ChunkMetadata",
    "construct_source_url",
    "create_chunks",
    "get_chunks_for_paper",
    "save_chunks",
    "load_chunks",
    "get_chunk_statistics",
    # Retrieval
    "RetrievedEvidence",
    "EvidenceRetriever",
    "save_retrieved_evidence",
    "load_retrieved_evidence",
    # LLM Judge
    "LLMJudge",
    "JudgeResult",
    "save_judge_results",
    # Rule Checks
    "CheckDetail",
    "RuleCheckResult",
    "run_all_checks",
    "save_rule_results",
    "check_has_variables",
    "check_has_population",
    "check_falsifiability",
    "check_causal_language",
    "check_evidence_references",
    "check_evidence_coverage",
    "check_specificity",
    "check_hypothesis_length",
    "check_conflicting_evidence_acknowledged",
    # Ranking
    "RankedHypothesis",
    "rank_hypotheses",
    "apply_ranking",
    "get_ranking_summary",
    "save_rankings",
    # Deduplication
    "DuplicateGroup",
    "DeduplicationResult",
    "SemanticDeduplicator",
    "get_deduplicated_hypotheses",
    "save_deduplication_results",
]
