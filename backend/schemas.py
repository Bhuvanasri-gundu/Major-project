"""
backend/schemas.py — Typed Pydantic v2 schemas for FastAPI endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SuggestionItem(BaseModel):
    text: str
    type: str = Field(description="'gap' | 'concept' | 'topic'")
    category: Optional[str] = None
    details: Optional[str] = None
    score: float = 1.0


class SuggestionsResponse(BaseModel):
    query: str
    count: int
    suggestions: List[SuggestionItem]


class GenerateRequest(BaseModel):
    theme: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User's research theme or question in mHealth."
    )
    population: Optional[str] = Field(
        default="",
        max_length=300,
        description="Optional target population (e.g. adults with type 2 diabetes)."
    )
    objective: Optional[str] = Field(
        default="",
        max_length=500,
        description="Optional research objective."
    )
    hypothesis_count: int = Field(
        default=3,
        ge=0,
        le=3,
        description="Number of hypotheses to generate (0-3). 0 = gap matching and retrieval only."
    )
    selected_gap_id: Optional[str] = Field(
        default=None,
        description="Optional specific candidate gap ID to bridge."
    )


class EvidenceSourceRecord(BaseModel):
    chunk_id: str
    pmid: Optional[str] = None
    paper_title: Optional[str] = None
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    source_url: Optional[str] = None
    text_content: str
    relevance_score: float = 0.0
    evidence_category: str = "supporting"  # "supporting" | "contradictory" | "contextual"
    verification_status: str = "verified"


class ValidationDetails(BaseModel):
    overview: Dict[str, Any]
    gap_alignment: Dict[str, Any]
    evidence_sources: List[EvidenceSourceRecord] = []
    citation_audit: Dict[str, Any] = {}
    nli_validation: Optional[Dict[str, Any]] = None
    llm_judge: Optional[Dict[str, Any]] = None
    rule_checks: Dict[str, Any] = {}
    testability_assessment: Dict[str, Any] = {}
    ranking_deduplication: Dict[str, Any] = {}


class HypothesisOutput(BaseModel):
    """
    Hypothesis representation.
    IMPORTANT: Independent variables (IV) and dependent variables (DV) are preserved
    internally for rule checking and testability analysis, but are NOT displayed
    as separate fields on Screen 1 hypothesis cards.
    """
    hypothesis_id: str
    hypothesis_number: int
    hypothesis: str
    target_population: str
    proposed_mechanism: Optional[str] = None
    directionality: Optional[str] = None
    gap_reference: str
    gap_id: str
    evidence_summary: Optional[str] = None
    validation_status_badge: str  # e.g. "Checks passed" | "Review required" | "Issues detected"
    composite_rank_score: Optional[float] = None
    quality_tier: Optional[str] = None  # "high" | "medium" | "low"
    supporting_evidence_count: int = 0
    contradictory_evidence_count: int = 0
    # Internal variables for Screen 2 validation details only:
    internal_iv: Optional[str] = None
    internal_dv: Optional[str] = None
    validation: Optional[ValidationDetails] = None


class MatchedGap(BaseModel):
    gap_id: str
    concept_a: str
    concept_b: str
    gap_score: float
    semantic_similarity: float
    temporal_growth_avg: Optional[float] = None
    priority_score: Optional[float] = None
    priority_rank: Optional[int] = None
    category: Optional[str] = None
    matching_score: float = 0.0
    matching_explanation: str = ""
    supporting_topic_ids: List[str] = []
    supporting_pmids_count: int = 0
    description: str = ""
    is_precomputed: bool = True


class RunResponse(BaseModel):
    run_id: str
    theme: str
    population: str
    objective: str
    hypothesis_count_requested: int
    hypothesis_count_generated: int
    status: str  # "completed" | "partially_completed" | "no_hypotheses_requested" | "failed"
    current_stage: str
    started_at: str
    completed_at: Optional[str] = None
    stages_completed: List[str] = []
    warnings: List[str] = []
    failures: List[str] = []
    matched_gaps: List[MatchedGap] = []
    hypotheses: List[HypothesisOutput] = []


class EvaluationBenchmark(BaseModel):
    category: str
    metric_name: str
    value: Any
    unit: str
    description: str
    methodology: str
