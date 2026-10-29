"""
backend/routes/api.py — REST API routes for the mHealth Hypothesis Generation system.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.schemas import (
    GenerateRequest,
    RunResponse,
    SuggestionsResponse,
    SuggestionItem,
    MatchedGap,
)
from backend.services.theme_matcher import get_theme_matcher
from backend.services.pipeline_service import get_pipeline_service
from backend.services.evaluation_service import get_evaluation_benchmarks

logger = logging.getLogger("APIRouter")
router = APIRouter(prefix="/api")


@router.get("/health")
def health_check():
    """Health check reporting pipeline status and Gemini key presence without exposing keys."""
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return {
        "status": "healthy",
        "service": "Ontology-Guided Hypothesis Generation API",
        "gemini_api_configured": has_gemini,
        "configured_model": os.environ.get("LLM_MODEL_NAME", "gemini-3.1-flash-lite"),
        "corpus_screened_papers": 12642,
        "candidate_gaps_count": 50,
    }


@router.get("/suggestions", response_model=SuggestionsResponse)
def get_theme_suggestions(q: str = Query(..., min_length=2, description="Search query string")):
    """
    Live theme suggestions derived from actual candidate research gaps,
    MeSH concepts, and discovery topics.
    """
    try:
        matcher = get_theme_matcher()
        results = matcher.get_live_suggestions(q, limit=8)
        items = [
            SuggestionItem(
                text=r["text"],
                type=r["type"],
                category=r.get("category"),
                details=r.get("details"),
                score=r.get("score", 1.0),
            )
            for r in results
        ]
        return SuggestionsResponse(query=q, count=len(items), suggestions=items)
    except Exception as e:
        logger.error("Error generating suggestions for query '%s': %s", q, e)
        raise HTTPException(status_code=500, detail="Failed to fetch theme suggestions from backend data.")


@router.post("/generate", response_model=RunResponse)
def generate_hypotheses_endpoint(request: GenerateRequest):
    """
    Trigger genuine evidence-grounded hypothesis generation.
    Retrieves evidence from FAISS index and invokes Gemini model.
    """
    try:
        pipeline = get_pipeline_service()
        run_result = pipeline.execute_run(request)
        return run_result
    except ValueError as ve:
        logger.warning("Validation error in generation request: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Generation pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation pipeline error: {str(e)}")


@router.get("/runs/latest", response_model=RunResponse)
def get_latest_run_endpoint():
    """Fetches the latest completed hypothesis generation run."""
    pipeline = get_pipeline_service()
    run = pipeline.get_latest_run()
    if not run:
        raise HTTPException(status_code=404, detail="No hypothesis generation runs have been recorded yet.")
    return run


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run_by_id_endpoint(run_id: str):
    """Fetches a specific run by ID for the complete validation workspace."""
    pipeline = get_pipeline_service()
    run = pipeline.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run


@router.get("/gaps", response_model=List[MatchedGap])
def get_candidate_gaps_endpoint(limit: int = 50):
    """Returns candidate research gaps from Phase 1 literature analysis."""
    matcher = get_theme_matcher()
    gaps = matcher._gaps[:limit]
    return [
        MatchedGap(
            gap_id=g["gap_id"],
            concept_a=g["concept_a"],
            concept_b=g["concept_b"],
            gap_score=g["gap_score"],
            semantic_similarity=g["semantic_similarity"],
            temporal_growth_avg=g.get("temporal_growth_avg"),
            matching_score=g.get("gap_score", 0.0),
            matching_explanation="Phase 1 literature gap",
            supporting_topic_ids=g.get("supporting_topic_ids", []),
            supporting_pmids_count=g.get("papers_joint", 0),
            description=g.get("description", ""),
            is_precomputed=True,
        )
        for g in gaps
    ]


@router.get("/evaluation")
def get_evaluation_endpoint():
    """Returns Phase 3 project-level empirical benchmarks."""
    return get_evaluation_benchmarks()
