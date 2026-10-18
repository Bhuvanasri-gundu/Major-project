"""
retrieval_eval.py — Rigorous Empirical IR Evaluation for Evidence Retrieval.

Calculates:
- Precision@K (K=5, 10, 20)
- Recall@K (K=5, 10, 20)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K)

Ground Truth Formulation:
-------------------------
Derived directly from empirical PubMed MeSH indexing in Phase 1 (document_concepts.csv):
- Graded Relevance 2 (High): Chunks from joint papers containing both Concept A and Concept B
- Graded Relevance 1 (Relevant): Chunks from papers containing either Concept A or Concept B
- Graded Relevance 0 (Not Relevant): Chunks containing neither concept
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QueryEvalMetrics:
    """Evaluation metrics for a single retrieval query/gap."""

    gap_id: str
    concept_a: str
    concept_b: str
    num_retrieved: int
    num_relevant_pool: int
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateRetrievalReport:
    """Aggregated retrieval benchmark across all evaluated queries."""

    num_queries: int
    k_values: List[int]
    mean_precision_at_k: Dict[int, float]
    mean_recall_at_k: Dict[int, float]
    mean_mrr: float
    mean_ndcg_at_k: Dict[int, float]
    query_details: List[QueryEvalMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "mean_precision_at_k": {str(k): round(v, 4) for k, v in self.mean_precision_at_k.items()},
            "mean_recall_at_k": {str(k): round(v, 4) for k, v in self.mean_recall_at_k.items()},
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg_at_k": {str(k): round(v, 4) for k, v in self.mean_ndcg_at_k.items()},
            "query_details": [q.to_dict() for q in self.query_details],
        }


def compute_dcg(relevance_scores: Sequence[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        # Using standard formula: rel / log2(i + 1)
        dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg(retrieved_relevances: Sequence[float], ideal_relevances: Sequence[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg(retrieved_relevances, k)
    idcg = compute_dcg(sorted(ideal_relevances, reverse=True), k)
    if idcg <= 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def evaluate_retrieval_for_gap(
    gap_record: Any,
    retrieved_items: Sequence[Any],
    doc_concepts_df: pd.DataFrame,
    k_values: Sequence[int] = (5, 10, 20),
) -> QueryEvalMetrics:
    """Evaluate retrieval results for a single gap against corpus ground truth.

    Parameters
    ----------
    gap_record : dict or object
        Gap with concept_a and concept_b.
    retrieved_items : sequence of RetrievedEvidence or dict
        Retrieved passages ordered by rank.
    doc_concepts_df : pd.DataFrame
        Empirical mapping of pmid -> concept from Phase 1.
    k_values : sequence of int
        Evaluation cutoffs.

    Returns
    -------
    QueryEvalMetrics
    """
    def _get(obj, key, default=""):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    gap_id = str(_get(gap_record, "gap_id", "GAP-UNKNOWN"))
    concept_a = str(_get(gap_record, "concept_a", "")).strip().lower()
    concept_b = str(_get(gap_record, "concept_b", "")).strip().lower()

    concept_col = "normalized_concept" if "normalized_concept" in doc_concepts_df.columns else "concept"

    # Identify true relevant PMIDs from corpus
    pmids_a = set(
        doc_concepts_df[doc_concepts_df[concept_col].str.lower() == concept_a]["pmid"].astype(str)
    )
    pmids_b = set(
        doc_concepts_df[doc_concepts_df[concept_col].str.lower() == concept_b]["pmid"].astype(str)
    )
    joint_pmids = pmids_a.intersection(pmids_b)
    relevant_pool = pmids_a.union(pmids_b)

    # Map retrieved items to graded relevance:
    # 2 = joint paper (both concepts)
    # 1 = single-concept paper (either concept)
    # 0 = neither
    retrieved_relevances: List[float] = []
    binary_relevant: List[bool] = []
    first_relevant_rank: int | None = None

    for rank_idx, item in enumerate(retrieved_items, start=1):
        pmid = str(_get(item, "source_pmid", _get(item, "pmid", ""))).strip()
        if pmid in joint_pmids:
            rel = 2.0
            is_rel = True
        elif pmid in relevant_pool:
            rel = 1.0
            is_rel = True
        else:
            rel = 0.0
            is_rel = False

        retrieved_relevances.append(rel)
        binary_relevant.append(is_rel)
        if is_rel and first_relevant_rank is None:
            first_relevant_rank = rank_idx

    # Compute P@K, R@K, nDCG@K
    precision_at_k: Dict[int, float] = {}
    recall_at_k: Dict[int, float] = {}
    ndcg_at_k: Dict[int, float] = {}

    # Construct ideal relevance list: joint papers (2.0) followed by single-concept papers (1.0)
    ideal_pool = [2.0] * len(joint_pmids) + [1.0] * (len(relevant_pool) - len(joint_pmids))

    total_relevant = len(relevant_pool)

    for k in k_values:
        sub_binary = binary_relevant[:k]
        hits = sum(1 for b in sub_binary if b)
        precision_at_k[k] = round(hits / k, 4) if k > 0 else 0.0
        recall_at_k[k] = round(hits / total_relevant, 4) if total_relevant > 0 else 0.0
        ndcg_at_k[k] = round(compute_ndcg(retrieved_relevances, ideal_pool, k), 4)

    mrr = round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0

    return QueryEvalMetrics(
        gap_id=gap_id,
        concept_a=concept_a,
        concept_b=concept_b,
        num_retrieved=len(retrieved_items),
        num_relevant_pool=total_relevant,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
    )


def evaluate_retrieval_suite(
    prepared_gaps: Sequence[Any],
    retrieved_evidence: Dict[str, Sequence[Any]],
    doc_concepts_df: pd.DataFrame,
    k_values: Sequence[int] = (5, 10, 20),
) -> AggregateRetrievalReport:
    """Execute aggregate IR evaluation across all gaps."""
    query_details: List[QueryEvalMetrics] = []

    for gap in prepared_gaps:
        gid = gap.get("gap_id") if isinstance(gap, dict) else getattr(gap, "gap_id", "")
        ev_items = retrieved_evidence.get(gid, [])
        if ev_items:
            metric = evaluate_retrieval_for_gap(gap, ev_items, doc_concepts_df, k_values=k_values)
            query_details.append(metric)

    if not query_details:
        return AggregateRetrievalReport(
            num_queries=0,
            k_values=list(k_values),
            mean_precision_at_k={k: 0.0 for k in k_values},
            mean_recall_at_k={k: 0.0 for k in k_values},
            mean_mrr=0.0,
            mean_ndcg_at_k={k: 0.0 for k in k_values},
            query_details=[],
        )

    mean_p = {
        k: float(np.mean([q.precision_at_k.get(k, 0.0) for q in query_details]))
        for k in k_values
    }
    mean_r = {
        k: float(np.mean([q.recall_at_k.get(k, 0.0) for q in query_details]))
        for k in k_values
    }
    mean_mrr = float(np.mean([q.mrr for q in query_details]))
    mean_ndcg = {
        k: float(np.mean([q.ndcg_at_k.get(k, 0.0) for q in query_details]))
        for k in k_values
    }

    return AggregateRetrievalReport(
        num_queries=len(query_details),
        k_values=list(k_values),
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_mrr=mean_mrr,
        mean_ndcg_at_k=mean_ndcg,
        query_details=query_details,
    )
