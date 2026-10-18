"""
grounding_eval.py — Evidence Grounding & Traceability Metrics.

Evaluates:
- Valid evidence-reference rate (fraction of cited IDs actually present in candidate pool)
- Traceability rate (hypotheses with 100% valid, traceable literature citations)
- Evidence coverage ratio across retrieved candidate pool
- NLI entailment / support vs contradiction vs neutral distribution
- Flagged rate (hypotheses flagged for insufficient evidence or ungrounded variables)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceGroundingReport:
    """Summary of evidence grounding evaluation."""

    total_hypotheses: int
    total_citations: int
    valid_citations: int
    invalid_citations: int
    citation_validity_rate: float
    fully_traceable_hypotheses: int
    traceability_rate: float
    mean_evidence_coverage: float
    nli_distribution: Dict[str, int]
    nli_percentages: Dict[str, float]
    mean_nli_alignment_score: float
    flagged_insufficient_support: int
    flagged_insufficient_support_rate: float
    fabricated_citation_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_hypotheses": self.total_hypotheses,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "citation_validity_rate": round(self.citation_validity_rate, 4),
            "fully_traceable_hypotheses": self.fully_traceable_hypotheses,
            "traceability_rate": round(self.traceability_rate, 4),
            "mean_evidence_coverage": round(self.mean_evidence_coverage, 4),
            "nli_distribution": self.nli_distribution,
            "nli_percentages": {k: round(v, 2) for k, v in self.nli_percentages.items()},
            "mean_nli_alignment_score": round(self.mean_nli_alignment_score, 4),
            "flagged_insufficient_support": self.flagged_insufficient_support,
            "flagged_insufficient_support_rate": round(self.flagged_insufficient_support_rate, 4),
            "fabricated_citation_count": self.fabricated_citation_count,
        }


def evaluate_grounding(
    hypotheses: Sequence[Any],
    verifications: Dict[str, Any],
    nli_results: Dict[str, Any],
) -> EvidenceGroundingReport:
    """Compute evidence grounding and citation fidelity metrics."""
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    total_hyps = len(hypotheses)
    if total_hyps == 0:
        return EvidenceGroundingReport(
            total_hypotheses=0,
            total_citations=0,
            valid_citations=0,
            invalid_citations=0,
            citation_validity_rate=0.0,
            fully_traceable_hypotheses=0,
            traceability_rate=0.0,
            mean_evidence_coverage=0.0,
            nli_distribution={"supported": 0, "neutral": 0, "contradicted": 0, "mixed": 0},
            nli_percentages={"supported": 0.0, "neutral": 0.0, "contradicted": 0.0, "mixed": 0.0},
            mean_nli_alignment_score=0.0,
            flagged_insufficient_support=0,
            flagged_insufficient_support_rate=0.0,
            fabricated_citation_count=0,
        )

    total_citations = 0
    valid_citations = 0
    invalid_citations = 0
    fully_traceable = 0
    coverages: List[float] = []
    fabricated_count = 0
    flagged_insufficient = 0

    nli_counts = {"supported": 0, "neutral": 0, "contradicted": 0, "mixed": 0}
    nli_scores: List[float] = []

    for hyp in hypotheses:
        hid = str(_get(hyp, "hypothesis_id", ""))
        verif = verifications.get(hid, {})

        ref_res = _get(verif, "reference_result", {})
        valid_ids = _get(ref_res, "valid_ids") or _get(verif, "valid_ids", []) or []
        invalid_ids = _get(ref_res, "invalid_ids") or _get(verif, "invalid_ids", []) or []
        fab = int(_get(ref_res, "fabricated_count") or _get(verif, "fabricated_count", 0) or 0)
        cov = float(_get(verif, "evidence_coverage") or _get(verif, "coverage_ratio", 0.0) or 0.0)
        status = str(_get(ref_res, "verification_status") or _get(verif, "verification_status", "passed"))

        cited_cnt = len(valid_ids) + len(invalid_ids)
        total_citations += cited_cnt
        valid_citations += len(valid_ids)
        invalid_citations += len(invalid_ids)
        fabricated_count += fab
        coverages.append(cov)

        if len(invalid_ids) == 0 and len(valid_ids) > 0:
            fully_traceable += 1

        if status == "failed" or cov < 0.05 or len(valid_ids) == 0:
            flagged_insufficient += 1

        # NLI metrics
        nli = nli_results.get(hid, {})
        align = str(_get(nli, "overall_alignment", "neutral")).lower()
        if align in nli_counts:
            nli_counts[align] += 1
        else:
            nli_counts["neutral"] += 1

        score = _get(nli, "alignment_score", None)
        if score is not None:
            nli_scores.append(float(score))

    validity_rate = (valid_citations / total_citations) if total_citations > 0 else 1.0
    traceability_rate = fully_traceable / total_hyps
    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    mean_nli_score = float(np.mean(nli_scores)) if nli_scores else 0.5

    nli_pcts = {
        k: (v / total_hyps * 100.0) for k, v in nli_counts.items()
    }

    return EvidenceGroundingReport(
        total_hypotheses=total_hyps,
        total_citations=total_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        citation_validity_rate=validity_rate,
        fully_traceable_hypotheses=fully_traceable,
        traceability_rate=traceability_rate,
        mean_evidence_coverage=mean_cov,
        nli_distribution=nli_counts,
        nli_percentages=nli_pcts,
        mean_nli_alignment_score=mean_nli_score,
        flagged_insufficient_support=flagged_insufficient,
        flagged_insufficient_support_rate=flagged_insufficient / total_hyps,
        fabricated_citation_count=fabricated_count,
    )
