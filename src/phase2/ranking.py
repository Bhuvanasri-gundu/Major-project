"""
ranking.py — Transparent Multi-Dimensional Hypothesis Ranking Module for Phase 2.

Combines evidence verification, NLI entailment, LLM-as-a-judge scores, and
rule-based compliance checks into a unified, reproducible rank score.

Ranking Formulation:
--------------------
  RankScore = w1 * evidence_support +
              w2 * judge_quality +
              w3 * testability +
              w4 * rule_compliance -
              w5 * contradiction_penalty

Where:
  - evidence_support: verification.coverage_ratio * (nli_support_ratio if available else 0.5)
  - judge_quality: mean(judge_scores) / 5.0 (normalized to [0, 1])
  - testability: judge.testability / 5.0
  - rule_compliance: rule_checks.passed_count / total_checks
  - contradiction_penalty: nli_contradiction_ratio * 0.5
  - Default weights: w1=0.25, w2=0.30, w3=0.15, w4=0.15, w5=0.15

Quality Tiers:
  - 'high':   RankScore >= 0.70
  - 'medium': 0.45 <= RankScore < 0.70
  - 'low':    RankScore < 0.45
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w1": 0.25,  # evidence_support
    "w2": 0.30,  # judge_quality
    "w3": 0.15,  # testability
    "w4": 0.15,  # rule_compliance
    "w5": 0.15,  # contradiction_penalty
}

DEFAULT_TIER_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.45,
}


def _extract_field(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Helper to retrieve a value from a dict or an object attribute."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return default
    for k in keys:
        if hasattr(obj, k):
            val = getattr(obj, k)
            if val is not None:
                return val
    return default


def _build_index(collection: Any) -> Dict[str, Any]:
    """Index a collection of objects or dicts by hypothesis_id."""
    if collection is None:
        return {}
    if isinstance(collection, dict):
        # Could be a map of {hid: obj} or a single object dict
        if "hypothesis_id" in collection or "id" in collection:
            hid = str(_extract_field(collection, ["hypothesis_id", "id"]))
            return {hid: collection}
        return {str(k): v for k, v in collection.items()}
    if isinstance(collection, (list, tuple, set)):
        indexed = {}
        for item in collection:
            hid = _extract_field(item, ["hypothesis_id", "id"])
            if hid is not None:
                indexed[str(hid)] = item
        return indexed
    return {}


@dataclass
class RankedHypothesis:
    """Outcome of hypothesis ranking and validation synthesis.

    Attributes
    ----------
    hypothesis_id : str
        Unique identifier of the hypothesis.
    gap_id : str
        Identifier of the associated research gap.
    rank_position : int
        1-based rank position (1 = highest scoring hypothesis).
    rank_score : float
        Composite score in range [0.0, 1.0].
    evidence_support_score : float
        Grounding component: coverage * support ratio.
    judge_quality_score : float
        Mean LLM judge score normalized to [0.0, 1.0].
    testability_score : float
        Testability score from judge normalized to [0.0, 1.0].
    rule_compliance_score : float
        Ratio of rule checks passed in [0.0, 1.0].
    contradiction_penalty : float
        Penalty deducted due to contradictory evidence in [0.0, 0.5].
    validation_warnings : list of str
        Diagnostic notes regarding missing components or caveats.
    quality_tier : str
        Assigned tier: 'high', 'medium', or 'low'.
    """

    hypothesis_id: str
    gap_id: str
    rank_position: int
    rank_score: float
    evidence_support_score: float
    judge_quality_score: float
    testability_score: float
    rule_compliance_score: float
    contradiction_penalty: float
    validation_warnings: List[str]
    quality_tier: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RankedHypothesis:
        """Instantiate dataclass from dictionary."""
        return cls(
            hypothesis_id=str(data.get("hypothesis_id", "UNKNOWN")),
            gap_id=str(data.get("gap_id", "UNKNOWN")),
            rank_position=int(data.get("rank_position", 0)),
            rank_score=float(data.get("rank_score", 0.0)),
            evidence_support_score=float(data.get("evidence_support_score", 0.0)),
            judge_quality_score=float(data.get("judge_quality_score", 0.0)),
            testability_score=float(data.get("testability_score", 0.0)),
            rule_compliance_score=float(data.get("rule_compliance_score", 0.0)),
            contradiction_penalty=float(data.get("contradiction_penalty", 0.0)),
            validation_warnings=list(data.get("validation_warnings", [])),
            quality_tier=str(data.get("quality_tier", "low")),
        )


def _compute_evidence_support(
    hyp: Any,
    verif: Any,
    nli: Any,
    warnings: List[str],
) -> tuple[float, float]:
    """Compute evidence support score and contradiction penalty.

    Returns (evidence_support_score, contradiction_penalty).
    """
    # 1. Coverage ratio from verification
    coverage_ratio: float
    if verif is not None:
        cov = _extract_field(verif, ["coverage_ratio", "valid_ratio", "precision"])
        if cov is not None:
            try:
                coverage_ratio = max(0.0, min(1.0, float(cov)))
            except (ValueError, TypeError):
                coverage_ratio = 0.5
        else:
            valid_cnt = _extract_field(verif, ["valid_citations_count", "valid_count"])
            total_cnt = _extract_field(verif, ["total_citations_count", "total_count"])
            if valid_cnt is not None and total_cnt and float(total_cnt) > 0:
                coverage_ratio = float(valid_cnt) / float(total_cnt)
            else:
                coverage_ratio = 0.5
    else:
        # Heuristic from hypothesis citations
        ev_ids = _extract_field(
            hyp,
            ["supporting_evidence_ids", "evidence_ids", "pmids", "supporting_pmids"],
            [],
        )
        if ev_ids and len(ev_ids) > 0:
            coverage_ratio = 0.8
        else:
            coverage_ratio = 0.4
        warnings.append("Verification data missing; used evidence citation presence default.")

    # 2. NLI support and contradiction ratios
    nli_support_ratio: float
    contradiction_ratio: float = 0.0

    if nli is not None:
        sup = _extract_field(nli, ["support_ratio", "entailment_ratio"])
        contra = _extract_field(nli, ["contradiction_ratio", "contradict_ratio"])

        if sup is not None:
            try:
                nli_support_ratio = max(0.0, min(1.0, float(sup)))
            except (ValueError, TypeError):
                nli_support_ratio = 0.5
        else:
            entail_cnt = _extract_field(nli, ["entailment_count", "supported_count"], 0)
            total_cnt = _extract_field(nli, ["total_pairs", "total_checked"], 0)
            nli_support_ratio = (float(entail_cnt) / float(total_cnt)) if total_cnt and float(total_cnt) > 0 else 0.5

        if contra is not None:
            try:
                contradiction_ratio = max(0.0, min(1.0, float(contra)))
            except (ValueError, TypeError):
                contradiction_ratio = 0.0
        else:
            contra_cnt = _extract_field(nli, ["contradiction_count", "conflict_count"], 0)
            total_cnt = _extract_field(nli, ["total_pairs", "total_checked"], 0)
            contradiction_ratio = (float(contra_cnt) / float(total_cnt)) if total_cnt and float(total_cnt) > 0 else 0.0
    else:
        nli_support_ratio = 0.5
        contradiction_ratio = 0.0
        warnings.append("NLI validation data missing; assumed neutral NLI support ratio (0.5).")

    evidence_support = float(coverage_ratio * nli_support_ratio)
    contradiction_penalty = float(contradiction_ratio * 0.5)

    return evidence_support, contradiction_penalty


def _compute_judge_scores(
    judge: Any,
    warnings: List[str],
) -> tuple[float, float]:
    """Compute normalized judge quality and testability scores.

    Returns (judge_quality_score, testability_score) in [0.0, 1.0].
    """
    if judge is not None:
        if hasattr(judge, "mean_score") and callable(judge.mean_score):
            m_score = judge.mean_score()
        else:
            m_score = _extract_field(judge, ["mean_score", "average_score"])
            if m_score is None:
                dim_keys = [
                    "relevance_to_gap",
                    "clarity",
                    "logical_consistency",
                    "testability",
                    "evidence_alignment",
                    "novelty_potential",
                ]
                scores = []
                for k in dim_keys:
                    val = _extract_field(judge, [k])
                    if val is not None:
                        try:
                            scores.append(float(val))
                        except (ValueError, TypeError):
                            pass
                m_score = sum(scores) / len(scores) if scores else 3.0

        test_val = _extract_field(judge, ["testability"], 3.0)
        try:
            testability_score = float(test_val) / 5.0
        except (ValueError, TypeError):
            testability_score = 0.6

        judge_quality_score = float(m_score) / 5.0
    else:
        judge_quality_score = 0.6  # baseline 3.0 / 5.0
        testability_score = 0.6
        warnings.append("LLM Judge evaluation missing; assigned neutral baseline scores (0.6).")

    # Clamping
    judge_quality_score = max(0.0, min(1.0, judge_quality_score))
    testability_score = max(0.0, min(1.0, testability_score))

    return judge_quality_score, testability_score


def _compute_rule_compliance(
    rule: Any,
    warnings: List[str],
) -> float:
    """Compute rule compliance score as passed_checks / total_checks."""
    if rule is not None:
        passed = _extract_field(rule, ["passed_count"])
        failed = _extract_field(rule, ["failed_count"], 0)
        warns = _extract_field(rule, ["warning_count"], 0)

        if passed is not None:
            try:
                p = float(passed)
                f = float(failed)
                w = float(warns)
                total = p + f + w
                if total > 0:
                    return max(0.0, min(1.0, p / total))
            except (ValueError, TypeError):
                pass

        # Check list of check details
        checks = _extract_field(rule, ["checks"], [])
        if checks and len(checks) > 0:
            pass_c = 0
            for c in checks:
                status = str(_extract_field(c, ["status"], "fail")).lower()
                if status == "pass":
                    pass_c += 1
            return max(0.0, min(1.0, pass_c / len(checks)))

    warnings.append("Rule checks data missing; assigned baseline compliance default (0.8).")
    return 0.8


def rank_hypotheses(
    hypotheses: Sequence[Any],
    verifications: Optional[Any] = None,
    nli_results: Optional[Any] = None,
    judge_results: Optional[Any] = None,
    rule_results: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
    tier_thresholds: Optional[Dict[str, float]] = None,
) -> List[RankedHypothesis]:
    """Rank hypotheses using the transparent multi-dimensional formula.

    Formula:
      RankScore = w1 * evidence_support +
                  w2 * judge_quality +
                  w3 * testability +
                  w4 * rule_compliance -
                  w5 * contradiction_penalty

    Parameters
    ----------
    hypotheses : sequence of dict or object
        Hypotheses to rank.
    verifications : dict, list, or None
        Evidence verification results keyed by hypothesis_id or list.
    nli_results : dict, list, or None
        NLI entailment results keyed by hypothesis_id or list.
    judge_results : dict, list, or None
        LLM Judge results keyed by hypothesis_id or list.
    rule_results : dict, list, or None
        Rule check results keyed by hypothesis_id or list.
    weights : dict, optional
        Custom weights for w1-w5.
    tier_thresholds : dict, optional
        Thresholds for 'high' and 'medium' tiers.

    Returns
    -------
    list of RankedHypothesis
        Ranked hypotheses sorted descending by rank_score with rank_position 1..N.
    """
    if not hypotheses:
        logger.warning("Empty hypotheses list supplied to rank_hypotheses.")
        return []

    # Config resolution
    cfg = load_config()
    phase2_cfg = cfg.get("phase2", {}) if isinstance(cfg, dict) else {}
    ranking_cfg = phase2_cfg.get("ranking", cfg.get("ranking", {}))
    w = dict(DEFAULT_WEIGHTS)
    
    cfg_weights = ranking_cfg.get("weights", {})
    if isinstance(cfg_weights, dict):
        weight_name_map = {
            "evidence_support": "w1",
            "judge_quality": "w2",
            "testability": "w3",
            "rule_compliance": "w4",
            "contradiction_penalty": "w5",
        }
        for k, val in cfg_weights.items():
            mapped_key = weight_name_map.get(k, k)
            try:
                w[mapped_key] = float(val)
            except (ValueError, TypeError):
                pass

    if weights:
        weight_name_map = {
            "evidence_support": "w1",
            "judge_quality": "w2",
            "testability": "w3",
            "rule_compliance": "w4",
            "contradiction_penalty": "w5",
        }
        for k, val in weights.items():
            mapped_key = weight_name_map.get(k, k)
            try:
                w[mapped_key] = float(val)
            except (ValueError, TypeError):
                pass

    thresholds = dict(DEFAULT_TIER_THRESHOLDS)
    cfg_tiers = ranking_cfg.get("quality_tiers", ranking_cfg.get("tier_thresholds", {}))
    if isinstance(cfg_tiers, dict):
        for tier_k in ["high", "medium"]:
            if tier_k in cfg_tiers:
                try:
                    thresholds[tier_k] = float(cfg_tiers[tier_k])
                except (ValueError, TypeError):
                    pass
    if tier_thresholds:
        thresholds.update(tier_thresholds)

    # Build indexes for O(1) lookups
    verif_idx = _build_index(verifications)
    nli_idx = _build_index(nli_results)
    judge_idx = _build_index(judge_results)
    rule_idx = _build_index(rule_results)

    ranked_items: List[RankedHypothesis] = []

    for hyp in hypotheses:
        hid = str(_extract_field(hyp, ["hypothesis_id", "id"], "HYP-UNKNOWN"))
        gid = str(_extract_field(hyp, ["gap_id", "candidate_gap_id"], "GAP-UNKNOWN"))
        warnings: List[str] = []

        # Retrieve validation artifacts for this hypothesis
        verif = verif_idx.get(hid)
        nli = nli_idx.get(hid)
        judge = judge_idx.get(hid)
        rule = rule_idx.get(hid)

        # 1. Evidence Support & Contradiction Penalty
        ev_support, contra_penalty = _compute_evidence_support(hyp, verif, nli, warnings)

        # 2. Judge Quality & Testability
        judge_quality, testability = _compute_judge_scores(judge, warnings)

        # 3. Rule Compliance
        rule_compliance = _compute_rule_compliance(rule, warnings)

        # 4. Composite Rank Score
        raw_score = (
            w["w1"] * ev_support
            + w["w2"] * judge_quality
            + w["w3"] * testability
            + w["w4"] * rule_compliance
            - w["w5"] * contra_penalty
        )
        rank_score = round(max(0.0, min(1.0, raw_score)), 4)

        # 5. Quality Tier assignment
        if rank_score >= thresholds["high"]:
            tier = "high"
        elif rank_score >= thresholds["medium"]:
            tier = "medium"
        else:
            tier = "low"

        ranked_items.append(
            RankedHypothesis(
                hypothesis_id=hid,
                gap_id=gid,
                rank_position=0,  # will be assigned post-sorting
                rank_score=rank_score,
                evidence_support_score=round(ev_support, 4),
                judge_quality_score=round(judge_quality, 4),
                testability_score=round(testability, 4),
                rule_compliance_score=round(rule_compliance, 4),
                contradiction_penalty=round(contra_penalty, 4),
                validation_warnings=warnings,
                quality_tier=tier,
            )
        )

    # Sort descending by rank_score, breaking ties with judge_quality then evidence_support
    ranked_items.sort(
        key=lambda x: (x.rank_score, x.judge_quality_score, x.evidence_support_score),
        reverse=True,
    )

    # Assign 1-based rank positions
    for pos, item in enumerate(ranked_items, start=1):
        item.rank_position = pos

    logger.info(
        "Ranked %d hypotheses (Top score: %.4f, Lowest: %.4f)",
        len(ranked_items),
        ranked_items[0].rank_score if ranked_items else 0.0,
        ranked_items[-1].rank_score if ranked_items else 0.0,
    )

    return ranked_items


def apply_ranking(
    hypotheses: Sequence[Any],
    rankings: Union[Sequence[RankedHypothesis], Dict[str, RankedHypothesis]],
) -> List[Dict[str, Any]]:
    """Merge ranking attributes into hypothesis records and return in rank order.

    Parameters
    ----------
    hypotheses : sequence of dict or object
        Original hypothesis records.
    rankings : list or dict of RankedHypothesis
        Computed ranking objects.

    Returns
    -------
    list of dict
        Enriched hypothesis records sorted ascending by rank_position.
    """
    ranking_map: Dict[str, RankedHypothesis] = {}
    if isinstance(rankings, dict):
        for k, v in rankings.items():
            ranking_map[str(k)] = v if isinstance(v, RankedHypothesis) else RankedHypothesis.from_dict(v)
    else:
        for r in rankings:
            ranking_map[r.hypothesis_id] = r

    enriched: List[Dict[str, Any]] = []

    for hyp in hypotheses:
        if isinstance(hyp, dict):
            item = dict(hyp)
        elif hasattr(hyp, "to_dict") and callable(hyp.to_dict):
            item = hyp.to_dict()
        else:
            item = {k: getattr(hyp, k) for k in dir(hyp) if not k.startswith("_")}

        hid = str(_extract_field(item, ["hypothesis_id", "id"], "HYP-UNKNOWN"))
        r = ranking_map.get(hid)

        if r:
            item["rank_position"] = r.rank_position
            item["rank_score"] = r.rank_score
            item["quality_tier"] = r.quality_tier
            item["evidence_support_score"] = r.evidence_support_score
            item["judge_quality_score"] = r.judge_quality_score
            item["testability_score"] = r.testability_score
            item["rule_compliance_score"] = r.rule_compliance_score
            item["contradiction_penalty"] = r.contradiction_penalty
            item["validation_warnings"] = r.validation_warnings
        else:
            item["rank_position"] = 999999
            item["rank_score"] = 0.0
            item["quality_tier"] = "unranked"

        enriched.append(item)

    enriched.sort(key=lambda x: x.get("rank_position", 999999))
    return enriched


def get_ranking_summary(
    rankings: Sequence[Union[RankedHypothesis, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Calculate descriptive summary statistics for a collection of rankings.

    Parameters
    ----------
    rankings : sequence of RankedHypothesis or dict
        Ranked hypothesis outputs.

    Returns
    -------
    dict
        Descriptive summary containing counts, tier breakdown, score metrics.
    """
    if not rankings:
        return {
            "total_ranked": 0,
            "quality_tiers": {"high": 0, "medium": 0, "low": 0},
            "score_mean": 0.0,
            "score_median": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_std": 0.0,
            "top_ranked_id": None,
        }

    scores: List[float] = []
    tiers: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    warning_count = 0
    top_id: Optional[str] = None
    min_rank = 999999

    for r in rankings:
        score = float(_extract_field(r, ["rank_score"], 0.0))
        tier = str(_extract_field(r, ["quality_tier"], "low")).lower()
        hid = str(_extract_field(r, ["hypothesis_id", "id"], "UNKNOWN"))
        pos = int(_extract_field(r, ["rank_position"], 999999))
        warns = _extract_field(r, ["validation_warnings"], [])

        scores.append(score)
        tiers[tier] = tiers.get(tier, 0) + 1
        warning_count += len(warns)

        if pos < min_rank:
            min_rank = pos
            top_id = hid

    n = len(scores)
    mean_val = sum(scores) / n
    sorted_scores = sorted(scores)
    median_val = (
        sorted_scores[n // 2]
        if n % 2 != 0
        else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2.0
    )
    variance = sum((x - mean_val) ** 2 for x in scores) / n if n > 0 else 0.0
    std_val = math.sqrt(variance)

    return {
        "total_ranked": n,
        "quality_tiers": tiers,
        "score_mean": round(mean_val, 4),
        "score_median": round(median_val, 4),
        "score_min": round(min(scores), 4),
        "score_max": round(max(scores), 4),
        "score_std": round(std_val, 4),
        "total_warnings": warning_count,
        "top_ranked_id": top_id,
    }


def save_rankings(
    rankings: Union[Sequence[RankedHypothesis], Dict[str, RankedHypothesis]],
    output_path: Union[str, Path],
) -> Path:
    """Save ranked hypotheses to a JSON file.

    Parameters
    ----------
    rankings : sequence or dict of RankedHypothesis
        Ranking objects.
    output_path : str or Path
        Target file path.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(rankings, dict):
        payload = [
            v.to_dict() if hasattr(v, "to_dict") else v
            for v in rankings.values()
        ]
    else:
        payload = [
            r.to_dict() if hasattr(r, "to_dict") else r
            for r in rankings
        ]

    # Ensure list is sorted by rank_position
    payload.sort(key=lambda x: x.get("rank_position", 0))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d hypothesis rankings to %s", len(payload), path)
    return path
