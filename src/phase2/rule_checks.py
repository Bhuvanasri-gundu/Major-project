"""
rule_checks.py — Rule-Based Validation Checks for Generated Hypotheses.

Performs deterministic, rule-based scientific quality checks on candidate
mHealth hypotheses to verify:
  1. Variable completeness (Independent Variable and Dependent Variable present)
  2. Target population specification
  3. Falsifiability and avoidance of purely vague/hedging claims
  4. Appropriate causal language alignment with proposed relationship type
  5. Evidence references presence
  6. Evidence pool coverage and verification
  7. Specificity and measurable outcome terminology
  8. Appropriate hypothesis length (20-200 words)
  9. Explicit acknowledgment of contradictory evidence in limitations

All checks produce structured CheckDetail items and aggregate into a
RuleCheckResult with pass, warn, or fail status.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from ..config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)


# Standard regexes and term sets
_CAUSAL_VERBS = re.compile(
    r"\b(causes?|leads?\s+to|results?\s+in|triggers?|induces?|determines?|produces?)\b",
    re.IGNORECASE,
)
_VAGUE_HEDGES = re.compile(
    r"\b(it\s+is\s+possible\s+that|could\s+potentially|may\s+perhaps|can\s+conceivably|might\s+possibly)\b",
    re.IGNORECASE,
)
_DIRECTIONAL_VERBS = re.compile(
    r"\b(increases?|decreases?|improves?|reduces?|enhances?|diminishes?|correlates?|correlated|associates?|associated|predicts?|mitigates?|augments?|lowers?|elevates?)\b",
    re.IGNORECASE,
)
_GENERIC_POPULATION = {
    "people",
    "patients",
    "humans",
    "individuals",
    "general population",
    "everyone",
    "users",
    "subjects",
    "participants",
    "general",
    "adults",
}
_SPECIFICITY_KEYWORDS = {
    "adherence",
    "compliance",
    "frequency",
    "rate",
    "percentage",
    "score",
    "reduction",
    "increase",
    "decrease",
    "improvement",
    "duration",
    "latency",
    "retention",
    "incidence",
    "prevalence",
    "mortality",
    "morbidity",
    "hba1c",
    "blood pressure",
    "systolic",
    "diastolic",
    "heart rate",
    "hrv",
    "steps",
    "physical activity",
    "biomarker",
    "glycemic",
    "glucose",
    "symptom",
    "anxiety",
    "depression",
    "quality of life",
    "qol",
    "phq-9",
    "gad-7",
    "wearable",
    "sensor",
    "sensor-derived",
    "readmission",
    "emergency department",
    "engagement",
    "daily",
    "weekly",
    "monthly",
}


def _extract_field(obj: Any, keys: Sequence[str], default: Any = "") -> Any:
    """Helper to retrieve a field from a dict or an object."""
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


def _extract_evidence_ids(hypothesis: Any, keys: Sequence[str]) -> List[str]:
    """Helper to extract evidence IDs as a list of strings."""
    raw = _extract_field(hypothesis, keys, [])
    if isinstance(raw, str):
        # Could be semicolon or comma separated string
        parts = [p.strip() for p in re.split(r"[;, ]+", raw) if p.strip()]
        return parts
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


@dataclass
class CheckDetail:
    """Outcome of an individual rule check.

    Attributes
    ----------
    check_name : str
        Name of the validation check.
    status : str
        Outcome: 'pass', 'warning', or 'fail'.
    message : str
        Explanatory message summarizing findings.
    """

    check_name: str
    status: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert check detail to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckDetail:
        """Construct from dict."""
        return cls(
            check_name=str(data.get("check_name", "unknown_check")),
            status=str(data.get("status", "fail")),
            message=str(data.get("message", "")),
        )


@dataclass
class RuleCheckResult:
    """Aggregated validation result for all rule-based checks on a hypothesis.

    Attributes
    ----------
    hypothesis_id : str
        Unique identifier of the validated hypothesis.
    checks : list of CheckDetail
        Individual check details.
    passed_count : int
        Number of checks that passed.
    failed_count : int
        Number of checks that failed.
    warning_count : int
        Number of checks that produced warnings.
    overall_status : str
        Aggregate status: 'pass' (all pass), 'warn' (no failures, >=1 warning), or 'fail' (>=1 fail).
    """

    hypothesis_id: str
    checks: List[CheckDetail]
    passed_count: int
    failed_count: int
    warning_count: int
    overall_status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "checks": [c.to_dict() for c in self.checks],
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "overall_status": self.overall_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuleCheckResult:
        """Construct instance from dictionary."""
        checks_data = data.get("checks", [])
        checks = [
            CheckDetail.from_dict(c) if isinstance(c, dict) else c
            for c in checks_data
        ]
        return cls(
            hypothesis_id=str(data.get("hypothesis_id", "UNKNOWN")),
            checks=checks,
            passed_count=int(data.get("passed_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            warning_count=int(data.get("warning_count", 0)),
            overall_status=str(data.get("overall_status", "fail")),
        )


def check_has_variables(hypothesis: Any) -> CheckDetail:
    """Check that both independent variable (IV) and dependent variable (DV) are present.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Pass if both IV and DV are specified, non-empty, and distinct.
    """
    iv = str(_extract_field(hypothesis, ["independent_variable", "iv", "intervention"], "")).strip()
    dv = str(_extract_field(hypothesis, ["dependent_variable", "dv", "outcome"], "")).strip()

    if not iv and not dv:
        return CheckDetail(
            check_name="has_variables",
            status="fail",
            message="Both Independent Variable (IV) and Dependent Variable (DV) are missing or empty.",
        )
    if not iv:
        return CheckDetail(
            check_name="has_variables",
            status="fail",
            message=f"Independent Variable (IV) is missing or empty. (DV='{dv}')",
        )
    if not dv:
        return CheckDetail(
            check_name="has_variables",
            status="fail",
            message=f"Dependent Variable (DV) is missing or empty. (IV='{iv}')",
        )
    if iv.lower() == dv.lower():
        return CheckDetail(
            check_name="has_variables",
            status="fail",
            message=f"Independent and Dependent variables cannot be identical ('{iv}').",
        )

    return CheckDetail(
        check_name="has_variables",
        status="pass",
        message=f"Variables specified: IV='{iv}', DV='{dv}'.",
    )


def check_has_population(hypothesis: Any) -> CheckDetail:
    """Check that target population is specified and non-generic.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Pass if target population is well-defined; warning if overly generic; fail if absent.
    """
    pop = str(_extract_field(hypothesis, ["target_population", "population", "cohort"], "")).strip()

    if not pop:
        return CheckDetail(
            check_name="has_population",
            status="fail",
            message="Target population is missing or empty.",
        )
    if len(pop) < 3:
        return CheckDetail(
            check_name="has_population",
            status="fail",
            message=f"Target population specification '{pop}' is too short.",
        )

    if pop.lower() in _GENERIC_POPULATION:
        return CheckDetail(
            check_name="has_population",
            status="warning",
            message=(
                f"Target population '{pop}' is overly broad/generic; "
                "recommend specifying clinical, demographic, or geographic qualifiers."
            ),
        )

    return CheckDetail(
        check_name="has_population",
        status="pass",
        message=f"Target population specified: '{pop}'.",
    )


def check_falsifiability(hypothesis: Any) -> CheckDetail:
    """Check that hypothesis asserts an empirical claim rather than vague, unfalsifiable hedges.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Pass if directional claim is made; warning or fail if purely speculative hedging.
    """
    stmt = str(
        _extract_field(
            hypothesis,
            ["hypothesis", "hypothesis_text", "hypothesis_statement", "text", "statement"],
            "",
        )
    ).strip()

    if not stmt:
        return CheckDetail(
            check_name="falsifiability",
            status="fail",
            message="Hypothesis statement is empty; cannot evaluate falsifiability.",
        )

    has_vague_hedge = bool(_VAGUE_HEDGES.search(stmt))
    has_directional = bool(_DIRECTIONAL_VERBS.search(stmt))

    # Also detect isolated 'may' without directional outcome
    has_lone_may = bool(re.search(r"\bmay\s+(?:have\s+an\s+effect|impact|influence)\b", stmt, re.I))

    if has_vague_hedge and not has_directional:
        return CheckDetail(
            check_name="falsifiability",
            status="fail",
            message=(
                "Hypothesis statement uses unfalsifiable speculative hedges "
                "('it is possible that', 'could potentially') without specifying "
                "a testable directional effect."
            ),
        )

    if has_lone_may or (has_vague_hedge and has_directional):
        return CheckDetail(
            check_name="falsifiability",
            status="warning",
            message=(
                "Hypothesis contains hedging terminology ('may', 'could potentially'). "
                "Consider phrasing as an explicit directional scientific claim."
            ),
        )

    if not has_directional:
        return CheckDetail(
            check_name="falsifiability",
            status="warning",
            message=(
                "Hypothesis does not contain standard directional verbs "
                "(e.g., 'increases', 'reduces', 'improves', 'is associated with')."
            ),
        )

    return CheckDetail(
        check_name="falsifiability",
        status="pass",
        message="Hypothesis asserts a clear, testable directional relationship.",
    )


def check_causal_language(hypothesis: Any) -> CheckDetail:
    """Flag unsupported causal language when proposed relationship is correlational or associational.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Warning if causal language is used with non-causal relationship type; pass otherwise.
    """
    stmt = str(
        _extract_field(
            hypothesis,
            ["hypothesis", "hypothesis_text", "hypothesis_statement", "text", "statement"],
            "",
        )
    ).strip()
    rel = str(
        _extract_field(
            hypothesis,
            ["proposed_relationship", "relationship_type", "relationship"],
            "association",
        )
    ).strip().lower()

    causal_match = _CAUSAL_VERBS.search(stmt)

    associational_terms = {"association", "correlation", "correlational", "observational", "associative"}
    is_associational = any(term in rel for term in associational_terms)

    if is_associational and causal_match:
        matched_verb = causal_match.group(0)
        return CheckDetail(
            check_name="causal_language",
            status="warning",
            message=(
                f"Unsupported causal claim detected ('{matched_verb}') "
                f"while proposed relationship is designated as '{rel}'. "
                "Suggest using associational phrasing ('is associated with', 'correlates with')."
            ),
        )

    return CheckDetail(
        check_name="causal_language",
        status="pass",
        message=f"Language alignment verified for relationship type '{rel}'.",
    )


def check_evidence_references(hypothesis: Any) -> CheckDetail:
    """Check that supporting evidence references are present and non-empty.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Pass if supporting PMIDs/evidence IDs are attached; fail otherwise.
    """
    ev_ids = _extract_evidence_ids(
        hypothesis,
        ["supporting_evidence_ids", "evidence_ids", "pmids", "supporting_pmids"],
    )

    if not ev_ids:
        return CheckDetail(
            check_name="evidence_references",
            status="fail",
            message="Hypothesis does not reference any supporting evidence PMIDs/IDs.",
        )

    return CheckDetail(
        check_name="evidence_references",
        status="pass",
        message=f"Hypothesis references {len(ev_ids)} supporting evidence items: {', '.join(ev_ids[:5])}{'...' if len(ev_ids) > 5 else ''}.",
    )


def check_evidence_coverage(
    hypothesis: Any,
    evidence_pool: Optional[Union[Sequence[Any], Dict[str, Any], Set[str]]] = None,
) -> CheckDetail:
    """Verify that cited evidence references actually exist in the retrieved evidence pool.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.
    evidence_pool : list, dict, or set, optional
        Retrieved evidence pool (e.g., set/list of PMIDs, or dict mapping PMID -> document).

    Returns
    -------
    CheckDetail
        Pass if >= 1 cited ID exists in pool (or 100%); warning if partial; fail if 0% or pool missing.
    """
    if evidence_pool is None:
        return CheckDetail(
            check_name="evidence_coverage",
            status="pass",
            message="Evidence pool was not provided; skipped pool coverage verification.",
        )

    cited_ids = set(
        _extract_evidence_ids(
            hypothesis,
            ["supporting_evidence_ids", "evidence_ids", "pmids", "supporting_pmids"],
        )
    )

    if not cited_ids:
        return CheckDetail(
            check_name="evidence_coverage",
            status="fail",
            message="No cited evidence IDs found to match against evidence pool.",
        )

    # Resolve pool IDs into a set of normalized string keys
    pool_ids: Set[str] = set()
    if isinstance(evidence_pool, dict):
        for k, v in evidence_pool.items():
            pool_ids.add(str(k).strip())
            if isinstance(v, dict) and "pmid" in v:
                pool_ids.add(str(v["pmid"]).strip())
    elif isinstance(evidence_pool, (list, tuple, set)):
        for item in evidence_pool:
            if isinstance(item, dict):
                cid = item.get("chunk_id", item.get("pmid", item.get("id")))
                if cid:
                    pool_ids.add(str(cid).strip())
                pmid = item.get("pmid")
                if pmid:
                    pool_ids.add(str(pmid).strip())
            else:
                cid = getattr(item, "chunk_id", getattr(item, "pmid", str(item)))
                pool_ids.add(str(cid).strip())

    if not pool_ids:
        return CheckDetail(
            check_name="evidence_coverage",
            status="warning",
            message="Evidence pool provided was empty; unable to confirm citation grounding.",
        )

    valid_citations = cited_ids.intersection(pool_ids)
    coverage_ratio = len(valid_citations) / len(cited_ids)

    if len(valid_citations) == 0:
        return CheckDetail(
            check_name="evidence_coverage",
            status="fail",
            message=(
                f"None of the {len(cited_ids)} cited evidence IDs {list(cited_ids)[:3]} "
                f"were found in the retrieved evidence pool ({len(pool_ids)} available)."
            ),
        )

    if coverage_ratio < 1.0:
        return CheckDetail(
            check_name="evidence_coverage",
            status="warning",
            message=(
                f"Partial evidence grounding: {len(valid_citations)} of {len(cited_ids)} "
                f"({coverage_ratio * 100:.1f}%) cited IDs confirmed in evidence pool."
            ),
        )

    return CheckDetail(
        check_name="evidence_coverage",
        status="pass",
        message=f"All {len(cited_ids)} cited evidence IDs confirmed in the retrieved evidence pool.",
    )


def check_specificity(hypothesis: Any) -> CheckDetail:
    """Check that hypothesis contains measurable, operationalizable mHealth language.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Pass if measurable keywords/metrics are detected; fail otherwise.
    """
    stmt = str(
        _extract_field(
            hypothesis,
            ["hypothesis", "hypothesis_text", "hypothesis_statement", "text", "statement"],
            "",
        )
    ).lower()
    dv = str(_extract_field(hypothesis, ["dependent_variable", "dv", "outcome"], "")).lower()
    mech = str(_extract_field(hypothesis, ["proposed_mechanism", "mechanism"], "")).lower()

    combined_text = f"{stmt} {dv} {mech}"

    found_keywords = [kw for kw in _SPECIFICITY_KEYWORDS if kw in combined_text]

    if not found_keywords:
        return CheckDetail(
            check_name="specificity",
            status="fail",
            message=(
                "Hypothesis lacks measurable or operational terminology "
                "(no quantitative endpoints, clinical outcomes, or sensor metrics detected)."
            ),
        )

    if len(found_keywords) == 1:
        return CheckDetail(
            check_name="specificity",
            status="pass",
            message=f"Moderate specificity: contains measurable term '{found_keywords[0]}'.",
        )

    return CheckDetail(
        check_name="specificity",
        status="pass",
        message=f"High specificity: contains {len(found_keywords)} measurable terms ({', '.join(found_keywords[:4])}).",
    )


def check_hypothesis_length(
    hypothesis: Any,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
) -> CheckDetail:
    """Verify that hypothesis statement length is within acceptable bounds (default 20-200 words).

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.
    min_words : int, optional
        Minimum acceptable word count (defaults to config or 20).
    max_words : int, optional
        Maximum acceptable word count (defaults to config or 200).

    Returns
    -------
    CheckDetail
        Pass if min_words <= word_count <= max_words; fail if too short or too long.
    """
    if min_words is None or max_words is None:
        cfg = load_config()
        phase2_cfg = cfg.get("phase2", {}) if isinstance(cfg, dict) else {}
        rc_cfg = phase2_cfg.get("rule_checks", cfg.get("rule_checks", {}))
        if min_words is None:
            min_words = int(rc_cfg.get("min_hypothesis_words", 20))
        if max_words is None:
            max_words = int(rc_cfg.get("max_hypothesis_words", 200))

    stmt = str(
        _extract_field(
            hypothesis,
            ["hypothesis", "hypothesis_text", "hypothesis_statement", "text", "statement"],
            "",
        )
    ).strip()

    if not stmt:
        return CheckDetail(
            check_name="hypothesis_length",
            status="fail",
            message="Hypothesis statement is empty (0 words).",
        )

    words = stmt.split()
    word_count = len(words)

    if word_count < min_words:
        return CheckDetail(
            check_name="hypothesis_length",
            status="fail",
            message=(
                f"Hypothesis is too short ({word_count} words < {min_words} threshold). "
                "Insufficient detail to specify intervention, target cohort, and outcome."
            ),
        )

    if word_count > max_words:
        return CheckDetail(
            check_name="hypothesis_length",
            status="fail",
            message=(
                f"Hypothesis is excessively long ({word_count} words > {max_words} threshold). "
                "Needs concise reformulation to preserve clarity."
            ),
        )

    return CheckDetail(
        check_name="hypothesis_length",
        status="pass",
        message=f"Hypothesis length is optimal ({word_count} words, within {min_words}-{max_words} threshold).",
    )


def check_conflicting_evidence_acknowledged(hypothesis: Any) -> CheckDetail:
    """Check that contradictory or conflicting evidence, if present, is explicitly acknowledged.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis record.

    Returns
    -------
    CheckDetail
        Pass if no conflict exists, or if conflict is acknowledged in limitations; warning if unacknowledged.
    """
    contra_ids = _extract_evidence_ids(
        hypothesis,
        [
            "contradictory_evidence_ids",
            "conflicting_evidence_ids",
            "counter_evidence_ids",
            "contradictory_pmids",
        ],
    )

    if not contra_ids:
        return CheckDetail(
            check_name="conflicting_evidence_acknowledged",
            status="pass",
            message="No contradictory evidence identified for this hypothesis.",
        )

    limitations = str(
        _extract_field(
            hypothesis,
            ["limitations", "caveats", "conflicts_addressed", "boundary_conditions"],
            "",
        )
    ).strip()

    if not limitations:
        return CheckDetail(
            check_name="conflicting_evidence_acknowledged",
            status="warning",
            message=(
                f"Hypothesis cites {len(contra_ids)} contradictory evidence source(s) "
                f"({', '.join(contra_ids[:3])}), but does not provide a limitations statement."
            ),
        )

    conflict_indicators = [
        "conflict",
        "contradict",
        "mixed",
        "inconsistent",
        "heterogen",
        "counter",
        "diverg",
        "equivocal",
        "opposing",
        "negative finding",
        "null effect",
        "caveat",
    ]
    lim_lower = limitations.lower()
    has_conflict_mention = any(ind in lim_lower for ind in conflict_indicators)

    if has_conflict_mention:
        return CheckDetail(
            check_name="conflicting_evidence_acknowledged",
            status="pass",
            message=(
                f"Contradictory evidence ({len(contra_ids)} sources) is explicitly acknowledged "
                "and contextualized in limitations."
            ),
        )

    return CheckDetail(
        check_name="conflicting_evidence_acknowledged",
        status="warning",
        message=(
            f"Hypothesis notes {len(contra_ids)} contradictory evidence items, "
            "but limitations section does not explicitly discuss conflicting findings."
        ),
    )


def run_all_checks(
    hypothesis: Any,
    evidence_pool: Optional[Union[Sequence[Any], Dict[str, Any], Set[str]]] = None,
) -> RuleCheckResult:
    """Execute all nine rule-based validation checks on a candidate hypothesis.

    Parameters
    ----------
    hypothesis : dict or object
        Hypothesis entity to evaluate.
    evidence_pool : list, dict, or set, optional
        Retrieved evidence pool of PMIDs or documents for citation coverage check.

    Returns
    -------
    RuleCheckResult
        Comprehensive validation result with check details and pass/warn/fail status.
    """
    hyp_id = str(_extract_field(hypothesis, ["hypothesis_id", "id"], "HYP-UNKNOWN"))

    checks: List[CheckDetail] = [
        check_has_variables(hypothesis),
        check_has_population(hypothesis),
        check_falsifiability(hypothesis),
        check_causal_language(hypothesis),
        check_evidence_references(hypothesis),
        check_evidence_coverage(hypothesis, evidence_pool),
        check_specificity(hypothesis),
        check_hypothesis_length(hypothesis),
        check_conflicting_evidence_acknowledged(hypothesis),
    ]

    passed_count = sum(1 for c in checks if c.status == "pass")
    failed_count = sum(1 for c in checks if c.status == "fail")
    warning_count = sum(1 for c in checks if c.status == "warning")

    if failed_count > 0:
        overall_status = "fail"
    elif warning_count > 0:
        overall_status = "warn"
    else:
        overall_status = "pass"

    logger.debug(
        "Rule checks for %s: %d passed, %d warnings, %d failed -> %s",
        hyp_id,
        passed_count,
        warning_count,
        failed_count,
        overall_status,
    )

    return RuleCheckResult(
        hypothesis_id=hyp_id,
        checks=checks,
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        overall_status=overall_status,
    )


def save_rule_results(
    results: Union[List[RuleCheckResult], Dict[str, RuleCheckResult], Sequence[Any]],
    output_path: Union[str, Path],
) -> Path:
    """Save rule check results to a JSON file.

    Parameters
    ----------
    results : list or dict of RuleCheckResult
        Rule check evaluation results.
    output_path : str or Path
        Target destination path.

    Returns
    -------
    Path
        Resolved output Path.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, dict):
        payload = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in results.items()}
    else:
        payload = [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d rule check results to %s", len(results), path)
    return path
