"""
evidence_verification.py — Evidence Integrity & Reference Validation Module.

Validates the integrity, factual grounding, and provenance of evidence citations
in generated hypotheses. Verifies evidence IDs against retrieved candidate pools,
checks for fabricated citations, validates PMID existence in the mHealth corpus,
computes evidence coverage, and assesses content alignment between hypotheses
and cited scientific literature abstracts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

try:
    from ..config_loader import get_project_root, load_config
except (ImportError, ValueError):
    from src.config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Common English stopwords for content grounding lexical analysis
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


@dataclass
class VerificationResult:
    """Detailed result of verifying cited evidence IDs against the provided pool."""

    hypothesis_id: str
    valid_ids: List[str]
    invalid_ids: List[str]
    missing_ids: List[str]
    fabricated_count: int
    verification_status: str  # 'passed', 'failed', 'partial'
    total_cited: int = 0
    total_pool: int = 0
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert VerificationResult dataclass instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationResult:
        """Create VerificationResult instance from dictionary."""
        return cls(
            hypothesis_id=str(data.get("hypothesis_id", "")),
            valid_ids=list(data.get("valid_ids", [])),
            invalid_ids=list(data.get("invalid_ids", [])),
            missing_ids=list(data.get("missing_ids", [])),
            fabricated_count=int(data.get("fabricated_count", 0)),
            verification_status=str(data.get("verification_status", "failed")),
            total_cited=int(data.get("total_cited", 0)),
            total_pool=int(data.get("total_pool", 0)),
            details=data.get("details"),
        )


@dataclass
class VerificationReport:
    """Comprehensive verification report across all integrity and grounding dimensions."""

    hypothesis_id: str
    reference_result: VerificationResult
    content_grounded: bool
    content_similarity: float
    corpus_pmid_check: Dict[str, Any]
    evidence_coverage: float
    overall_passed: bool
    overall_score: float
    flags: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert VerificationReport instance to a dictionary."""
        d = asdict(self)
        if isinstance(self.reference_result, VerificationResult):
            d["reference_result"] = self.reference_result.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationReport:
        """Create VerificationReport instance from dictionary."""
        ref_data = data.get("reference_result", {})
        if isinstance(ref_data, dict):
            ref_result = VerificationResult.from_dict(ref_data)
        else:
            ref_result = ref_data

        return cls(
            hypothesis_id=str(data.get("hypothesis_id", "")),
            reference_result=ref_result,
            content_grounded=bool(data.get("content_grounded", False)),
            content_similarity=float(data.get("content_similarity", 0.0)),
            corpus_pmid_check=dict(data.get("corpus_pmid_check", {})),
            evidence_coverage=float(data.get("evidence_coverage", 0.0)),
            overall_passed=bool(data.get("overall_passed", False)),
            overall_score=float(data.get("overall_score", 0.0)),
            flags=list(data.get("flags", [])),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
        )


def _extract_id_from_item(item: Any, fallback_idx: int = 0) -> str:
    """Extract standard chunk/evidence ID from a pool item."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(
            item.get("chunk_id")
            or item.get("evidence_id")
            or item.get("id")
            or f"chunk_{fallback_idx}"
        ).strip()
    return str(
        getattr(item, "chunk_id", None)
        or getattr(item, "evidence_id", None)
        or getattr(item, "id", None)
        or f"chunk_{fallback_idx}"
    ).strip()


def _extract_text_from_item(item: Any) -> str:
    """Extract text/abstract content from an evidence item."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(
            item.get("text")
            or item.get("content")
            or item.get("passage")
            or item.get("clean_abstract")
            or item.get("abstract")
            or ""
        ).strip()
    return str(
        getattr(item, "text", None)
        or getattr(item, "content", None)
        or getattr(item, "passage", None)
        or getattr(item, "clean_abstract", None)
        or getattr(item, "abstract", None)
        or ""
    ).strip()


def _extract_pmid_from_item(item: Any) -> Optional[str]:
    """Extract PMID from an evidence item."""
    if isinstance(item, dict):
        pmid = item.get("pmid") or item.get("PMID")
        if pmid is not None:
            return str(pmid).strip()
    elif not isinstance(item, str):
        pmid = getattr(item, "pmid", None) or getattr(item, "PMID", None)
        if pmid is not None:
            return str(pmid).strip()

    # Try extracting numeric PMID pattern from string or chunk_id
    item_str = str(item)
    match = re.search(r"\b(\d{7,8})\b", item_str)
    if match:
        return match.group(1)
    return None


def _get_hypothesis_field(hyp: Any, field_name: str, default: Any = None) -> Any:
    """Safely get field from Hypothesis dataclass or dictionary."""
    if isinstance(hyp, dict):
        return hyp.get(field_name, default)
    return getattr(hyp, field_name, default)


def _tokenize_text(text: str) -> Set[str]:
    """Extract normalized lowercase alphanumeric content words (length >= 3, excluding stopwords)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _normalize_evidence_pool(evidence_pool: Any) -> List[Any]:
    """Safely convert evidence pool to list whether provided as list, dict, or collection."""
    if evidence_pool is None:
        return []
    if isinstance(evidence_pool, dict):
        return list(evidence_pool.values())
    if isinstance(evidence_pool, (list, tuple, set)):
        return list(evidence_pool)
    return [evidence_pool]


def verify_evidence_references(
    hypothesis: Union[Any, Dict[str, Any]],
    evidence_pool: Any,
) -> VerificationResult:
    """Verify that every cited evidence ID exists in the provided evidence pool.

    Parameters
    ----------
    hypothesis : Hypothesis or dict
        The hypothesis containing supporting_evidence_ids and contradictory_evidence_ids.
    evidence_pool : list or dict
        List of evidence chunk dictionaries or objects provided during generation.

    Returns
    -------
    VerificationResult
        Dataclass containing valid_ids, invalid_ids, missing_ids, fabricated_count,
        and verification_status.
    """
    evidence_pool = _normalize_evidence_pool(evidence_pool)
    hyp_id = str(_get_hypothesis_field(hypothesis, "hypothesis_id", "unknown_hyp"))

    sup_ids = _get_hypothesis_field(hypothesis, "supporting_evidence_ids", []) or []
    contra_ids = _get_hypothesis_field(hypothesis, "contradictory_evidence_ids", []) or []

    # Clean cited IDs
    cited_ids: List[str] = []
    for cid in list(sup_ids) + list(contra_ids):
        if cid:
            cleaned = str(cid).strip()
            if cleaned and cleaned not in cited_ids:
                cited_ids.append(cleaned)

    # Build pool set
    pool_id_map: Dict[str, Any] = {}
    for idx, item in enumerate(evidence_pool):
        eid = _extract_id_from_item(item, idx)
        if eid:
            pool_id_map[eid] = item

    pool_ids = set(pool_id_map.keys())

    valid_ids = [cid for cid in cited_ids if cid in pool_ids]
    invalid_ids = [cid for cid in cited_ids if cid not in pool_ids]
    missing_ids = [pid for pid in pool_ids if pid not in cited_ids]
    fabricated_count = len(invalid_ids)

    # Determine verification status
    if len(cited_ids) == 0:
        # No evidence was cited
        status = "failed"
    elif fabricated_count == 0 and len(valid_ids) > 0:
        status = "passed"
    elif len(valid_ids) == 0 and fabricated_count > 0:
        status = "failed"
    else:
        # Partial: some valid, some invalid
        status = "partial"

    logger.debug(
        "Hypothesis '%s' evidence reference verification: status=%s, valid=%d, invalid=%d",
        hyp_id,
        status,
        len(valid_ids),
        len(invalid_ids),
    )

    return VerificationResult(
        hypothesis_id=hyp_id,
        valid_ids=valid_ids,
        invalid_ids=invalid_ids,
        missing_ids=missing_ids,
        fabricated_count=fabricated_count,
        verification_status=status,
        total_cited=len(cited_ids),
        total_pool=len(evidence_pool),
        details={
            "supporting_cited": list(sup_ids),
            "contradictory_cited": list(contra_ids),
        },
    )


def verify_evidence_content(
    hypothesis: Union[Any, Dict[str, Any]],
    evidence_pool: Any,
) -> Dict[str, Any]:
    """Verify that cited evidence passages were actually included in the generation context

    and assess content grounding via lexical and entity overlap.

    Parameters
    ----------
    hypothesis : Hypothesis or dict
        Hypothesis to verify.
    evidence_pool : list or dict
        Pool of evidence chunks provided to the LLM.

    Returns
    -------
    dict
        Grounding metrics including content_grounded, content_similarity,
        cited_passages_found, overlap_terms, and details.
    """
    evidence_pool = _normalize_evidence_pool(evidence_pool)
    hyp_text = str(_get_hypothesis_field(hypothesis, "hypothesis", ""))
    rq_text = str(_get_hypothesis_field(hypothesis, "research_question", ""))
    iv_text = str(_get_hypothesis_field(hypothesis, "independent_variable", ""))
    dv_text = str(_get_hypothesis_field(hypothesis, "dependent_variable", ""))

    combined_hyp_text = f"{hyp_text} {rq_text} {iv_text} {dv_text}".strip()
    hyp_tokens = _tokenize_text(combined_hyp_text)

    sup_ids = _get_hypothesis_field(hypothesis, "supporting_evidence_ids", []) or []
    contra_ids = _get_hypothesis_field(hypothesis, "contradictory_evidence_ids", []) or []
    cited_ids = set(str(c).strip() for c in list(sup_ids) + list(contra_ids) if c)

    pool_id_map: Dict[str, Any] = {}
    for idx, item in enumerate(evidence_pool):
        eid = _extract_id_from_item(item, idx)
        if eid:
            pool_id_map[eid] = item

    cited_texts: List[str] = []
    found_count = 0
    missing_count = 0

    for cid in cited_ids:
        if cid in pool_id_map:
            item = pool_id_map[cid]
            text = _extract_text_from_item(item)
            if text:
                cited_texts.append(text)
            found_count += 1
        else:
            missing_count += 1

    combined_evidence_text = " ".join(cited_texts)
    ev_tokens = _tokenize_text(combined_evidence_text)

    # Compute Jaccard overlap of non-stopword tokens
    if hyp_tokens and ev_tokens:
        intersection = hyp_tokens.intersection(ev_tokens)
        union = hyp_tokens.union(ev_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0
        # Hypothesis coverage: what percentage of key hypothesis terms appear in evidence?
        hyp_coverage = len(intersection) / len(hyp_tokens) if hyp_tokens else 0.0
        overlap_terms = sorted(list(intersection))
    else:
        jaccard = 0.0
        hyp_coverage = 0.0
        overlap_terms = []

    # Check key variables presence in cited evidence
    iv_tokens = _tokenize_text(iv_text)
    dv_tokens = _tokenize_text(dv_text)
    iv_grounded = bool(iv_tokens.intersection(ev_tokens)) if iv_tokens else True
    dv_grounded = bool(dv_tokens.intersection(ev_tokens)) if dv_tokens else True

    # Composite similarity score (weighted Jaccard + hypothesis concept coverage)
    content_similarity = round(0.4 * jaccard + 0.6 * hyp_coverage, 4)

    # Grounding threshold: at least 15% hypothesis token coverage or both variables grounded
    content_grounded = bool(
        found_count > 0
        and (hyp_coverage >= 0.15 or (iv_grounded and dv_grounded and len(overlap_terms) >= 3))
    )

    return {
        "content_grounded": content_grounded,
        "content_similarity": content_similarity,
        "jaccard_similarity": round(jaccard, 4),
        "hypothesis_term_coverage": round(hyp_coverage, 4),
        "cited_passages_found": found_count,
        "cited_passages_missing": missing_count,
        "iv_grounded": iv_grounded,
        "dv_grounded": dv_grounded,
        "overlap_terms": overlap_terms[:25],
        "details": {
            "num_hyp_terms": len(hyp_tokens),
            "num_evidence_terms": len(ev_tokens),
            "num_overlapping_terms": len(overlap_terms),
        },
    }


def _load_corpus_pmids() -> Set[str]:
    """Load known PMIDs from project artifacts."""
    root = get_project_root()
    # Fast path: outputs/document_embedding_index.csv has all PMIDs
    index_csv = root / "outputs" / "document_embedding_index.csv"
    if index_csv.exists():
        try:
            df = pd.read_csv(index_csv, usecols=["pmid"], dtype={"pmid": str})
            return set(df["pmid"].str.strip().dropna().unique())
        except Exception as e:
            logger.debug("Could not read document_embedding_index.csv: %s", e)

    # Fallback: candidate_research_gaps.csv
    gaps_csv = root / "outputs" / "candidate_research_gaps.csv"
    if gaps_csv.exists():
        try:
            df = pd.read_csv(gaps_csv)
            pmids: Set[str] = set()
            for col in ["supporting_pmids_a", "supporting_pmids_b", "supporting_pmids_joint"]:
                if col in df.columns:
                    for val in df[col].dropna():
                        for p in str(val).split(";"):
                            p_clean = p.strip()
                            if p_clean:
                                pmids.add(p_clean)
            if pmids:
                return pmids
        except Exception as e:
            logger.debug("Could not read candidate_research_gaps.csv: %s", e)

    return set()


def check_metadata_consistency(
    hypothesis: Union[Any, Dict[str, Any]],
    papers_df: Optional[Union[pd.DataFrame, Dict[str, Any], Set[str], List[str]]] = None,
    evidence_pool: Optional[List[Union[Dict[str, Any], Any]]] = None,
) -> Dict[str, Any]:
    """Verify that PMIDs extracted from cited evidence correspond to real papers in the corpus.

    Parameters
    ----------
    hypothesis : Hypothesis or dict
        Hypothesis to verify.
    papers_df : pd.DataFrame, set, list, or dict, optional
        Corpus reference containing real PMIDs. If None, attempts to load from artifacts.
    evidence_pool : list, optional
        Provided evidence pool to map chunk IDs to their metadata PMIDs.

    Returns
    -------
    dict
        Metadata check result: real_pmids, unknown_pmids, all_valid,
        total_checked, and consistency_score.
    """
    # 1. Resolve known corpus PMIDs set
    known_pmids: Set[str] = set()
    if papers_df is not None:
        if isinstance(papers_df, pd.DataFrame):
            col = "pmid" if "pmid" in papers_df.columns else ("PMID" if "PMID" in papers_df.columns else None)
            if col:
                known_pmids = set(papers_df[col].astype(str).str.strip().unique())
        elif isinstance(papers_df, (set, list, tuple)):
            known_pmids = set(str(p).strip() for p in papers_df if p)
        elif isinstance(papers_df, dict):
            known_pmids = set(str(k).strip() for k in papers_df.keys())

    if not known_pmids:
        known_pmids = _load_corpus_pmids()

    # 2. Extract PMIDs associated with the hypothesis
    extracted_pmids: Set[str] = set()

    # From evidence pool mapping for cited IDs
    sup_ids = _get_hypothesis_field(hypothesis, "supporting_evidence_ids", []) or []
    contra_ids = _get_hypothesis_field(hypothesis, "contradictory_evidence_ids", []) or []
    cited_ids = set(str(c).strip() for c in list(sup_ids) + list(contra_ids) if c)

    if evidence_pool:
        pool_map: Dict[str, Any] = {}
        for idx, item in enumerate(evidence_pool):
            eid = _extract_id_from_item(item, idx)
            pool_map[eid] = item

        for cid in cited_ids:
            if cid in pool_map:
                pmid = _extract_pmid_from_item(pool_map[cid])
                if pmid:
                    extracted_pmids.add(pmid)

    # From cited IDs strings directly (e.g. "31699716_chunk_0" or "PMID_28088232")
    for cid in cited_ids:
        match = re.search(r"\b(\d{7,8})\b", cid)
        if match:
            extracted_pmids.add(match.group(1))

    # From hypothesis text or explanation if any PMIDs mentioned
    all_text = " ".join([
        str(_get_hypothesis_field(hypothesis, "hypothesis", "")),
        str(_get_hypothesis_field(hypothesis, "testability_explanation", "")),
    ])
    for match in re.finditer(r"\b(?:pmid[:\s]*)?(\d{7,8})\b", all_text, re.IGNORECASE):
        extracted_pmids.add(match.group(1))

    if not extracted_pmids:
        # No PMIDs were cited or extracted
        return {
            "real_pmids": [],
            "unknown_pmids": [],
            "all_valid": True,
            "total_checked": 0,
            "consistency_score": 1.0,
            "note": "No explicit PMIDs extracted from cited IDs or text.",
        }

    if not known_pmids:
        # Corpus reference unavailable; cannot verify PMIDs against corpus
        return {
            "real_pmids": list(extracted_pmids),
            "unknown_pmids": [],
            "all_valid": True,
            "total_checked": len(extracted_pmids),
            "consistency_score": 1.0,
            "note": "Corpus PMID index not available for offline verification.",
        }

    real_pmids = sorted([p for p in extracted_pmids if p in known_pmids])
    unknown_pmids = sorted([p for p in extracted_pmids if p not in known_pmids])
    all_valid = len(unknown_pmids) == 0 and len(real_pmids) > 0
    consistency_score = len(real_pmids) / len(extracted_pmids) if extracted_pmids else 1.0

    return {
        "real_pmids": real_pmids,
        "unknown_pmids": unknown_pmids,
        "all_valid": all_valid,
        "total_checked": len(extracted_pmids),
        "consistency_score": round(consistency_score, 4),
    }


def compute_evidence_coverage(
    hypothesis: Union[Any, Dict[str, Any]],
    evidence_pool: List[Union[Dict[str, Any], Any]],
) -> float:
    """Calculate what fraction of provided evidence in the pool was actually cited.

    Parameters
    ----------
    hypothesis : Hypothesis or dict
        Hypothesis to check.
    evidence_pool : list
        The list of available evidence chunks.

    Returns
    -------
    float
        Coverage ratio in [0.0, 1.0].
    """
    evidence_pool = _normalize_evidence_pool(evidence_pool)
    if not evidence_pool:
        return 0.0

    sup_ids = _get_hypothesis_field(hypothesis, "supporting_evidence_ids", []) or []
    contra_ids = _get_hypothesis_field(hypothesis, "contradictory_evidence_ids", []) or []
    cited_ids = set(str(c).strip() for c in list(sup_ids) + list(contra_ids) if c)

    pool_ids = set()
    for idx, item in enumerate(evidence_pool):
        eid = _extract_id_from_item(item, idx)
        if eid:
            pool_ids.add(eid)

    if not pool_ids:
        return 0.0

    valid_cited = cited_ids.intersection(pool_ids)
    coverage = len(valid_cited) / len(pool_ids)
    return round(float(coverage), 4)


def aggregate_verification(
    hypothesis: Union[Any, Dict[str, Any]],
    evidence_pool: Any,
    papers_df: Optional[Union[pd.DataFrame, Dict[str, Any], Set[str], List[str]]] = None,
) -> VerificationReport:
    """Run all verification checks and return a comprehensive VerificationReport.

    Parameters
    ----------
    hypothesis : Hypothesis or dict
        Hypothesis to verify.
    evidence_pool : list or dict
        Evidence chunks provided to the LLM during generation.
    papers_df : DataFrame, set, or None
        Known papers in corpus for PMID consistency check.

    Returns
    -------
    VerificationReport
        The aggregated verification report.
    """
    evidence_pool = _normalize_evidence_pool(evidence_pool)
    hyp_id = str(_get_hypothesis_field(hypothesis, "hypothesis_id", "unknown_hyp"))

    # 1. Reference check
    ref_result = verify_evidence_references(hypothesis, evidence_pool)

    # 2. Content grounding check
    content_result = verify_evidence_content(hypothesis, evidence_pool)

    # 3. Metadata consistency check
    pmid_result = check_metadata_consistency(hypothesis, papers_df, evidence_pool=evidence_pool)

    # 4. Evidence coverage check
    coverage = compute_evidence_coverage(hypothesis, evidence_pool)

    # Collect flags
    flags: List[str] = []
    if ref_result.fabricated_count > 0:
        flags.append(f"fabricated_evidence_detected:{ref_result.fabricated_count}_invalid_ids")
    if len(ref_result.valid_ids) == 0:
        flags.append("no_valid_evidence_cited")
    if not content_result["content_grounded"]:
        flags.append("insufficient_content_grounding")
    if pmid_result["unknown_pmids"]:
        flags.append(f"unknown_pmid_cited:{','.join(pmid_result['unknown_pmids'])}")
    if coverage < 0.05 and len(evidence_pool) >= 5:
        flags.append("low_evidence_coverage")

    # Composite overall score [0.0, 1.0]
    total_cited = max(1, ref_result.total_cited)
    ref_score = len(ref_result.valid_ids) / total_cited if ref_result.valid_ids else 0.0
    content_sim = content_result.get("content_similarity", 0.0)
    meta_score = pmid_result.get("consistency_score", 1.0)
    cov_score = min(1.0, coverage * 2.5)  # normalize reasonable coverage ~40% to 1.0

    overall_score = (
        0.35 * ref_score
        + 0.35 * min(1.0, content_sim * 2.0)
        + 0.20 * meta_score
        + 0.10 * cov_score
    )

    # Penalize fabricated citations severely
    if ref_result.fabricated_count > 0:
        overall_score *= 0.5

    overall_score = round(max(0.0, min(1.0, overall_score)), 4)

    # Overall passed condition:
    # 1. No fabricated citations
    # 2. Status is passed (or partial with >= 1 valid citation and 0 invalid)
    # 3. Content is grounded
    # 4. All cited PMIDs are valid in the corpus
    overall_passed = bool(
        ref_result.fabricated_count == 0
        and len(ref_result.valid_ids) > 0
        and content_result["content_grounded"]
        and pmid_result["all_valid"]
    )

    return VerificationReport(
        hypothesis_id=hyp_id,
        reference_result=ref_result,
        content_grounded=content_result["content_grounded"],
        content_similarity=content_sim,
        corpus_pmid_check=pmid_result,
        evidence_coverage=coverage,
        overall_passed=overall_passed,
        overall_score=overall_score,
        flags=flags,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_verification_results(
    results: Union[VerificationReport, List[VerificationReport], Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Save verification reports to a JSON file.

    Parameters
    ----------
    results : VerificationReport, list of VerificationReport, or dict
        Results to serialize.
    output_path : str or Path
        Target destination filepath.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, VerificationReport):
        data = results.to_dict()
    elif isinstance(results, list):
        data = [r.to_dict() if isinstance(r, VerificationReport) else r for r in results]
    elif isinstance(results, dict):
        data = {
            k: v.to_dict() if isinstance(v, VerificationReport) else v
            for k, v in results.items()
        }
    else:
        data = results

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved verification results to %s", path)
