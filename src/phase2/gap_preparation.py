"""
gap_preparation.py — Preparation, validation, contextualization, and prioritization
of candidate research gaps identified in Phase 1 for downstream hypothesis generation.

This module bridges Phase 1 outputs to Phase 2:
1. Loads Phase 1 gap artifacts (candidate_research_gaps.csv, gap_evidence.csv)
2. Validates schema and structural integrity of gap records
3. Calculates multi-criteria prioritization scores and categories
4. Contextualizes gaps against Phase 1 papers, topic summaries, and concept statistics
5. Serializes enriched gap definitions to structured JSON
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ..config_loader import get_project_root, load_config, resolve_path
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config_loader import get_project_root, load_config, resolve_path

logger = logging.getLogger(__name__)

# Required columns for Phase 1 gap files
REQUIRED_CANDIDATE_COLUMNS = [
    "concept_a",
    "concept_b",
    "semantic_similarity",
    "cooccurrence_score",
    "temporal_growth_avg",
    "gap_score",
    "papers_concept_a",
    "papers_concept_b",
    "papers_joint",
    "supporting_topic_ids",
    "supporting_pmids_a",
    "supporting_pmids_b",
    "supporting_pmids_joint",
]

PREPARED_GAP_SCHEMA = [
    "gap_id",
    "concept_a",
    "concept_b",
    "semantic_similarity",
    "cooccurrence_score",
    "temporal_growth_a",
    "temporal_growth_b",
    "temporal_growth_avg",
    "gap_score",
    "papers_concept_a",
    "papers_concept_b",
    "papers_joint",
    "supporting_topic_ids",
    "supporting_pmids_a",
    "supporting_pmids_b",
    "supporting_pmids_joint",
    "gap_note",
    "priority_rank",
    "priority_score",
    "gap_category",
]


def _parse_id_list(value: Any) -> list[str]:
    """Parse semicolon or comma separated string of IDs into a clean list of strings."""
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("nan", "none"):
        return []
    separator = ";" if ";" in val_str else ("," if "," in val_str else None)
    if separator:
        return [item.strip() for item in val_str.split(separator) if item.strip()]
    return [val_str]


def _categorize_gap(
    semantic_sim: float,
    cooccurrence: float,
    papers_joint: int,
    temporal_growth_avg: float,
    trend_a: str = "",
    trend_b: str = "",
    papers_a: int = 0,
    papers_b: int = 0,
) -> str:
    """Categorize a candidate research gap based on its quantitative indicators.

    Categories:
    - 'unexplored_intersection': Zero co-occurrence or zero joint literature, indicating complete gap
    - 'emerging_concepts': High average temporal growth or active upward momentum in both domains
    - 'high_similarity_low_cooccurrence': High semantic proximity with negligible cross-citation
    - 'mature_domain_silo': Substantial literature on each concept independently, but siloed execution
    - 'cross_domain_opportunity': Bridging complementary domains with emerging synergy
    """
    if cooccurrence <= 0.0001 or papers_joint == 0:
        return "unexplored_intersection"
    if temporal_growth_avg >= 0.35 or (
        str(trend_a).strip().lower() == "increasing" and str(trend_b).strip().lower() == "increasing"
    ):
        return "emerging_concepts"
    if semantic_sim >= 0.85 and cooccurrence <= 0.10:
        return "high_similarity_low_cooccurrence"
    if papers_a >= 100 and papers_b >= 100 and cooccurrence <= 0.10:
        return "mature_domain_silo"
    return "cross_domain_opportunity"


def validate_gap_record(gap_row: pd.Series | dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that a single gap record conforms to required constraints.

    Parameters
    ----------
    gap_row : pd.Series or dict
        Single row from candidate research gaps.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_detected_issues)
    """
    issues: list[str] = []

    concept_a = str(gap_row.get("concept_a", "")).strip()
    concept_b = str(gap_row.get("concept_b", "")).strip()

    if not concept_a or concept_a.lower() == "nan":
        issues.append("Missing or invalid concept_a")
    if not concept_b or concept_b.lower() == "nan":
        issues.append("Missing or invalid concept_b")
    if concept_a and concept_b and concept_a.lower() == concept_b.lower():
        issues.append("Reflexive concept pair (concept_a == concept_b)")

    try:
        sim = float(gap_row.get("semantic_similarity", np.nan))
        if pd.isna(sim) or not (-1.0 <= sim <= 1.0):
            issues.append(f"Semantic similarity out of bounds [-1, 1]: {sim}")
    except (ValueError, TypeError):
        issues.append(f"Non-numeric semantic similarity: {gap_row.get('semantic_similarity')}")

    try:
        cooc = float(gap_row.get("cooccurrence_score", np.nan))
        if pd.isna(cooc) or cooc < 0.0:
            issues.append(f"Invalid negative cooccurrence_score: {cooc}")
    except (ValueError, TypeError):
        issues.append(f"Non-numeric cooccurrence_score: {gap_row.get('cooccurrence_score')}")

    try:
        gap_score = float(gap_row.get("gap_score", np.nan))
        if pd.isna(gap_score) or gap_score <= 0.0:
            issues.append(f"Invalid non-positive gap_score: {gap_score}")
    except (ValueError, TypeError):
        issues.append(f"Non-numeric gap_score: {gap_row.get('gap_score')}")

    try:
        papers_a = int(gap_row.get("papers_concept_a", -1))
        papers_b = int(gap_row.get("papers_concept_b", -1))
        papers_joint = int(gap_row.get("papers_joint", -1))
        if papers_a < 0 or papers_b < 0 or papers_joint < 0:
            issues.append("Negative paper counts detected")
    except (ValueError, TypeError):
        issues.append("Non-integer paper counts detected")

    pmids_a = _parse_id_list(gap_row.get("supporting_pmids_a", ""))
    pmids_b = _parse_id_list(gap_row.get("supporting_pmids_b", ""))
    if not pmids_a and not pmids_b:
        issues.append("No supporting PMIDs provided for either concept")

    is_valid = len(issues) == 0
    return is_valid, issues


def prioritize_gaps(gaps_df: pd.DataFrame, max_gaps: int = 20) -> pd.DataFrame:
    """Prioritize candidate research gaps using transparent, multi-criteria scoring.

    Scoring Formulation:
    --------------------
    Each factor is min-max normalized across the candidate gap pool to [0, 1]:
      1. norm_gap_score (weight: 0.40):
         Phase 1 base score capturing SemanticSim * (1 - Cooc) * Growth
      2. norm_sem_sim (weight: 0.25):
         Semantic relatedness indicating conceptual plausibility for integration
      3. norm_temporal_growth (weight: 0.20):
         Recent research momentum in both concepts
      4. joint_sparsity (weight: 0.15):
         Inversely proportional to joint publication count. Fewer joint papers
         indicates higher potential for novel discovery:
         joint_sparsity = 1.0 - (papers_joint - min_j) / (max_j - min_j + eps)

    priority_score = 0.40 * norm_gap + 0.25 * norm_sim + 0.20 * norm_growth + 0.15 * joint_sparsity
    Sorted descending by priority_score, with rank 1 being highest priority.

    Parameters
    ----------
    gaps_df : pd.DataFrame
        DataFrame of candidate gaps.
    max_gaps : int, default=20
        Maximum number of gaps to return. If <= 0 or None, returns all gaps.

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with added 'priority_score' and 'priority_rank' columns.
    """
    if gaps_df.empty:
        logger.warning("Attempted to prioritize an empty gaps DataFrame.")
        return gaps_df.copy()

    df = gaps_df.copy()
    eps = 1e-8

    gap_scores = df["gap_score"].astype(float)
    sem_sims = df["semantic_similarity"].astype(float)
    growth_avgs = df["temporal_growth_avg"].astype(float)
    joints = df["papers_joint"].astype(float)

    min_g, max_g = gap_scores.min(), gap_scores.max()
    min_s, max_s = sem_sims.min(), sem_sims.max()
    min_t, max_t = growth_avgs.min(), growth_avgs.max()
    min_j, max_j = joints.min(), joints.max()

    norm_gap = (gap_scores - min_g) / (max_g - min_g + eps) if max_g > min_g else np.ones(len(df))
    norm_sim = (sem_sims - min_s) / (max_s - min_s + eps) if max_s > min_s else np.ones(len(df))
    norm_growth = (growth_avgs - min_t) / (max_t - min_t + eps) if max_t > min_t else np.ones(len(df))
    joint_sparsity = (
        1.0 - ((joints - min_j) / (max_j - min_j + eps)) if max_j > min_j else np.ones(len(df))
    )

    priority_score = (
        0.40 * norm_gap
        + 0.25 * norm_sim
        + 0.20 * norm_growth
        + 0.15 * joint_sparsity
    )

    df["priority_score"] = priority_score.round(4)
    df = df.sort_values(by=["priority_score", "gap_score"], ascending=[False, False]).reset_index(drop=True)
    df["priority_rank"] = df.index + 1

    logger.info(
        "Prioritized %d candidate gaps. Top score: %.4f (Rank 1: %s <-> %s)",
        len(df),
        df.iloc[0]["priority_score"],
        df.iloc[0]["concept_a"],
        df.iloc[0]["concept_b"],
    )

    if max_gaps and max_gaps > 0 and len(df) > max_gaps:
        logger.info("Retaining top %d gaps as requested (max_gaps=%d)", max_gaps, max_gaps)
        df = df.head(max_gaps).copy()

    return df


def load_phase1_gaps(outputs_dir: str | Path | None = None) -> pd.DataFrame:
    """Load Phase 1 candidate research gaps and gap evidence, validate and enrich.

    Merges candidate_research_gaps.csv and gap_evidence.csv, assigns structured
    gap IDs (GAP-001, GAP-002, ...), computes gap categories, and verifies
    integrity across all rows.

    Parameters
    ----------
    outputs_dir : str or Path, optional
        Path to Phase 1 outputs directory. Defaults to '<project_root>/outputs'.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame conforming to PREPARED_GAP_SCHEMA.
    """
    if outputs_dir is None:
        outputs_dir = resolve_path("outputs")
    outputs_path = Path(outputs_dir).resolve()

    gaps_file = outputs_path / "candidate_research_gaps.csv"
    evidence_file = outputs_path / "gap_evidence.csv"

    if not gaps_file.exists():
        raise FileNotFoundError(f"Candidate research gaps not found: {gaps_file}")
    if not evidence_file.exists():
        raise FileNotFoundError(f"Gap evidence file not found: {evidence_file}")

    logger.info("Loading Phase 1 candidate gaps from %s", gaps_file)
    gaps_df = pd.read_csv(gaps_file)

    logger.info("Loading Phase 1 gap evidence from %s", evidence_file)
    evidence_df = pd.read_csv(evidence_file)

    # Validate required columns in candidate_research_gaps
    missing_cols = [c for c in REQUIRED_CANDIDATE_COLUMNS if c not in gaps_df.columns]
    if missing_cols:
        raise ValueError(f"candidate_research_gaps.csv missing required columns: {missing_cols}")

    # Standardize string keys
    gaps_df["concept_a"] = gaps_df["concept_a"].astype(str).str.strip()
    gaps_df["concept_b"] = gaps_df["concept_b"].astype(str).str.strip()
    evidence_df["concept_a"] = evidence_df["concept_a"].astype(str).str.strip()
    evidence_df["concept_b"] = evidence_df["concept_b"].astype(str).str.strip()

    # Determine unique columns in evidence to merge
    evidence_extra_cols = ["concept_a", "concept_b"]
    for col in ["evidence_description", "trend_a", "trend_b"]:
        if col in evidence_df.columns:
            evidence_extra_cols.append(col)

    merged_df = pd.merge(
        gaps_df,
        evidence_df[evidence_extra_cols],
        on=["concept_a", "concept_b"],
        how="left",
    )

    if "gap_note" not in merged_df.columns:
        merged_df["gap_note"] = ""
    merged_df["gap_note"] = merged_df["gap_note"].fillna("").astype(str)

    if "evidence_description" not in merged_df.columns:
        merged_df["evidence_description"] = ""
    merged_df["evidence_description"] = merged_df["evidence_description"].fillna("").astype(str)

    # Compute categories
    categories = []
    for _, row in merged_df.iterrows():
        cat = _categorize_gap(
            semantic_sim=float(row.get("semantic_similarity", 0.0)),
            cooccurrence=float(row.get("cooccurrence_score", 0.0)),
            papers_joint=int(row.get("papers_joint", 0)),
            temporal_growth_avg=float(row.get("temporal_growth_avg", 0.0)),
            trend_a=str(row.get("trend_a", "")),
            trend_b=str(row.get("trend_b", "")),
            papers_a=int(row.get("papers_concept_a", 0)),
            papers_b=int(row.get("papers_concept_b", 0)),
        )
        categories.append(cat)
    merged_df["gap_category"] = categories

    # Prioritize gaps across the full pool
    merged_df = prioritize_gaps(merged_df, max_gaps=-1)

    # Assign stable GAP-xxx IDs based on priority ranking
    merged_df["gap_id"] = [f"GAP-{i+1:03d}" for i in range(len(merged_df))]

    # Run validation checks on each record
    total_issues = 0
    for _, row in merged_df.iterrows():
        is_valid, issues = validate_gap_record(row)
        if not is_valid:
            total_issues += len(issues)
            logger.warning("Gap %s has validation issues: %s", row.get("gap_id"), issues)

    if total_issues == 0:
        logger.info("All %d candidate gaps passed schema validation successfully.", len(merged_df))
    else:
        logger.warning("%d validation warnings detected across candidate gaps.", total_issues)

    return merged_df


def load_phase1_artifacts(project_root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all supporting Phase 1 data artifacts into memory.

    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. Defaults to auto-resolved project root.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:
        - 'papers': mhealth_preprocessed.csv
        - 'topic_assignments': topic_assignments.csv
        - 'document_concepts': document_concepts.csv
        - 'topic_summary': topic_summary.csv
        - 'concept_summary': concept_summary.csv
        - 'mesh_normalized': mesh_normalized.csv
        - 'embedding_index': document_embedding_index.csv
    """
    if project_root is None:
        project_root = get_project_root()
    outputs_path = (Path(project_root) / "outputs").resolve()

    artifacts_meta = {
        "papers": "mhealth_preprocessed.csv",
        "topic_assignments": "topic_assignments.csv",
        "document_concepts": "document_concepts.csv",
        "topic_summary": "topic_summary.csv",
        "concept_summary": "concept_summary.csv",
        "mesh_normalized": "mesh_normalized.csv",
        "embedding_index": "document_embedding_index.csv",
    }

    artifacts: dict[str, pd.DataFrame] = {}
    for key, filename in artifacts_meta.items():
        file_path = outputs_path / filename
        if not file_path.exists():
            logger.error("Phase 1 artifact missing: %s", file_path)
            raise FileNotFoundError(f"Required Phase 1 artifact missing: {file_path}")
        logger.info("Loading artifact '%s' from %s", key, file_path.name)
        artifacts[key] = pd.read_csv(file_path)

    # Cast PMIDs to string in papers for reliable lookups
    artifacts["papers"]["pmid"] = artifacts["papers"]["pmid"].astype(str).str.strip()

    logger.info("Successfully loaded all %d Phase 1 supporting artifacts.", len(artifacts))
    return artifacts


def get_gap_context(
    gap_row: pd.Series | dict[str, Any],
    papers_df: pd.DataFrame,
    topic_summary_df: pd.DataFrame,
    concept_summary_df: pd.DataFrame,
) -> dict[str, Any]:
    """Gather complete contextual evidence and metadata for a single research gap.

    Parameters
    ----------
    gap_row : pd.Series or dict
        Single gap record.
    papers_df : pd.DataFrame
        Phase 1 preprocessed papers DataFrame (indexed or searchable by pmid).
    topic_summary_df : pd.DataFrame
        Phase 1 topic summary DataFrame.
    concept_summary_df : pd.DataFrame
        Phase 1 concept summary DataFrame.

    Returns
    -------
    dict[str, Any]
        Structured context dictionary containing paper abstracts, topic summaries,
        and concept statistics.
    """
    concept_a = str(gap_row.get("concept_a", "")).strip()
    concept_b = str(gap_row.get("concept_b", "")).strip()
    gap_id = str(gap_row.get("gap_id", "GAP-UNKNOWN"))

    pmids_a = _parse_id_list(gap_row.get("supporting_pmids_a", ""))
    pmids_b = _parse_id_list(gap_row.get("supporting_pmids_b", ""))
    pmids_joint = _parse_id_list(gap_row.get("supporting_pmids_joint", ""))
    topic_ids = [int(tid) for tid in _parse_id_list(gap_row.get("supporting_topic_ids", "")) if str(tid).isdigit()]

    # Ensure papers_df pmid lookup index exists
    papers_indexed = papers_df.copy()
    if papers_indexed.index.name != "pmid_str":
        papers_indexed["pmid_str"] = papers_indexed["pmid"].astype(str).str.strip()
        papers_indexed = papers_indexed.set_index("pmid_str")

    def _fetch_paper_records(pmid_list: list[str]) -> list[dict[str, Any]]:
        records = []
        for pmid in pmid_list:
            if pmid in papers_indexed.index:
                row = papers_indexed.loc[pmid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                records.append({
                    "pmid": pmid,
                    "title": str(row.get("title", "")),
                    "abstract": str(row.get("abstract", "")),
                    "clean_document": str(row.get("clean_document", "")),
                    "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else None,
                    "journal": str(row.get("journal", "")),
                    "doi": str(row.get("doi", "")) if pd.notna(row.get("doi")) else "",
                    "relevance_score": float(row.get("relevance_score", 0.0)) if pd.notna(row.get("relevance_score")) else 0.0,
                    "relevance_label": str(row.get("relevance_label", "")),
                })
        return records

    papers_a_records = _fetch_paper_records(pmids_a)
    papers_b_records = _fetch_paper_records(pmids_b)
    papers_joint_records = _fetch_paper_records(pmids_joint)

    # Topic summary context
    topic_context = []
    if "topic_id" in topic_summary_df.columns:
        topic_subset = topic_summary_df[topic_summary_df["topic_id"].isin(topic_ids)]
        for _, t_row in topic_subset.iterrows():
            topic_context.append({
                "topic_id": int(t_row.get("topic_id", -1)),
                "count": int(t_row.get("count", 0)),
                "percentage": float(t_row.get("percentage", 0.0)),
                "top_terms": str(t_row.get("top_terms", "")),
                "representative_titles": str(t_row.get("representative_titles", "")),
            })

    # Concept stats context
    def _fetch_concept_stats(concept_name: str) -> dict[str, Any]:
        if "concept" in concept_summary_df.columns:
            c_row = concept_summary_df[
                concept_summary_df["concept"].str.lower() == concept_name.lower()
            ]
            if not c_row.empty:
                r = c_row.iloc[0]
                return {
                    "concept": concept_name,
                    "total_mentions": int(r.get("total_mentions", 0)),
                    "document_count": int(r.get("document_count", 0)),
                }
        return {"concept": concept_name, "total_mentions": 0, "document_count": 0}

    concept_a_stats = _fetch_concept_stats(concept_a)
    concept_b_stats = _fetch_concept_stats(concept_b)

    context = {
        "gap_id": gap_id,
        "concept_a": concept_a,
        "concept_b": concept_b,
        "semantic_similarity": float(gap_row.get("semantic_similarity", 0.0)),
        "cooccurrence_score": float(gap_row.get("cooccurrence_score", 0.0)),
        "temporal_growth_avg": float(gap_row.get("temporal_growth_avg", 0.0)),
        "gap_score": float(gap_row.get("gap_score", 0.0)),
        "priority_score": float(gap_row.get("priority_score", 0.0)),
        "priority_rank": int(gap_row.get("priority_rank", 0)) if pd.notna(gap_row.get("priority_rank")) else None,
        "gap_category": str(gap_row.get("gap_category", "")),
        "gap_note": str(gap_row.get("gap_note", "")),
        "evidence_description": str(gap_row.get("evidence_description", "")),
        "concept_a_stats": concept_a_stats,
        "concept_b_stats": concept_b_stats,
        "supporting_topics": topic_context,
        "papers_concept_a": papers_a_records,
        "papers_concept_b": papers_b_records,
        "papers_joint": papers_joint_records,
        "total_papers_a": len(papers_a_records),
        "total_papers_b": len(papers_b_records),
        "total_papers_joint": len(papers_joint_records),
    }

    return context


def _convert_to_serializable(val: Any) -> Any:
    """Helper function to convert numpy/pandas types to JSON-serializable types."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.ndarray, pd.Series)):
        return [_convert_to_serializable(v) for v in val]
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return val


def save_prepared_gaps(gaps_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save prepared gaps DataFrame to JSON with complete metadata.

    Parameters
    ----------
    gaps_df : pd.DataFrame
        Prepared and prioritized gaps DataFrame.
    output_path : str or Path
        Destination JSON filepath.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for _, row in gaps_df.iterrows():
        rec = {}
        for col in gaps_df.columns:
            val = row[col]
            if col in ("supporting_pmids_a", "supporting_pmids_b", "supporting_pmids_joint", "supporting_topic_ids"):
                rec[col] = _parse_id_list(val)
            else:
                rec[col] = _convert_to_serializable(val)
        records.append(rec)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d prepared gaps with full metadata to %s", len(records), out_file)
