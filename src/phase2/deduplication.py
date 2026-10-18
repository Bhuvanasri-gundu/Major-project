"""
deduplication.py — Semantic Deduplication of Generated Hypotheses for Phase 2.

Identifies and consolidates semantically redundant hypotheses while preserving
critical substantive variations in:
  - Target populations (e.g., pediatric vs. geriatric, rural vs. urban)
  - Independent variables / interventions (e.g., SMS alerts vs. smartwatch haptics)
  - Dependent variables / clinical endpoints (e.g., systolic BP vs. medication adherence)
  - Direction of relationship (e.g., positive enhancement vs. inverse mitigation)

Embeddings are computed via SentenceTransformers, and grouping uses a greedy
leader strategy that retains the highest-quality / highest-ranked representative
from each redundancy cluster.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)


def _extract_field(obj: Any, keys: Sequence[str], default: Any = "") -> Any:
    """Helper to extract a field from a dict or an object."""
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


def _extract_evidence_ids(hypothesis: Any) -> List[str]:
    """Helper to extract list of evidence IDs."""
    raw = _extract_field(
        hypothesis,
        ["supporting_evidence_ids", "evidence_ids", "pmids", "supporting_pmids"],
        [],
    )
    if isinstance(raw, str):
        return [p.strip() for p in re.split(r"[;, ]+", raw) if p.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _tokenize_terms(text: str) -> Set[str]:
    """Tokenize and remove common stop words for substantive difference checks."""
    stops = {
        "the", "a", "an", "in", "of", "to", "for", "with", "on", "at", "by",
        "and", "or", "is", "are", "was", "were", "be", "been", "that", "this",
    }
    words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
    return {w for w in words if w not in stops}


def _jaccard_overlap(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0


@dataclass
class DuplicateGroup:
    """Group of semantically redundant hypotheses represented by a single chosen hypothesis.

    Attributes
    ----------
    representative_id : str
        ID of the retained, representative hypothesis.
    duplicate_ids : list of str
        IDs of redundant hypotheses merged into this representative.
    similarity_scores : dict of str to float
        Pairwise cosine similarity of each duplicate to the representative.
    reason : str
        Human-readable explanation of why hypotheses were grouped.
    """

    representative_id: str
    duplicate_ids: List[str]
    similarity_scores: Dict[str, float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert duplicate group to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DuplicateGroup:
        """Create duplicate group from dictionary."""
        return cls(
            representative_id=str(data.get("representative_id", "")),
            duplicate_ids=list(data.get("duplicate_ids", [])),
            similarity_scores={
                str(k): float(v) for k, v in data.get("similarity_scores", {}).items()
            },
            reason=str(data.get("reason", "")),
        )


@dataclass
class DeduplicationResult:
    """Summary of semantic deduplication operation.

    Attributes
    ----------
    total_input : int
        Number of hypotheses processed.
    total_output : int
        Number of distinct hypotheses retained.
    duplicate_groups : list of DuplicateGroup
        Clusters of identified duplicates.
    kept_ids : list of str
        List of retained hypothesis IDs.
    removed_ids : list of str
        List of pruned duplicate hypothesis IDs.
    """

    total_input: int
    total_output: int
    duplicate_groups: List[DuplicateGroup]
    kept_ids: List[str]
    removed_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
            "kept_ids": self.kept_ids,
            "removed_ids": self.removed_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeduplicationResult:
        """Instantiate result from dictionary."""
        groups = [
            DuplicateGroup.from_dict(g) if isinstance(g, dict) else g
            for g in data.get("duplicate_groups", [])
        ]
        return cls(
            total_input=int(data.get("total_input", 0)),
            total_output=int(data.get("total_output", 0)),
            duplicate_groups=groups,
            kept_ids=list(data.get("kept_ids", [])),
            removed_ids=list(data.get("removed_ids", [])),
        )


class SemanticDeduplicator:
    """Clusters and deduplicates candidate hypotheses based on dense semantic embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.85,
        device: str = "auto",
    ) -> None:
        """Initialize deduplicator with SentenceTransformer model.

        Parameters
        ----------
        model_name : str
            SentenceTransformer model identifier.
        threshold : float
            Cosine similarity threshold above which hypotheses are candidates for merging.
        device : str
            Device ('auto', 'cuda', 'cpu').
        """
        cfg = load_config()
        phase2_cfg = cfg.get("phase2", {}) if isinstance(cfg, dict) else {}
        dedup_cfg = phase2_cfg.get("deduplication", {})

        self.model_name = dedup_cfg.get(
            "embedding_model", dedup_cfg.get("model_name", model_name)
        )
        self.threshold = float(
            dedup_cfg.get(
                "similarity_threshold", dedup_cfg.get("threshold", threshold)
            )
        )

        # Device detection
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model with fallback handling."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading deduplication model '%s' on %s...", self.model_name, self.device)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Deduplication model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "Failed to load SentenceTransformer model '%s' (%s). "
                "SemanticDeduplicator will use TF-IDF/lexical overlap fallback.",
                self.model_name,
                exc,
            )
            self.model = None

    def _compute_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Compute normalized dense embeddings or fallback sparse vectors."""
        if self.model is not None:
            embeddings = self.model.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)

        # Fallback: CountVectorizer / TF-IDF if SentenceTransformer not loaded
        logger.info("Using TF-IDF fallback for semantic deduplication embeddings.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(texts).toarray()
        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (tfidf_mat / norms).astype(np.float32)

    def _are_substantively_different(self, hyp1: Any, hyp2: Any) -> bool:
        """Check if two hypotheses differ in meaningful scientific dimensions.

        Dimensions checked:
          1. Independent variable (IV) / intervention
          2. Dependent variable (DV) / primary outcome
          3. Target population (e.g. pediatric vs geriatric, rural vs urban)
          4. Proposed relationship direction (e.g. positive increase vs negative reduction)

        Returns
        -------
        bool
            True if hypotheses are substantively distinct and should NOT be merged.
        """
        # 1. Independent Variable
        iv1 = str(_extract_field(hyp1, ["independent_variable", "iv", "intervention"], "")).strip()
        iv2 = str(_extract_field(hyp2, ["independent_variable", "iv", "intervention"], "")).strip()
        if iv1 and iv2:
            tokens1 = _tokenize_terms(iv1)
            tokens2 = _tokenize_terms(iv2)
            if tokens1 and tokens2:
                iv_overlap = _jaccard_overlap(tokens1, tokens2)
                # If IVs share zero tokens, they are distinct interventions
                if iv_overlap < 0.15:
                    return True

        # 2. Dependent Variable
        dv1 = str(_extract_field(hyp1, ["dependent_variable", "dv", "outcome"], "")).strip()
        dv2 = str(_extract_field(hyp2, ["dependent_variable", "dv", "outcome"], "")).strip()
        if dv1 and dv2:
            dv_tokens1 = _tokenize_terms(dv1)
            dv_tokens2 = _tokenize_terms(dv2)
            if dv_tokens1 and dv_tokens2:
                dv_overlap = _jaccard_overlap(dv_tokens1, dv_tokens2)
                # If DVs share zero tokens, outcomes are fundamentally different
                if dv_overlap < 0.15:
                    return True

        # 3. Target Population
        pop1 = str(_extract_field(hyp1, ["target_population", "population"], "")).strip()
        pop2 = str(_extract_field(hyp2, ["target_population", "population"], "")).strip()
        if pop1 and pop2:
            pop_tok1 = _tokenize_terms(pop1)
            pop_tok2 = _tokenize_terms(pop2)
            if pop_tok1 and pop_tok2:
                pop_overlap = _jaccard_overlap(pop_tok1, pop_tok2)
                # Distinct population cohorts (e.g. adolescent vs adult, rural vs urban)
                critical_demographics = {
                    "pediatric", "children", "adolescent", "youth",
                    "adult", "elderly", "geriatric", "older",
                    "rural", "urban", "low-income", "pregnant",
                    "type 1", "type 2", "hiv", "cancer", "cardiac"
                }
                c1 = {w for w in pop_tok1 if any(cd in w for cd in critical_demographics)}
                c2 = {w for w in pop_tok2 if any(cd in w for cd in critical_demographics)}
                if c1 and c2 and not c1.intersection(c2):
                    return True
                if pop_overlap < 0.20 and len(pop_tok1) >= 2 and len(pop_tok2) >= 2:
                    return True

        # 4. Relationship Direction
        rel1 = str(_extract_field(hyp1, ["proposed_relationship", "relationship_type"], "")).strip().lower()
        rel2 = str(_extract_field(hyp2, ["proposed_relationship", "relationship_type"], "")).strip().lower()

        stmt1 = str(_extract_field(hyp1, ["hypothesis_text", "hypothesis_statement", "text"], "")).lower()
        stmt2 = str(_extract_field(hyp2, ["hypothesis_text", "hypothesis_statement", "text"], "")).lower()

        is_positive1 = any(w in stmt1 or w in rel1 for w in ["increase", "improve", "enhance", "higher", "positive"])
        is_negative1 = any(w in stmt1 or w in rel1 for w in ["decrease", "reduce", "lower", "negative", "mitigate"])

        is_positive2 = any(w in stmt2 or w in rel2 for w in ["increase", "improve", "enhance", "higher", "positive"])
        is_negative2 = any(w in stmt2 or w in rel2 for w in ["decrease", "reduce", "lower", "negative", "mitigate"])

        # If one clearly asserts an increase and the other asserts a decrease on the same construct
        if (is_positive1 and not is_negative1 and is_negative2 and not is_positive2) or \
           (is_negative1 and not is_positive1 and is_positive2 and not is_negative2):
            return True

        return False

    def _select_representative(self, group: Sequence[Any]) -> Any:
        """Select the highest-quality hypothesis from a duplicate cluster.

        Selection criteria:
          1. Highest rank_score (if available)
          2. Best (lowest) rank_position > 0
          3. Highest number of supporting evidence references
          4. Most comprehensive statement text length
        """
        def _score_key(hyp: Any) -> Tuple[float, int, int, int]:
            score = float(_extract_field(hyp, ["rank_score"], -1.0))
            pos = int(_extract_field(hyp, ["rank_position"], 999999))
            neg_pos = -pos if pos > 0 else -999999
            ev_count = len(_extract_evidence_ids(hyp))
            stmt = str(_extract_field(hyp, ["hypothesis", "hypothesis_text", "hypothesis_statement", "text"], ""))
            return (score, neg_pos, ev_count, len(stmt))

        return max(group, key=_score_key)

    def deduplicate(
        self,
        hypotheses: Sequence[Any],
        rankings: Optional[Any] = None,
    ) -> DeduplicationResult:
        """Compare all hypothesis pairs by semantic similarity and deduplicate.

        Parameters
        ----------
        hypotheses : sequence of dict or object
            Candidate hypotheses to evaluate.
        rankings : optional list or dict
            Rankings to merge if provided.

        Returns
        -------
        DeduplicationResult
            Summary of retained and removed hypothesis IDs and duplicate clusters.
        """
        n = len(hypotheses)
        if n == 0:
            return DeduplicationResult(0, 0, [], [], [])

        # If rankings provided, merge ranking attributes
        if rankings:
            from .ranking import apply_ranking
            try:
                hypotheses = apply_ranking(hypotheses, rankings)
            except Exception as exc:
                logger.debug("Could not merge rankings: %s", exc)

        # Extract textual statements
        texts = [
            str(
                _extract_field(
                    h,
                    ["hypothesis", "hypothesis_text", "hypothesis_statement", "text", "statement"],
                    "",
                )
            ).strip()
            for h in hypotheses
        ]

        hyp_ids = [
            str(_extract_field(h, ["hypothesis_id", "id"], f"HYP-{i}"))
            for i, h in enumerate(hypotheses)
        ]

        logger.info("Computing semantic embeddings for %d hypotheses...", n)
        embeddings = self._compute_embeddings(texts)

        # Compute pairwise cosine similarities: S = E . E^T
        similarity_matrix = np.dot(embeddings, embeddings.T)
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

        # Order candidates so higher quality / better ranked candidates are evaluated first
        indexed_items = list(enumerate(hypotheses))
        indexed_items.sort(
            key=lambda x: (
                float(_extract_field(x[1], ["rank_score"], 0.0)),
                -int(_extract_field(x[1], ["rank_position"], 999999)) if int(_extract_field(x[1], ["rank_position"], 999999)) > 0 else -999999,
                len(_extract_evidence_ids(x[1])),
            ),
            reverse=True,
        )

        visited_indices: Set[int] = set()
        duplicate_groups: List[DuplicateGroup] = []
        kept_ids: List[str] = []
        removed_ids: List[str] = []

        for orig_idx, hyp in indexed_items:
            if orig_idx in visited_indices:
                continue

            current_hid = hyp_ids[orig_idx]
            cluster_indices = [orig_idx]
            cluster_dups: List[str] = []
            sim_scores: Dict[str, float] = {}

            # Find candidates with high similarity to current leader
            for other_idx, other_hyp in indexed_items:
                if other_idx == orig_idx or other_idx in visited_indices:
                    continue

                sim = float(similarity_matrix[orig_idx, other_idx])
                if sim >= self.threshold:
                    # Check for substantive differences
                    if not self._are_substantively_different(hyp, other_hyp):
                        other_hid = hyp_ids[other_idx]
                        visited_indices.add(other_idx)
                        cluster_indices.append(other_idx)
                        cluster_dups.append(other_hid)
                        sim_scores[other_hid] = round(sim, 4)
                        removed_ids.append(other_hid)
                    else:
                        logger.debug(
                            "Preserved hypothesis %s from merge with %s due to substantive domain differences (sim=%.3f)",
                            hyp_ids[other_idx],
                            current_hid,
                            sim,
                        )

            visited_indices.add(orig_idx)

            if cluster_dups:
                # Select best representative among the cluster
                group_hypotheses = [hypotheses[i] for i in cluster_indices]
                representative = self._select_representative(group_hypotheses)
                rep_id = str(_extract_field(representative, ["hypothesis_id", "id"], current_hid))

                # If the representative changed from current leader, adjust kept/removed lists
                if rep_id != current_hid:
                    removed_ids.remove(rep_id)
                    removed_ids.append(current_hid)
                    cluster_dups = [
                        hyp_ids[i] for i in cluster_indices if hyp_ids[i] != rep_id
                    ]
                    # recompute sim_scores relative to rep
                    rep_idx = hyp_ids.index(rep_id)
                    sim_scores = {
                        dup_id: round(float(similarity_matrix[rep_idx, hyp_ids.index(dup_id)]), 4)
                        for dup_id in cluster_dups
                    }

                kept_ids.append(rep_id)
                duplicate_groups.append(
                    DuplicateGroup(
                        representative_id=rep_id,
                        duplicate_ids=cluster_dups,
                        similarity_scores=sim_scores,
                        reason=(
                            f"Semantic similarity >= {self.threshold:.2f} with no substantive "
                            "difference in variables, target cohort, or relationship direction."
                        ),
                    )
                )
            else:
                kept_ids.append(current_hid)

        logger.info(
            "Deduplication complete: %d input -> %d retained (%d duplicates merged across %d groups)",
            n,
            len(kept_ids),
            len(removed_ids),
            len(duplicate_groups),
        )

        return DeduplicationResult(
            total_input=n,
            total_output=len(kept_ids),
            duplicate_groups=duplicate_groups,
            kept_ids=kept_ids,
            removed_ids=removed_ids,
        )


def get_deduplicated_hypotheses(
    hypotheses: Sequence[Any],
    dedup_result: DeduplicationResult,
) -> List[Any]:
    """Filter candidate hypotheses to only retain unique representatives.

    Preserves all original metadata, ranking attributes, and validation history.

    Parameters
    ----------
    hypotheses : sequence of dict or object
        Original hypotheses.
    dedup_result : DeduplicationResult
        Output of semantic deduplication.

    Returns
    -------
    list of dict or object
        Filtered hypotheses containing only items in kept_ids.
    """
    kept_set = set(dedup_result.kept_ids)
    retained: List[Any] = []

    for hyp in hypotheses:
        hid = str(_extract_field(hyp, ["hypothesis_id", "id"], ""))
        if hid in kept_set:
            retained.append(hyp)

    return retained


def save_deduplication_results(
    result: DeduplicationResult,
    output_path: Union[str, Path],
) -> Path:
    """Save deduplication result to a JSON file.

    Parameters
    ----------
    result : DeduplicationResult
        Result object to serialize.
    output_path : str or Path
        Target destination path.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Saved deduplication report to %s", path)
    return path
