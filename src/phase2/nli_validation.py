"""
nli_validation.py — Natural Language Inference (NLI) Alignment Verification Module.

Leverages a Cross-Encoder transformer model (`cross-encoder/nli-deberta-v3-small`)
to evaluate directional entailment, neutral framing, and contradiction between
retrieved evidence passages (premises) and generated scientific hypotheses (hypotheses).
Enforces empirical grounding and guards against counter-factual assertions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.special

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

try:
    from ..config_loader import get_project_root, load_config
except (ImportError, ValueError):
    from src.config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class PassageNLI:
    """Passage-level Natural Language Inference result."""

    evidence_id: str
    label: str  # 'entailment', 'contradiction', 'neutral'
    entailment_score: float
    contradiction_score: float
    neutral_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert PassageNLI instance to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PassageNLI:
        """Create PassageNLI instance from dictionary."""
        return cls(
            evidence_id=str(data.get("evidence_id", "")),
            label=str(data.get("label", "neutral")),
            entailment_score=float(data.get("entailment_score", 0.0)),
            contradiction_score=float(data.get("contradiction_score", 0.0)),
            neutral_score=float(data.get("neutral_score", 0.0)),
        )


@dataclass
class NLIResult:
    """Aggregated Natural Language Inference verification result for a hypothesis."""

    hypothesis_id: str
    passage_results: List[PassageNLI]
    support_count: int
    contradiction_count: int
    neutral_count: int
    overall_alignment: str  # 'supported', 'contradicted', 'neutral', 'mixed'
    alignment_score: float  # continuous normalized score in [0.0, 1.0]
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert NLIResult instance to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "passage_results": [
                p.to_dict() if isinstance(p, PassageNLI) else p for p in self.passage_results
            ],
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "neutral_count": self.neutral_count,
            "overall_alignment": self.overall_alignment,
            "alignment_score": round(self.alignment_score, 4),
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NLIResult:
        """Create NLIResult instance from dictionary."""
        passages_raw = data.get("passage_results", [])
        passage_objs = [
            PassageNLI.from_dict(p) if isinstance(p, dict) else p for p in passages_raw
        ]
        return cls(
            hypothesis_id=str(data.get("hypothesis_id", "")),
            passage_results=passage_objs,
            support_count=int(data.get("support_count", 0)),
            contradiction_count=int(data.get("contradiction_count", 0)),
            neutral_count=int(data.get("neutral_count", 0)),
            overall_alignment=str(data.get("overall_alignment", "neutral")),
            alignment_score=float(data.get("alignment_score", 0.5)),
            model_used=str(data.get("model_used", "unknown")),
        )


class NLIValidator:
    """Cross-Encoder NLI model evaluator for hypothesis-evidence alignment."""

    def __init__(
        self,
        model_name: Optional[str] = "cross-encoder/nli-deberta-v3-small",
        device: str = "auto",
        batch_size: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize NLIValidator with DeBERTa Cross-Encoder.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face Cross-Encoder model name. Defaults to 'cross-encoder/nli-deberta-v3-small'.
        device : str, default 'auto'
            Compute device ('cuda', 'cpu', or 'auto').
        batch_size : int, default 16
            Inference batch size.
        config : dict, optional
            Config dictionary. If None, loaded from load_config().
        """
        self.config = config or self._load_default_config()
        nli_cfg = self.config.get("phase2", {}).get("nli", {})

        # Model identifier
        self.model_name = (
            model_name
            or nli_cfg.get("model_name")
            or "cross-encoder/nli-deberta-v3-small"
        )
        self.batch_size = int(batch_size or nli_cfg.get("batch_size", 16))
        self.support_threshold = float(nli_cfg.get("support_threshold", 0.60))
        self.contradiction_threshold = float(nli_cfg.get("contradiction_threshold", 0.60))

        # Device selection
        if device == "auto":
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self.model = None
        self.available = False
        self.id2label: Dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.label2id: Dict[str, int] = {"contradiction": 0, "entailment": 1, "neutral": 2}

        self._initialize_model()

    @staticmethod
    def _load_default_config() -> Dict[str, Any]:
        """Safely load configuration file."""
        try:
            return load_config()
        except Exception as e:
            logger.warning("Could not load config.yaml: %s. Using default NLI parameters.", e)
            return {}

    def _initialize_model(self) -> None:
        """Load the Cross-Encoder model with proper error handling and fallback."""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning(
                "sentence_transformers.CrossEncoder is not installed. NLI validation disabled."
            )
            self.available = False
            return

        try:
            logger.info(
                "Loading Cross-Encoder NLI model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )

            # Inspect model label mapping if available
            if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                cfg = self.model.model.config
                if hasattr(cfg, "id2label") and cfg.id2label:
                    self.id2label = {int(k): str(v).lower() for k, v in cfg.id2label.items()}
                if hasattr(cfg, "label2id") and cfg.label2id:
                    self.label2id = {str(k).lower(): int(v) for k, v in cfg.label2id.items()}

            self.available = True
            logger.info(
                "Successfully loaded NLI model '%s'. Label mapping: %s",
                self.model_name,
                self.id2label,
            )

        except Exception as e:
            logger.warning(
                "Failed to load Cross-Encoder NLI model '%s': %s. "
                "NLI validation will gracefully degrade and return neutral status without simulated data.",
                self.model_name,
                e,
            )
            self.model = None
            self.available = False

    def is_available(self) -> bool:
        """Check whether the NLI model is successfully loaded and ready for inference."""
        return self.available and self.model is not None

    def _predict_nli(
        self, premise: str, hypothesis: str
    ) -> Tuple[str, Dict[str, float]]:
        """Predict NLI label and calibrated probabilities for a single premise-hypothesis pair.

        Parameters
        ----------
        premise : str
            The scientific evidence passage / abstract excerpt.
        hypothesis : str
            The generated hypothesis statement.

        Returns
        -------
        tuple of (str, dict)
            Predicted label ('entailment', 'contradiction', 'neutral') and dictionary
            mapping each class to its softmax probability.
        """
        if not self.is_available():
            raise RuntimeError("NLI model is not available for prediction.")

        raw_logits = self.model.predict([(premise, hypothesis)])
        probs = scipy.special.softmax(raw_logits, axis=-1)[0]

        scores_dict: Dict[str, float] = {}
        for idx, prob in enumerate(probs):
            lbl = self.id2label.get(idx, f"label_{idx}")
            scores_dict[lbl] = float(prob)

        entail_score = scores_dict.get("entailment", 0.0)
        contra_score = scores_dict.get("contradiction", 0.0)
        neutral_score = scores_dict.get("neutral", 0.0)

        # High-confidence thresholding
        if entail_score >= self.support_threshold and entail_score > contra_score:
            best_label = "entailment"
        elif contra_score >= self.contradiction_threshold and contra_score > entail_score:
            best_label = "contradiction"
        else:
            # Fall back to argmax
            argmax_idx = int(np.argmax(probs))
            best_label = self.id2label.get(argmax_idx, "neutral")

        return best_label, {
            "entailment": round(entail_score, 4),
            "contradiction": round(contra_score, 4),
            "neutral": round(neutral_score, 4),
        }

    def predict_batch_nli(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Predict NLI labels for multiple premise-hypothesis pairs in efficient mini-batches.

        Parameters
        ----------
        pairs : list of tuple of (premise, hypothesis)
            List of text pairs.

        Returns
        -------
        list of tuple of (str, dict)
            List of (best_label, scores_dict) per pair.
        """
        if not pairs:
            return []

        if not self.is_available():
            raise RuntimeError("NLI model is not available for batch prediction.")

        raw_logits = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        probs = scipy.special.softmax(raw_logits, axis=-1)
        results = []

        for row_probs in probs:
            scores_dict: Dict[str, float] = {}
            for idx, prob in enumerate(row_probs):
                lbl = self.id2label.get(idx, f"label_{idx}")
                scores_dict[lbl] = float(prob)

            entail_score = scores_dict.get("entailment", 0.0)
            contra_score = scores_dict.get("contradiction", 0.0)
            neutral_score = scores_dict.get("neutral", 0.0)

            if entail_score >= self.support_threshold and entail_score > contra_score:
                best_label = "entailment"
            elif contra_score >= self.contradiction_threshold and contra_score > entail_score:
                best_label = "contradiction"
            else:
                argmax_idx = int(np.argmax(row_probs))
                best_label = self.id2label.get(argmax_idx, "neutral")

            results.append(
                (
                    best_label,
                    {
                        "entailment": round(entail_score, 4),
                        "contradiction": round(contra_score, 4),
                        "neutral": round(neutral_score, 4),
                    },
                )
            )

        return results

    def aggregate_results(
        self, passage_results: List[PassageNLI]
    ) -> Tuple[int, int, int, str, float]:
        """Aggregate passage-level NLI inferences into global hypothesis alignment metrics.

        Aggregation Method:
        -------------------
        1. Categorical Classification:
           - 'supported': At least one passage entails the hypothesis and zero passages contradict it.
           - 'contradicted': At least one passage contradicts the hypothesis and zero passages entail it.
           - 'mixed': Both supporting (entailing) and contradictory passages are observed.
           - 'neutral': All evaluated passages are classified as neutral / inconclusive.

        2. Continuous Alignment Score:
           Computes the mean directional evidence margin across all passages:
             margin = mean(entailment_score) - mean(contradiction_score)
           Normalized to [0.0, 1.0] via affine transformation:
             alignment_score = 0.5 + 0.5 * margin
           Where:
             1.0 = absolute entailment across all cited passages
             0.5 = neutral or balanced evidence
             0.0 = absolute contradiction / refutation by literature

        Parameters
        ----------
        passage_results : list of PassageNLI
            Passage-level NLI evaluations.

        Returns
        -------
        tuple of (support_count, contradiction_count, neutral_count, overall_alignment, alignment_score)
        """
        if not passage_results:
            return 0, 0, 0, "neutral", 0.5

        support_count = sum(1 for p in passage_results if p.label == "entailment")
        contradiction_count = sum(1 for p in passage_results if p.label == "contradiction")
        neutral_count = sum(1 for p in passage_results if p.label == "neutral")

        # Categorical alignment
        if support_count > 0 and contradiction_count == 0:
            overall_alignment = "supported"
        elif contradiction_count > 0 and support_count == 0:
            overall_alignment = "contradicted"
        elif support_count > 0 and contradiction_count > 0:
            overall_alignment = "mixed"
        else:
            overall_alignment = "neutral"

        # Continuous alignment score
        mean_entailment = float(np.mean([p.entailment_score for p in passage_results]))
        mean_contradiction = float(np.mean([p.contradiction_score for p in passage_results]))
        directional_margin = mean_entailment - mean_contradiction

        alignment_score = max(0.0, min(1.0, 0.5 + 0.5 * directional_margin))
        alignment_score = round(alignment_score, 4)

        return (
            support_count,
            contradiction_count,
            neutral_count,
            overall_alignment,
            alignment_score,
        )

    def validate_hypothesis(
        self,
        hypothesis_text: str,
        evidence_texts: List[str],
        evidence_ids: List[str],
        hypothesis_id: str = "unknown_hyp",
    ) -> NLIResult:
        """Validate hypothesis alignment against an arbitrary set of evidence passages.

        Parameters
        ----------
        hypothesis_text : str
            The hypothesis statement text to evaluate.
        evidence_texts : list of str
            Passage texts to serve as premises.
        evidence_ids : list of str
            Identifiers for each passage text.
        hypothesis_id : str, default 'unknown_hyp'
            Unique identifier of the hypothesis.

        Returns
        -------
        NLIResult
            Aggregated NLI result containing passage-level classifications and alignment scores.
        """
        if not self.is_available():
            logger.warning(
                "NLI model '%s' is unavailable. Gracefully returning degraded neutral result without simulated data.",
                self.model_name,
            )
            return NLIResult(
                hypothesis_id=hypothesis_id,
                passage_results=[],
                support_count=0,
                contradiction_count=0,
                neutral_count=0,
                overall_alignment="neutral",
                alignment_score=0.5,
                model_used=f"{self.model_name} (unavailable)",
            )

        if not evidence_texts:
            logger.info("No evidence texts provided for hypothesis '%s'.", hypothesis_id)
            return NLIResult(
                hypothesis_id=hypothesis_id,
                passage_results=[],
                support_count=0,
                contradiction_count=0,
                neutral_count=0,
                overall_alignment="neutral",
                alignment_score=0.5,
                model_used=self.model_name,
            )

        # Prepare pairs: Premise = evidence passage, Hypothesis = generated hypothesis
        pairs = [(str(premise), str(hypothesis_text)) for premise in evidence_texts]

        # Ensure evidence_ids alignment
        aligned_ids = list(evidence_ids)
        if len(aligned_ids) < len(pairs):
            aligned_ids.extend([f"chunk_{i}" for i in range(len(aligned_ids), len(pairs))])

        batch_predictions = self.predict_batch_nli(pairs)

        passage_results: List[PassageNLI] = []
        for idx, (label, scores) in enumerate(batch_predictions):
            p_nli = PassageNLI(
                evidence_id=str(aligned_ids[idx]),
                label=label,
                entailment_score=scores["entailment"],
                contradiction_score=scores["contradiction"],
                neutral_score=scores["neutral"],
            )
            passage_results.append(p_nli)

        (
            sup_count,
            contra_count,
            neut_count,
            alignment,
            score,
        ) = self.aggregate_results(passage_results)

        logger.debug(
            "Hypothesis '%s' NLI: alignment=%s, score=%.4f (supports=%d, contradicts=%d, neutral=%d)",
            hypothesis_id,
            alignment,
            score,
            sup_count,
            contra_count,
            neut_count,
        )

        return NLIResult(
            hypothesis_id=hypothesis_id,
            passage_results=passage_results,
            support_count=sup_count,
            contradiction_count=contra_count,
            neutral_count=neut_count,
            overall_alignment=alignment,
            alignment_score=score,
            model_used=self.model_name,
        )


def save_nli_results(
    results: Union[NLIResult, List[NLIResult], Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Save NLI validation results to a JSON file.

    Parameters
    ----------
    results : NLIResult, list of NLIResult, or dict
        Results to serialize.
    output_path : str or Path
        Target filepath.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, NLIResult):
        data = results.to_dict()
    elif isinstance(results, list):
        data = [r.to_dict() if isinstance(r, NLIResult) else r for r in results]
    elif isinstance(results, dict):
        data = {k: v.to_dict() if isinstance(v, NLIResult) else v for k, v in results.items()}
    else:
        data = results

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved NLI results to %s", path)
