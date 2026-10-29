"""
llm_judge.py — LLM-as-a-Judge Quality Assessment Module for Phase 2.

Evaluates evidence-grounded hypotheses generated in mHealth research across
six rigorous evaluation dimensions:
  1. relevance_to_gap: How directly the hypothesis bridges the candidate gap.
  2. clarity: Syntactic clarity, precision, and structural readability.
  3. logical_consistency: Coherent rationale without contradictions.
  4. testability: Empirical falsifiability and clinical feasibility.
  5. evidence_alignment: Grounding in provided scientific literature.
  6. novelty_potential: Non-obvious synthesis offering genuine research value.

Supports both online Gemini evaluation (via google.genai) and offline/mock
heuristic scoring for reproducibility and continuous testing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ..config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)


def _extract_field(obj: Any, keys: Sequence[str], default: Any = "") -> Any:
    """Helper to extract a field from a dict or an object using multiple candidate keys."""
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


def _format_evidence_passages(evidence_passages: Any) -> str:
    """Format evidence passages into a clean prompt string with PMIDs preserved."""
    if not evidence_passages:
        return "No specific evidence passages provided."

    formatted_items = []
    if isinstance(evidence_passages, list):
        for i, item in enumerate(evidence_passages, 1):
            if isinstance(item, dict):
                pmid = item.get("pmid", item.get("id", "Unknown PMID"))
                title = item.get("title", "")
                text = item.get("text", item.get("abstract", item.get("clean_abstract", "")))
                prefix = f"[{i}] PMID {pmid}"
                if title:
                    prefix += f" - {title}"
                formatted_items.append(f"{prefix}\nAbstract/Excerpt: {text.strip()}")
            elif isinstance(item, str):
                formatted_items.append(f"[{i}] {item.strip()}")
            else:
                formatted_items.append(f"[{i}] {str(item).strip()}")
    elif isinstance(evidence_passages, dict):
        for i, (k, v) in enumerate(evidence_passages.items(), 1):
            formatted_items.append(f"[{i}] Evidence (Ref {k}): {str(v).strip()}")
    else:
        formatted_items.append(str(evidence_passages).strip())

    return "\n\n".join(formatted_items)


@dataclass
class JudgeResult:
    """Assessment outcome from the LLM Judge for a single hypothesis.

    Attributes
    ----------
    hypothesis_id : str
        Unique identifier of the evaluated hypothesis.
    relevance_to_gap : int
        Score 1-5 on relevance to the specific research gap.
    clarity : int
        Score 1-5 on structural clarity and syntactical quality.
    logical_consistency : int
        Score 1-5 on absence of internal contradictions and rationale logic.
    testability : int
        Score 1-5 on empirical falsifiability and operationalizability.
    evidence_alignment : int
        Score 1-5 on grounding and fidelity to retrieved scientific evidence.
    novelty_potential : int
        Score 1-5 on innovation and non-triviality of the bridged concepts.
    relevance_justification : str
        Qualitative rationale for relevance score.
    clarity_justification : str
        Qualitative rationale for clarity score.
    consistency_justification : str
        Qualitative rationale for logical consistency score.
    testability_justification : str
        Qualitative rationale for testability score.
    evidence_justification : str
        Qualitative rationale for evidence alignment score.
    novelty_justification : str
        Qualitative rationale for novelty score.
    overall_assessment : str
        Synthesized summary judgment of the hypothesis.
    model_used : str
        Model identifier or evaluator engine (e.g. 'gemini-2.0-flash', 'mock-rule-evaluator').
    timestamp : str
        ISO 8601 formatted timestamp of the evaluation.
    """

    hypothesis_id: str
    relevance_to_gap: int
    clarity: int
    logical_consistency: int
    testability: int
    evidence_alignment: int
    novelty_potential: int
    relevance_justification: str
    clarity_justification: str
    consistency_justification: str
    testability_justification: str
    evidence_justification: str
    novelty_justification: str
    overall_assessment: str
    model_used: str
    timestamp: str

    def mean_score(self) -> float:
        """Calculate the unweighted arithmetic mean across all six 1-5 dimensions."""
        scores = [
            self.relevance_to_gap,
            self.clarity,
            self.logical_consistency,
            self.testability,
            self.evidence_alignment,
            self.novelty_potential,
        ]
        return float(sum(scores) / len(scores))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        d = asdict(self)
        d["mean_score"] = round(self.mean_score(), 4)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JudgeResult:
        """Create JudgeResult instance from dictionary."""
        valid_fields = {
            "hypothesis_id",
            "relevance_to_gap",
            "clarity",
            "logical_consistency",
            "testability",
            "evidence_alignment",
            "novelty_potential",
            "relevance_justification",
            "clarity_justification",
            "consistency_justification",
            "testability_justification",
            "evidence_justification",
            "novelty_justification",
            "overall_assessment",
            "model_used",
            "timestamp",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        # Cast integer scores
        for score_key in [
            "relevance_to_gap",
            "clarity",
            "logical_consistency",
            "testability",
            "evidence_alignment",
            "novelty_potential",
        ]:
            if score_key in filtered:
                try:
                    filtered[score_key] = int(round(float(filtered[score_key])))
                except (ValueError, TypeError):
                    filtered[score_key] = 3

        return cls(**filtered)


class LLMJudge:
    """Evaluates hypothesis quality using Google Gemini or a heuristic mock evaluator."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        """Initialize LLMJudge with Gemini client or prepare mock fallback.

        Parameters
        ----------
        model_name : str, optional
            Gemini model identifier (e.g., 'gemini-2.0-flash'). If None,
            reads from config or falls back to 'gemini-2.0-flash'.
        api_key : str, optional
            Gemini API key. If None, checks GEMINI_API_KEY environment variable.
        use_mock : bool, default False
            If True, skips initializing the live client and uses evaluate_mock.
        """
        cfg = load_config()
        phase2_cfg = cfg.get("phase2", {}) if isinstance(cfg, dict) else {}

        judge_cfg = phase2_cfg.get("judge", {}) if isinstance(phase2_cfg, dict) else {}
        llm_cfg = phase2_cfg.get("llm", {}) if isinstance(phase2_cfg, dict) else {}
        default_model = (
            judge_cfg.get("model_name")
            or llm_cfg.get("model_name")
            or phase2_cfg.get("judge_model")
            or "gemini-3.1-flash-lite"
        )
        self.model_name = model_name or default_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.use_mock = use_mock or bool(judge_cfg.get("use_mock", False)) or bool(llm_cfg.get("use_mock", False))

        self.client = None
        if not self.use_mock and self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(
                    "LLMJudge initialized with google.genai client (model=%s)",
                    self.model_name,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize google.genai Client: %s. "
                    "LLMJudge will fallback to mock evaluation when evaluate_hypothesis is called.",
                    exc,
                )
                self.client = None
        else:
            logger.info(
                "No GEMINI_API_KEY detected. LLMJudge will operate in mock evaluation mode."
            )

    def _build_judge_prompt(
        self,
        hypothesis: Any,
        gap: Any,
        evidence: Any,
    ) -> str:
        """Construct a structured evaluation prompt requesting strict JSON response.

        Parameters
        ----------
        hypothesis : dict or object
            Hypothesis entity with statement, variables, population, and rationale.
        gap : dict or object
            Research gap containing concept_a, concept_b, gap_score, note/description.
        evidence : Any
            Retrieved evidence passages / excerpts from abstracts.

        Returns
        -------
        str
            Formatted prompt string.
        """
        hyp_id = _extract_field(hypothesis, ["hypothesis_id", "id"], "HYP-UNKNOWN")
        statement = _extract_field(
            hypothesis,
            ["hypothesis_text", "hypothesis_statement", "text", "statement"],
            "No statement provided",
        )
        iv = _extract_field(
            hypothesis, ["independent_variable", "iv", "intervention"], "Unspecified IV"
        )
        dv = _extract_field(
            hypothesis, ["dependent_variable", "dv", "outcome"], "Unspecified DV"
        )
        pop = _extract_field(
            hypothesis, ["target_population", "population"], "Unspecified population"
        )
        mech = _extract_field(
            hypothesis, ["proposed_mechanism", "mechanism", "rationale"], "Unspecified mechanism"
        )
        rel_type = _extract_field(
            hypothesis, ["proposed_relationship", "relationship_type", "relationship"], "Unspecified"
        )

        gap_a = _extract_field(gap, ["concept_a", "concept1"], "Concept A")
        gap_b = _extract_field(gap, ["concept_b", "concept2"], "Concept B")
        gap_score = _extract_field(gap, ["gap_score", "score"], "N/A")
        gap_desc = _extract_field(
            gap, ["evidence_description", "gap_note", "description"], "Temporal-semantic literature gap"
        )

        evidence_str = _format_evidence_passages(evidence)

        prompt = f"""You are an expert scientific evaluator specializing in mobile health (mHealth) research methodology.
Assess the quality and scientific rigor of the following candidate mHealth research hypothesis.

### CONTEXT: CANDIDATE RESEARCH GAP
- Concept A: {gap_a}
- Concept B: {gap_b}
- Gap Score: {gap_score}
- Gap Description: {gap_desc}

### CANDIDATE HYPOTHESIS [ID: {hyp_id}]
- Statement: {statement}
- Independent Variable (IV): {iv}
- Dependent Variable (DV): {dv}
- Target Population: {pop}
- Proposed Relationship: {rel_type}
- Proposed Mechanism / Rationale: {mech}

### SUPPORTING SCIENTIFIC EVIDENCE (Retrieved Abstracts)
{evidence_str}

### EVALUATION RUBRIC (Rate each dimension on an integer scale from 1 to 5):
1. relevance_to_gap (1-5): Does the hypothesis meaningfully bridge Concept A ({gap_a}) and Concept B ({gap_b})?
   1 = Irrelevant; 3 = Tangentially connected; 5 = Directly and explicitly unifies both concepts.
2. clarity (1-5): Is the hypothesis syntactically unambiguous, formal, and clearly formulated?
   1 = Confusing/unintelligible; 3 = Comprehensible with minor ambiguity; 5 = Exceptionally crisp, precise scientific English.
3. logical_consistency (1-5): Are the variables, mechanism, and proposed relationship logically sound without self-contradiction?
   1 = Internally contradictory; 3 = Plausible but mechanism has gaps; 5 = Flawless causal/associational reasoning.
4. testability (1-5): Can this hypothesis be empirically tested/falsified using feasible mHealth trial designs, sensors, apps, or observational cohorts?
   1 = Untestable / unfalsifiable; 3 = Testable with significant methodology hurdles; 5 = Highly operationalizable with standard mHealth protocols.
5. evidence_alignment (1-5): Is the hypothesis consistent with the provided abstract passages without hallucinating contradictory empirical facts?
   1 = Directly contradicts cited evidence; 3 = Neutral/partial alignment; 5 = Strongly grounded in and justified by the evidence.
6. novelty_potential (1-5): Does this hypothesis represent an innovative scientific inquiry rather than a trivial, already-proven truism?
   1 = Completely trivial or already universal common sense; 3 = Incremental variation; 5 = High-impact, creative synthesis.

### INSTRUCTIONS:
Respond ONLY with a valid JSON object matching this exact schema:
{{
  "relevance_to_gap": <int 1-5>,
  "relevance_justification": "<string>",
  "clarity": <int 1-5>,
  "clarity_justification": "<string>",
  "logical_consistency": <int 1-5>,
  "consistency_justification": "<string>",
  "testability": <int 1-5>,
  "testability_justification": "<string>",
  "evidence_alignment": <int 1-5>,
  "evidence_justification": "<string>",
  "novelty_potential": <int 1-5>,
  "novelty_justification": "<string>",
  "overall_assessment": "<string summarizing strengths and key limitations>"
}}
Do NOT output any markdown backticks or commentary outside the JSON object.
"""
        return prompt

    def _parse_judge_response(
        self,
        response_text: str,
        hypothesis_id: str,
        model_used: str,
    ) -> JudgeResult:
        """Parse JSON response from LLM with robust multi-stage fallbacks.

        Parameters
        ----------
        response_text : str
            Raw text response returned by the LLM.
        hypothesis_id : str
            Hypothesis ID for tracking.
        model_used : str
            Identifier of the model used.

        Returns
        -------
        JudgeResult
            Validated JudgeResult object.
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        cleaned = response_text.strip()

        # Remove markdown fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        json_obj: Optional[Dict[str, Any]] = None
        try:
            json_obj = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to extract JSON substring between first { and last }
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    json_obj = json.loads(match.group(1))
                except json.JSONDecodeError as exc:
                    logger.warning("Failed regex JSON recovery for %s: %s", hypothesis_id, exc)

        if not json_obj or not isinstance(json_obj, dict):
            logger.error(
                "Unable to parse LLM judge response for %s. Raw response preview: %s",
                hypothesis_id,
                cleaned[:200],
            )
            return JudgeResult(
                hypothesis_id=hypothesis_id,
                relevance_to_gap=3,
                clarity=3,
                logical_consistency=3,
                testability=3,
                evidence_alignment=3,
                novelty_potential=3,
                relevance_justification="Fallback: Parsing error from LLM response.",
                clarity_justification="Fallback: Parsing error from LLM response.",
                consistency_justification="Fallback: Parsing error from LLM response.",
                testability_justification="Fallback: Parsing error from LLM response.",
                evidence_justification="Fallback: Parsing error from LLM response.",
                novelty_justification="Fallback: Parsing error from LLM response.",
                overall_assessment="Judge evaluation could not be parsed; assigned neutral baseline scores.",
                model_used=model_used,
                timestamp=now_ts,
            )

        def _get_clamped_score(key: str, default: int = 3) -> int:
            val = json_obj.get(key, default)
            try:
                score = int(round(float(val)))
                return max(1, min(5, score))
            except (ValueError, TypeError):
                return default

        return JudgeResult(
            hypothesis_id=hypothesis_id,
            relevance_to_gap=_get_clamped_score("relevance_to_gap"),
            clarity=_get_clamped_score("clarity"),
            logical_consistency=_get_clamped_score("logical_consistency"),
            testability=_get_clamped_score("testability"),
            evidence_alignment=_get_clamped_score("evidence_alignment"),
            novelty_potential=_get_clamped_score("novelty_potential"),
            relevance_justification=str(
                json_obj.get("relevance_justification", "Relevance assessed by LLM.")
            ),
            clarity_justification=str(
                json_obj.get("clarity_justification", "Clarity assessed by LLM.")
            ),
            consistency_justification=str(
                json_obj.get("consistency_justification", "Logical consistency assessed by LLM.")
            ),
            testability_justification=str(
                json_obj.get("testability_justification", "Testability assessed by LLM.")
            ),
            evidence_justification=str(
                json_obj.get("evidence_justification", "Evidence alignment assessed by LLM.")
            ),
            novelty_justification=str(
                json_obj.get("novelty_justification", "Novelty assessed by LLM.")
            ),
            overall_assessment=str(
                json_obj.get("overall_assessment", "Overall synthesis completed by LLM judge.")
            ),
            model_used=model_used,
            timestamp=now_ts,
        )

    def evaluate_hypothesis(
        self,
        hypothesis: Any,
        gap: Any,
        evidence_passages: Any = None,
    ) -> JudgeResult:
        """Rate hypothesis on 6 dimensions using Gemini, falling back to mock on failure.

        Parameters
        ----------
        hypothesis : dict or object
            Hypothesis data.
        gap : dict or object
            Candidate research gap data.
        evidence_passages : list or dict, optional
            Retrieved supporting passages or abstracts.

        Returns
        -------
        JudgeResult
            Result with scores 1-5 and justifications.
        """
        hyp_id = _extract_field(hypothesis, ["hypothesis_id", "id"], "HYP-UNKNOWN")

        if not self.client:
            logger.info(
                "Gemini client not configured or API key unavailable. Using mock evaluation for %s.",
                hyp_id,
            )
            return self.evaluate_mock(hypothesis, gap, evidence_passages)

        prompt = self._build_judge_prompt(hypothesis, gap, evidence_passages)

        try:
            logger.info("Calling Gemini judge (%s) for hypothesis %s...", self.model_name, hyp_id)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            response_text = response.text or ""
            return self._parse_judge_response(
                response_text=response_text,
                hypothesis_id=hyp_id,
                model_used=self.model_name,
            )
        except Exception as exc:
            logger.warning(
                "LLMJudge API call failed for %s (%s). Falling back to mock evaluation.",
                hyp_id,
                exc,
            )
            mock_res = self.evaluate_mock(hypothesis, gap, evidence_passages)
            mock_res.overall_assessment += f" (Note: Gemini API fallback due to: {exc})"
            return mock_res

    def evaluate_mock(
        self,
        hypothesis: Any,
        gap: Any,
        evidence_passages: Any = None,
    ) -> JudgeResult:
        """Heuristic rule-based evaluation for testing and reproducible offline execution.

        Scores dimensions using feature-based text heuristics:
          - relevance_to_gap: lexical and conceptual overlap with gap concepts.
          - clarity: sentence structure, readability, word count in sweet spot.
          - logical_consistency: presence of explicit directional mechanism.
          - testability: presence of operational, measurable, or clinical keywords.
          - evidence_alignment: citations count and passage overlap.
          - novelty_potential: synthesis between distinct concepts.

        Parameters
        ----------
        hypothesis : dict or object
            Hypothesis data.
        gap : dict or object
            Candidate research gap data.
        evidence_passages : list or dict, optional
            Retrieved evidence passages.

        Returns
        -------
        JudgeResult
            Deterministic, rule-calculated JudgeResult.
        """
        hyp_id = _extract_field(hypothesis, ["hypothesis_id", "id"], "HYP-MOCK")
        statement = _extract_field(
            hypothesis,
            ["hypothesis_text", "hypothesis_statement", "text", "statement"],
            "",
        ).strip()
        iv = _extract_field(
            hypothesis, ["independent_variable", "iv", "intervention"], ""
        ).strip()
        dv = _extract_field(
            hypothesis, ["dependent_variable", "dv", "outcome"], ""
        ).strip()
        pop = _extract_field(
            hypothesis, ["target_population", "population"], ""
        ).strip()
        mech = _extract_field(
            hypothesis, ["proposed_mechanism", "mechanism", "rationale"], ""
        ).strip()

        gap_a = str(_extract_field(gap, ["concept_a", "concept1"], "")).lower().strip()
        gap_b = str(_extract_field(gap, ["concept_b", "concept2"], "")).lower().strip()

        stmt_lower = statement.lower()
        words = stmt_lower.split()
        word_count = len(words)

        # 1. Relevance to gap (1-5)
        # Check if concept A and concept B terms appear in statement, iv, dv, or mechanism
        combined_text = f"{stmt_lower} {iv.lower()} {dv.lower()} {mech.lower()}"
        a_present = any(term in combined_text for term in gap_a.split() if len(term) > 3) if gap_a else False
        b_present = any(term in combined_text for term in gap_b.split() if len(term) > 3) if gap_b else False

        if a_present and b_present:
            rel_score = 5
            rel_just = f"Both concept terms '{gap_a}' and '{gap_b}' are explicitly integrated into the hypothesis."
        elif a_present or b_present:
            rel_score = 4
            rel_just = f"Partial conceptual bridge: touches on gap concepts but could emphasize both sides more symmetrically."
        else:
            rel_score = 3
            rel_just = "Indirect relevance to the candidate research gap concepts."

        # 2. Clarity (1-5)
        # Ideal length between 25 and 90 words, ending with punctuation, mentions population
        if word_count < 15:
            clarity_score = 2
            clarity_just = "Hypothesis is excessively terse (<15 words), omitting operational details."
        elif word_count > 120:
            clarity_score = 3
            clarity_just = "Hypothesis is overly verbose (>120 words), compromising conciseness."
        else:
            # Check for structured components
            has_if_then_or_direct = any(k in stmt_lower for k in ["lead to", "associated with", "increase", "decrease", "improve", "reduce", "predict", "enhance"])
            if has_if_then_or_direct and (iv and dv):
                clarity_score = 5
                clarity_just = "Well-structured directional formulation with clearly delineated independent and dependent variables."
            else:
                clarity_score = 4
                clarity_just = "Adequate clarity; grammatical and easily understood, with minor room for tighter phrasing."

        # 3. Logical consistency (1-5)
        # Check whether IV and DV are distinct and whether a mechanism exists
        if iv and dv and iv.lower() == dv.lower():
            consist_score = 1
            consist_just = "Severe logical flaw: Independent and dependent variables are identical."
        elif len(mech) > 20:
            consist_score = 5
            consist_just = "Coherent logical framework supported by an explicit biological/behavioral mechanism."
        elif len(mech) > 0:
            consist_score = 4
            consist_just = "Logically sound relationship; mechanism is outlined though brief."
        else:
            consist_score = 3
            consist_just = "Plausible relationship, but lacks explicit underlying mechanistic justification."

        # 4. Testability (1-5)
        # Specific measurement/sensor/app/clinical outcome terms
        testable_keywords = [
            "sensor", "wearable", "app", "mobile", "adherence", "compliance",
            "rate", "frequency", "score", "level", "reduction", "increase",
            "hba1c", "blood pressure", "steps", "physical activity", "intervention",
            "randomized", "trial", "cohort", "longitudinal", "efficacy", "outcome"
        ]
        testable_hits = sum(1 for kw in testable_keywords if kw in stmt_lower or kw in dv.lower())
        if testable_hits >= 3 and pop:
            test_score = 5
            test_just = f"Highly operationalizable in mHealth with {testable_hits} specific measurable variables and defined population."
        elif testable_hits >= 1:
            test_score = 4
            test_just = "Empirically testable using standard mobile health measurement instruments."
        else:
            test_score = 3
            test_just = "Testable in principle, but variables would require substantial operational definition."

        # 5. Evidence alignment (1-5)
        evidence_ids = _extract_field(
            hypothesis,
            ["supporting_evidence_ids", "evidence_ids", "pmids", "supporting_pmids"],
            [],
        )
        n_passages = len(evidence_passages) if isinstance(evidence_passages, (list, dict)) else 0
        n_ev_ids = len(evidence_ids) if isinstance(evidence_ids, list) else 0

        if n_passages >= 2 or n_ev_ids >= 2:
            ev_score = 5
            ev_just = f"Strong grounding supported by multiple empirical abstract references ({max(n_passages, n_ev_ids)} citations)."
        elif n_passages >= 1 or n_ev_ids >= 1:
            ev_score = 4
            ev_just = "Supported by direct reference to literature corpus evidence."
        else:
            ev_score = 3
            ev_just = "Consistent with general mHealth literature, but lacks dense specific citations."

        # 6. Novelty potential (1-5)
        gap_score = _extract_field(gap, ["gap_score", "score"], 0.5)
        gap_score_val = 0.0
        try:
            gap_score_val = float(gap_score)
        except (ValueError, TypeError):
            gap_score_val = 0.5

        if gap_score_val > 1.0 or (a_present and b_present and testable_hits >= 2):
            nov_score = 5
            nov_just = "Bridges historically disparate concepts with high temporal growth potential."
        elif gap_score_val > 0.5:
            nov_score = 4
            nov_just = "Offers innovative cross-domain application between emerging mHealth constructs."
        else:
            nov_score = 3
            nov_just = "Moderate novelty; represents an intuitive extension of existing mobile health findings."

        now_ts = datetime.now(timezone.utc).isoformat()
        overall = (
            f"Mock Evaluation Summary: Hypothesis achieves a solid composite profile "
            f"(Relevance: {rel_score}/5, Testability: {test_score}/5, Evidence: {ev_score}/5). "
            f"Directional claim is formulated with clear scientific intent."
        )

        return JudgeResult(
            hypothesis_id=hyp_id,
            relevance_to_gap=rel_score,
            clarity=clarity_score,
            logical_consistency=consist_score,
            testability=test_score,
            evidence_alignment=ev_score,
            novelty_potential=nov_score,
            relevance_justification=rel_just,
            clarity_justification=clarity_just,
            consistency_justification=consist_just,
            testability_justification=test_just,
            evidence_justification=ev_just,
            novelty_justification=nov_just,
            overall_assessment=overall,
            model_used="mock-rule-evaluator",
            timestamp=now_ts,
        )


def save_judge_results(
    results: Union[List[JudgeResult], Dict[str, JudgeResult], Sequence[Any]],
    output_path: Union[str, Path],
) -> Path:
    """Save judge evaluation results to a JSON file.

    Parameters
    ----------
    results : list or dict of JudgeResult
        Collection of JudgeResult instances.
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

    logger.info("Saved %d judge result records to %s", len(results), path)
    return path
