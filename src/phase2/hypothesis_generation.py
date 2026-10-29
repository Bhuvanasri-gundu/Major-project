"""
hypothesis_generation.py — LLM-Based Hypothesis Generation Module for Phase 2.

Generates structured, evidence-grounded scientific hypotheses bridging identified
mHealth research gaps using Google Gemini (via the `google-genai` SDK).
Grounds all hypotheses strictly in retrieved abstract-level evidence chunks
and enforces scientific rigor (testability, variable operationalization,
uncertainty reporting, and distinction between association and causation).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import google.genai as genai
    from google.genai import errors as genai_errors
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    genai_errors = None

try:
    from ..config_loader import get_project_root, load_config
except (ImportError, ValueError):
    from src.config_loader import get_project_root, load_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class Hypothesis:
    """Structured representation of a generated scientific hypothesis."""

    hypothesis_id: str
    gap_id: str
    hypothesis: str
    research_question: str
    independent_variable: str
    dependent_variable: str
    target_population: str
    intervention_or_exposure: str
    proposed_relationship: str  # 'association', 'prediction', or 'causal'
    potential_contribution: str
    supporting_evidence_ids: List[str] = field(default_factory=list)
    contradictory_evidence_ids: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    testability_explanation: str = ""
    generation_model: str = "gemini-2.0-flash"
    generation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_llm_output: Union[str, Dict[str, Any]] = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis dataclass instance to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Hypothesis:
        """Create a Hypothesis dataclass instance from a dictionary."""
        # Clean and coerce list fields if needed
        list_fields = [
            "supporting_evidence_ids",
            "contradictory_evidence_ids",
            "uncertainties",
            "limitations",
        ]
        sanitized = dict(data)
        for lf in list_fields:
            val = sanitized.get(lf)
            if val is None:
                sanitized[lf] = []
            elif isinstance(val, str):
                sanitized[lf] = [v.strip() for v in val.split(";") if v.strip()]
            elif not isinstance(val, list):
                sanitized[lf] = list(val)

        return cls(
            hypothesis_id=str(sanitized.get("hypothesis_id", "")),
            gap_id=str(sanitized.get("gap_id", "")),
            hypothesis=str(sanitized.get("hypothesis", "")),
            research_question=str(sanitized.get("research_question", "")),
            independent_variable=str(sanitized.get("independent_variable", "")),
            dependent_variable=str(sanitized.get("dependent_variable", "")),
            target_population=str(sanitized.get("target_population", "")),
            intervention_or_exposure=str(sanitized.get("intervention_or_exposure", "")),
            proposed_relationship=str(sanitized.get("proposed_relationship", "association")),
            potential_contribution=str(sanitized.get("potential_contribution", "")),
            supporting_evidence_ids=sanitized.get("supporting_evidence_ids", []),
            contradictory_evidence_ids=sanitized.get("contradictory_evidence_ids", []),
            uncertainties=sanitized.get("uncertainties", []),
            limitations=sanitized.get("limitations", []),
            testability_explanation=str(sanitized.get("testability_explanation", "")),
            generation_model=str(sanitized.get("generation_model", "unknown")),
            generation_timestamp=str(
                sanitized.get("generation_timestamp", datetime.now(timezone.utc).isoformat())
            ),
            raw_llm_output=sanitized.get("raw_llm_output", ""),
        )


class HypothesisGenerator:
    """Generates structured hypotheses bridging mHealth research gaps using Gemini."""

    REQUIRED_FIELDS = [
        "hypothesis",
        "research_question",
        "independent_variable",
        "dependent_variable",
        "target_population",
        "proposed_relationship",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_hypotheses_per_gap: int = 3,
        config: Optional[Dict[str, Any]] = None,
        use_mock: Optional[bool] = None,
    ) -> None:
        """Initialize the HypothesisGenerator.

        Parameters
        ----------
        model_name : str, optional
            Gemini model identifier. If None, loaded from config or defaults to 'gemini-2.0-flash'.
        api_key : str, optional
            Google Gemini API key. If None, checked from GEMINI_API_KEY, then GOOGLE_API_KEY.
        max_hypotheses_per_gap : int, default 3
            Maximum number of hypotheses to generate per research gap.
        config : dict, optional
            Configuration dictionary. If None, loaded using load_config().
        use_mock : bool, optional
            If True, use mock generation instead of calling live LLM API.
        """
        self.config = config or self._load_default_config()
        phase2_cfg = self.config.get("phase2", {})
        llm_cfg = phase2_cfg.get("llm", {})

        # Model resolution
        self.model_name = (
            model_name
            or llm_cfg.get("model_name")
            or "gemini-3.1-flash-lite"
        )

        # Hyperparameters
        self.max_hypotheses_per_gap = int(
            max_hypotheses_per_gap or llm_cfg.get("max_hypotheses_per_gap", 3)
        )
        self.temperature = float(llm_cfg.get("temperature", 0.3))
        self.max_output_tokens = int(llm_cfg.get("max_output_tokens", 4096))
        self.retry_attempts = int(llm_cfg.get("retry_attempts", 3))
        self.retry_delay_seconds = float(llm_cfg.get("retry_delay_seconds", 5.0))
        self.request_timeout = float(llm_cfg.get("request_timeout", 60.0))
        self.use_mock = (
            use_mock if use_mock is not None else bool(llm_cfg.get("use_mock", False))
        )

        # API key resolution
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        # Initialize Gemini Client if possible
        self.client = None
        if not GENAI_AVAILABLE:
            logger.warning(
                "google-genai package is not available. Only mock generation will function."
            )
        elif self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(
                    "HypothesisGenerator initialized with model '%s' and active Gemini client.",
                    self.model_name,
                )
            except Exception as e:
                logger.error("Failed to initialize genai.Client: %s", e)
                self.client = None
        else:
            logger.info(
                "No API key provided or found in environment (GEMINI_API_KEY/GOOGLE_API_KEY). "
                "API generation will require an API key; mock generation remains available."
            )

    @staticmethod
    def _load_default_config() -> Dict[str, Any]:
        """Safely load default configuration."""
        try:
            return load_config()
        except Exception as e:
            logger.warning("Could not load config.yaml: %s. Using default parameters.", e)
            return {}

    def _extract_gap_metadata(self, gap: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Extract normalized gap metadata from either a dict, pandas Series, or object."""
        def _get(key: str, default: Any = "") -> Any:
            if isinstance(gap, dict):
                return gap.get(key, default)
            return getattr(gap, key, default)

        concept_a = str(_get("concept_a", "")).strip()
        concept_b = str(_get("concept_b", "")).strip()

        # Derive a stable gap_id if not present
        raw_gap_id = _get("gap_id", "")
        if not raw_gap_id:
            safe_a = re.sub(r"[^\w]+", "_", concept_a.lower()).strip("_")
            safe_b = re.sub(r"[^\w]+", "_", concept_b.lower()).strip("_")
            gap_id = f"gap_{safe_a}__{safe_b}"
        else:
            gap_id = str(raw_gap_id)

        return {
            "gap_id": gap_id,
            "concept_a": concept_a,
            "concept_b": concept_b,
            "gap_score": _get("gap_score", 0.0),
            "semantic_similarity": _get("semantic_similarity", 0.0),
            "cooccurrence_score": _get("cooccurrence_score", 0.0),
            "temporal_growth_avg": _get("temporal_growth_avg", _get("temporal_growth_a", 0.0)),
            "temporal_growth_a": _get("temporal_growth_a", 0.0),
            "temporal_growth_b": _get("temporal_growth_b", 0.0),
            "trend_a": _get("trend_a", "unknown"),
            "trend_b": _get("trend_b", "unknown"),
            "papers_concept_a": _get("papers_concept_a", 0),
            "papers_concept_b": _get("papers_concept_b", 0),
            "papers_joint": _get("papers_joint", 0),
            "evidence_description": _get("evidence_description", _get("gap_note", "")),
            "supporting_topic_ids": _get("supporting_topic_ids", ""),
        }

    def _normalize_evidence_passages(
        self, evidence_passages: List[Union[Dict[str, Any], Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize evidence passages into consistent dictionary structures."""
        normalized = []
        for idx, passage in enumerate(evidence_passages):
            if isinstance(passage, dict):
                p_dict = passage
            elif hasattr(passage, "to_dict"):
                p_dict = passage.to_dict()
            elif hasattr(passage, "__dict__"):
                p_dict = passage.__dict__
            else:
                p_dict = {"text": str(passage)}

            chunk_id = str(
                p_dict.get("chunk_id")
                or p_dict.get("evidence_id")
                or p_dict.get("id")
                or f"passage_{idx+1}"
            )
            text = str(
                p_dict.get("text")
                or p_dict.get("content")
                or p_dict.get("passage")
                or p_dict.get("clean_abstract")
                or p_dict.get("abstract")
                or ""
            ).strip()
            pmid = str(p_dict.get("pmid", p_dict.get("PMID", ""))).strip()
            title = str(p_dict.get("title", p_dict.get("clean_title", ""))).strip()
            year = str(p_dict.get("year", "")).strip()

            normalized.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "pmid": pmid,
                    "title": title,
                    "year": year,
                    "concept": str(p_dict.get("concept", "")),
                    "relevance_score": p_dict.get("relevance_score", None),
                }
            )
        return normalized

    def _build_prompt(
        self,
        gap: Union[Dict[str, Any], Any],
        evidence: List[Union[Dict[str, Any], Any]],
        max_hypotheses: int = 3,
        user_theme: str = "",
        target_population: str = "",
        research_objective: str = "",
    ) -> str:
        """Build the complete, structured prompt for Gemini.

        Parameters
        ----------
        gap : dict or object
            Metadata of the research gap.
        evidence : list
            List of evidence passages/chunks.
        max_hypotheses : int, default 3
            Number of hypotheses requested.
        user_theme : str, optional
            Natural language research theme from user.
        target_population : str, optional
            User-specified target population.
        research_objective : str, optional
            User-specified research objective.

        Returns
        -------
        str
            The formatted prompt including instructions, gap context, evidence list, and JSON schema.
        """
        g = self._extract_gap_metadata(gap)
        norm_evidence = self._normalize_evidence_passages(evidence)

        evidence_section_lines = []
        if norm_evidence:
            for item in norm_evidence:
                header_parts = [f"ID: {item['chunk_id']}"]
                if item["pmid"]:
                    header_parts.append(f"PMID: {item['pmid']}")
                if item["year"]:
                    header_parts.append(f"Year: {item['year']}")
                if item["title"]:
                    header_parts.append(f"Title: {item['title']}")
                header = " | ".join(header_parts)
                evidence_section_lines.append(f"[{header}]\nContent: {item['text']}\n")
        else:
            evidence_section_lines.append("No specific literature passages retrieved for this gap.")

        evidence_block = "\n".join(evidence_section_lines)

        user_context_block = ""
        if user_theme or target_population or research_objective:
            context_lines = ["=== USER-SPECIFIED RESEARCH CONTEXT ==="]
            if user_theme:
                context_lines.append(f"- Research Theme: {user_theme}")
            if target_population:
                context_lines.append(f"- Target Population Constraint: {target_population} (MANDATORY: You MUST focus the hypothesis and target_population on this specific cohort)")
            else:
                context_lines.append("- Target Population: Unspecified by user (derive a realistic, clinically grounded patient cohort directly from the retrieved evidence)")
            if research_objective:
                context_lines.append(f"- Research Objective: {research_objective}")
            user_context_block = "\n".join(context_lines) + "\n\n"

        prompt = f"""You are an elite biomedical informatics and mHealth research methodologist.
Your task is to generate {max_hypotheses} novel, scientifically rigorous, testable, and evidence-grounded hypotheses that bridge a specific literature gap in mobile health (mHealth).

=== CORPUS & METHODOLOGICAL SCOPE ===
CRITICAL: The underlying corpus contains ONLY peer-reviewed scientific paper ABSTRACTS (not full-text papers).
- Ground all rationales and claims strictly in the evidence provided below.
- NEVER fabricate citations, authors, study numbers, or PMIDs.
- Reference specific evidence passages solely by their exact 'ID' (e.g. chunk_id) from the evidence section.
- Distinguish association from causation rigorously:
  * Use "causal" ONLY if the intervention design and biological/behavioral mechanism permit causal inference (e.g. prospective randomized controlled trial designs).
  * Otherwise, categorize relationship as "association" or "prediction".

{user_context_block}=== RESEARCH GAP CONTEXT ===
- Gap Identifier: {g['gap_id']}
- Concept A: {g['concept_a']} (Found in {g['papers_concept_a']} papers; Growth Trend: {g['trend_a']})
- Concept B: {g['concept_b']} (Found in {g['papers_concept_b']} papers; Growth Trend: {g['trend_b']})
- Semantic Relatedness: {g['semantic_similarity']:.4f} (High semantic alignment in debiased biomedical embedding space)
- Literature Co-occurrence: {g['cooccurrence_score']:.4f} (Very low joint study; only {g['papers_joint']} co-occurring papers)
- Gap Score: {g['gap_score']:.4f}
- Supporting Topics / Notes: {g['evidence_description']}

=== RETRIEVED EVIDENCE PASSAGES ({len(norm_evidence)} Passages) ===
{evidence_block}

=== INSTRUCTIONS FOR HYPOTHESIS GENERATION ===
Generate exactly {max_hypotheses} distinct hypotheses. For each hypothesis:
1. 'hypothesis': A precise, formal, and unambiguous empirical hypothesis statement.
2. 'research_question': An investigative research question addressing the gap.
3. 'independent_variable': The specific operationalized mHealth technology, sensor, algorithm, or intervention exposure.
4. 'dependent_variable': The specific operationalized clinical, behavioral, psychological, or health system outcome.
5. 'target_population': The specific demographic, clinical patient cohort, or setting (e.g., rural older adults with type 2 diabetes).
6. 'intervention_or_exposure': Concrete description of the intervention protocol, dosage, device, or exposure.
7. 'proposed_relationship': Exactly one of: "association", "prediction", or "causal".
8. 'potential_contribution': The scientific novelty and practical significance to mHealth if verified.
9. 'supporting_evidence_ids': Array of EXACT IDs (e.g. chunk_id) from the evidence block above that directly support this hypothesis.
10. 'contradictory_evidence_ids': Array of EXACT IDs from the evidence block above representing potential barriers, conflicting results, or negative findings (if none, return empty list []).
11. 'uncertainties': Array of specific biological, behavioral, technological, or ethical unknowns.
12. 'limitations': Array of practical limitations (e.g. sensor inaccuracy, digital literacy barriers, attrition).
13. 'testability_explanation': Concrete explanation of how this hypothesis can be empirically tested (e.g., 12-week micro-randomized trial, prospective observational cohort, or Pragmatic RCT).

=== RESPONSE FORMAT ===
You MUST return ONLY a valid JSON array of objects. Do not wrap in markdown or explanatory prose.
Follow this exact schema:
[
  {{
    "hypothesis": "...",
    "research_question": "...",
    "independent_variable": "...",
    "dependent_variable": "...",
    "target_population": "...",
    "intervention_or_exposure": "...",
    "proposed_relationship": "association|prediction|causal",
    "potential_contribution": "...",
    "supporting_evidence_ids": ["passage_id_1"],
    "contradictory_evidence_ids": [],
    "uncertainties": ["..."],
    "limitations": ["..."],
    "testability_explanation": "..."
  }}
]
"""
        return prompt

    def _parse_response(
        self, response_text: str, gap_id: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM JSON response with regex fallback and schema validation.

        Parameters
        ----------
        response_text : str
            Raw text output from LLM.
        gap_id : str
            Identifier of the gap being processed.

        Returns
        -------
        list of dict
            List of validated raw hypothesis dictionaries.
        """
        if not response_text or not response_text.strip():
            logger.error("LLM returned an empty response.")
            return []

        text = response_text.strip()

        # Remove Markdown formatting code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed_data = None

        # 1. Standard JSON decode attempt
        try:
            parsed_data = json.loads(text)
        except json.JSONDecodeError as jde:
            logger.warning("Direct JSON decode failed (%s). Attempting regex extraction.", jde)

        # 2. Regex fallback: search for JSON array `[...]`
        if parsed_data is None:
            array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if array_match:
                try:
                    parsed_data = json.loads(array_match.group(0))
                except json.JSONDecodeError:
                    parsed_data = None

        # 3. Regex fallback: search for top-level object `{ "hypotheses": [...] }`
        if parsed_data is None:
            obj_match = re.search(r"\{\s*\"hypotheses\"\s*:\s*(\[.*?\])\s*\}", text, re.DOTALL)
            if obj_match:
                try:
                    parsed_data = json.loads(obj_match.group(1))
                except json.JSONDecodeError:
                    parsed_data = None

        # 4. Regex fallback: individual JSON objects `{...}`
        if parsed_data is None:
            logger.info("Extracting individual JSON objects via regex.")
            object_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            candidates = []
            for m in object_matches:
                try:
                    obj = json.loads(m)
                    if isinstance(obj, dict) and "hypothesis" in obj:
                        candidates.append(obj)
                except json.JSONDecodeError:
                    continue
            if candidates:
                parsed_data = candidates

        # Format normalization
        if isinstance(parsed_data, dict):
            if "hypotheses" in parsed_data and isinstance(parsed_data["hypotheses"], list):
                parsed_data = parsed_data["hypotheses"]
            else:
                parsed_data = [parsed_data]

        if not isinstance(parsed_data, list):
            logger.error("Failed to parse valid list of hypotheses from response.")
            return []

        valid_hypotheses = []
        for idx, item in enumerate(parsed_data):
            if not isinstance(item, dict):
                continue
            if self._validate_hypothesis(item):
                item["gap_id"] = gap_id
                if not item.get("hypothesis_id"):
                    item["hypothesis_id"] = f"{gap_id}_hyp_{idx+1}"
                valid_hypotheses.append(item)
            else:
                logger.warning("Hypothesis at index %d failed validation: %s", idx, item)

        return valid_hypotheses

    def _validate_hypothesis(self, hyp_dict: Dict[str, Any]) -> bool:
        """Check required fields exist and have valid types."""
        if not isinstance(hyp_dict, dict):
            return False

        for req in self.REQUIRED_FIELDS:
            val = hyp_dict.get(req)
            if val is None or (isinstance(val, str) and not val.strip()):
                logger.debug("Missing or empty required field '%s' in hypothesis dict.", req)
                return False

        # Relationship must be recognized
        rel = str(hyp_dict.get("proposed_relationship", "")).lower()
        if "causal" in rel:
            hyp_dict["proposed_relationship"] = "causal"
        elif "predict" in rel:
            hyp_dict["proposed_relationship"] = "prediction"
        else:
            hyp_dict["proposed_relationship"] = "association"

        # Ensure list fields are indeed lists
        for lf in [
            "supporting_evidence_ids",
            "contradictory_evidence_ids",
            "uncertainties",
            "limitations",
        ]:
            val = hyp_dict.get(lf)
            if val is None:
                hyp_dict[lf] = []
            elif isinstance(val, str):
                hyp_dict[lf] = [v.strip() for v in val.split(";") if v.strip()]
            elif not isinstance(val, list):
                hyp_dict[lf] = list(val)

        return True

    def generate_hypotheses(
        self,
        gap: Union[Dict[str, Any], Any],
        evidence_passages: List[Union[Dict[str, Any], Any]],
        max_hypotheses: Optional[int] = None,
        user_theme: str = "",
        target_population: str = "",
        research_objective: str = "",
    ) -> List[Hypothesis]:
        """Generate structured hypotheses for a single research gap using Gemini.

        Parameters
        ----------
        gap : dict or object
            The gap data containing concept_a, concept_b, gap_score, etc.
        evidence_passages : list
            List of evidence chunk dictionaries or objects.
        max_hypotheses : int, optional
            Number of hypotheses to generate. Defaults to self.max_hypotheses_per_gap.
        user_theme : str, optional
            Natural language research theme from user.
        target_population : str, optional
            User-specified target population.
        research_objective : str, optional
            User-specified research objective.

        Returns
        -------
        list of Hypothesis
            List of validated Hypothesis instances.

        Raises
        ------
        ValueError
            If Gemini API key is missing or client is uninitialized.
        RuntimeError
            If API calls fail after all retry attempts.
        """
        k = max_hypotheses or self.max_hypotheses_per_gap
        gap_meta = self._extract_gap_metadata(gap)
        gap_id = gap_meta["gap_id"]

        if self.use_mock:
            logger.info("use_mock is True. Generating mock hypotheses for gap %s.", gap_id)
            return self.generate_mock_hypotheses(gap, evidence_passages, max_hypotheses=k)

        if not self.client:
            raise ValueError(
                "Gemini API key is required for hypothesis generation via LLM. "
                "Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                "add it to .env, or pass api_key to HypothesisGenerator. "
                "To generate offline test hypotheses without an API key, use generate_mock_hypotheses()."
            )

        prompt = self._build_prompt(
            gap,
            evidence_passages,
            max_hypotheses=k,
            user_theme=user_theme,
            target_population=target_population,
            research_objective=research_objective,
        )

        last_exception = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.info(
                    "Generating hypotheses for gap '%s' (Attempt %d/%d) via %s...",
                    gap_id,
                    attempt,
                    self.retry_attempts,
                    self.model_name,
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )

                response_text = response.text if response else ""
                if not response_text:
                    raise ValueError("Received empty response from Gemini API.")

                parsed_dicts = self._parse_response(response_text, gap_id)
                if not parsed_dicts:
                    raise ValueError(
                        f"Failed to parse any valid hypothesis objects from response: {response_text[:200]}"
                    )

                hypotheses: List[Hypothesis] = []
                for p_dict in parsed_dicts[:k]:
                    p_dict["generation_model"] = self.model_name
                    p_dict["raw_llm_output"] = response_text
                    hyp = Hypothesis.from_dict(p_dict)
                    hypotheses.append(hyp)

                logger.info(
                    "Successfully generated %d hypotheses for gap '%s'.",
                    len(hypotheses),
                    gap_id,
                )
                return hypotheses

            except Exception as e:
                last_exception = e
                logger.warning(
                    "Gemini API attempt %d failed for gap '%s': %s",
                    attempt,
                    gap_id,
                    str(e),
                )
                if attempt < self.retry_attempts:
                    backoff = self.retry_delay_seconds * (2 ** (attempt - 1))
                    logger.info("Backing off for %.1f seconds...", backoff)
                    time.sleep(backoff)

        raise RuntimeError(
            f"Failed to generate hypotheses for gap '{gap_id}' after {self.retry_attempts} attempts: {last_exception}"
        )

    def generate_mock_hypotheses(
        self,
        gap: Union[Dict[str, Any], Any],
        evidence_passages: List[Union[Dict[str, Any], Any]],
        max_hypotheses: int = 3,
    ) -> List[Hypothesis]:
        """Generate realistic mock hypotheses WITHOUT calling Gemini API.

        Used for automated testing, offline verification, and pipeline smoke testing.

        Parameters
        ----------
        gap : dict or object
            The research gap information.
        evidence_passages : list
            List of evidence passages or chunks.
        max_hypotheses : int, default 3
            Number of mock hypotheses to create.

        Returns
        -------
        list of Hypothesis
            List of synthetic but realistic Hypothesis instances.
        """
        gap_meta = self._extract_gap_metadata(gap)
        gap_id = gap_meta["gap_id"]
        c_a = gap_meta["concept_a"] or "mHealth intervention"
        c_b = gap_meta["concept_b"] or "clinical outcome"
        norm_ev = self._normalize_evidence_passages(evidence_passages)

        evidence_ids = [e["chunk_id"] for e in norm_ev]
        if not evidence_ids:
            evidence_ids = [f"{gap_id}_ev_mock_1", f"{gap_id}_ev_mock_2"]

        templates = [
            {
                "rel": "association",
                "h": f"Higher engagement with mobile-based {c_a} protocols is significantly associated with improved adoption and reach of {c_b}.",
                "rq": f"How is digital engagement in {c_a} related to patient-level outcomes in {c_b}?",
                "iv": f"Daily active engagement score in {c_a} application",
                "dv": f"Standardized utilization and efficacy metrics in {c_b}",
                "pop": f"Adult patient cohort indicated for {c_a} and {c_b} management",
                "exp": f"Smartphone-delivered interactive monitoring module integrating {c_a}",
                "contrib": f"Establishes initial empirical linkage between {c_a} and {c_b}, bridging the identified literature gap.",
                "test": f"Prospective observational cohort study measuring digital app telemetry against clinical outcomes over 6 months.",
            },
            {
                "rel": "prediction",
                "h": f"Continuous temporal monitoring via {c_a} predicts exacerbation events and requirement for {c_b} with high sensitivity.",
                "rq": f"Can wearable and mobile telemetry from {c_a} predict near-term clinical requirements in {c_b}?",
                "iv": f"Multi-modal biometric and self-report trend patterns collected via {c_a}",
                "dv": f"Incident acute exacerbation or referral requirement for {c_b} within 30 days",
                "pop": f"High-risk ambulatory patients transitioning between primary care and {c_b}",
                "exp": f"Passive sensor data collection coupled with daily prompt-based ecological momentary assessment",
                "contrib": f"Translates passive mHealth sensing into predictive risk stratification for {c_b}.",
                "test": f"Prospective predictive validation study evaluating area under the ROC curve (AUROC) in 300 ambulatory subjects.",
            },
            {
                "rel": "causal",
                "h": f"Implementation of a closed-loop {c_a} intervention directly reduces adverse disparities in {c_b} compared to standard care.",
                "rq": f"Does an automated feedback intervention utilizing {c_a} causally improve patient outcomes in {c_b}?",
                "iv": f"Automated closed-loop algorithmic feedback delivered via {c_a} interface",
                "dv": f"Primary clinical endpoint and access equity in {c_b}",
                "pop": f"Underserved or rural populations requiring access to {c_b}",
                "exp": f"Bi-weekly adaptive mobile guidance and task-sharing workflow with community health facilitators",
                "contrib": f"Provides definitive causal evidence for digital task-shifting and accessibility improvements in {c_b}.",
                "test": f"12-month cluster randomized controlled trial across 16 rural clinics comparing active intervention to treatment-as-usual.",
            },
        ]

        hypotheses = []
        count = min(max_hypotheses, len(templates))

        for idx in range(count):
            t = templates[idx]
            hyp_id = f"{gap_id}_hyp_{idx+1}"

            # Assign subsets of evidence IDs
            sup_ids = [evidence_ids[idx % len(evidence_ids)]]
            contra_ids = []
            if len(evidence_ids) > 1 and idx == 2:
                # Add a boundary condition / contradictory citation
                contra_ids = [evidence_ids[(idx + 1) % len(evidence_ids)]]

            hyp = Hypothesis(
                hypothesis_id=hyp_id,
                gap_id=gap_id,
                hypothesis=t["h"],
                research_question=t["rq"],
                independent_variable=t["iv"],
                dependent_variable=t["dv"],
                target_population=t["pop"],
                intervention_or_exposure=t["exp"],
                proposed_relationship=t["rel"],
                potential_contribution=t["contrib"],
                supporting_evidence_ids=sup_ids,
                contradictory_evidence_ids=contra_ids,
                uncertainties=[
                    f"Long-term adherence sustainability beyond 6 months for {c_a}.",
                    f"Differential tech literacy and smartphone availability across demographic subgroups.",
                ],
                limitations=[
                    "Corpus contains abstract-level biomedical evidence; micro-level dosage details require trial confirmation.",
                    "Potential selection bias toward tech-literate participants in initial deployments.",
                ],
                testability_explanation=t["test"],
                generation_model="mock-gemini-2.0-flash",
                generation_timestamp=datetime.now(timezone.utc).isoformat(),
                raw_llm_output="MOCK_GENERATION",
            )
            hypotheses.append(hyp)

        return hypotheses


def save_hypotheses(
    hypotheses: List[Hypothesis], output_path: Union[str, Path]
) -> None:
    """Save a list of Hypothesis objects to a JSON file.

    Parameters
    ----------
    hypotheses : list of Hypothesis
        List of hypotheses to serialize.
    output_path : str or Path
        Destination filepath.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [h.to_dict() if hasattr(h, "to_dict") else h for h in hypotheses]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d hypotheses to %s", len(hypotheses), path)


def load_hypotheses(input_path: Union[str, Path]) -> List[Hypothesis]:
    """Load hypotheses from a JSON file.

    Parameters
    ----------
    input_path : str or Path
        Path to JSON file containing serialized hypotheses.

    Returns
    -------
    list of Hypothesis
        Loaded Hypothesis instances.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Hypotheses file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    hypotheses = [Hypothesis.from_dict(d) for d in data if isinstance(d, dict)]
    logger.info("Loaded %d hypotheses from %s", len(hypotheses), path)
    return hypotheses
