"""
scripts/patch_llm_judge.py — Add batch evaluation capability to LLMJudge.
Evaluates all candidate hypotheses in a single structured Gemini API call,
reducing evaluation latency from ~20s down to ~3s while preserving all 6 rubrics.
"""

from pathlib import Path

JUDGE_PY = Path("src/phase2/llm_judge.py")
content = JUDGE_PY.read_text(encoding="utf-8")

batch_methods = '''    def _build_batch_judge_prompt(
        self,
        hypotheses: Sequence[Any],
        gap: Any,
        evidence: Any,
    ) -> str:
        """Construct a structured multi-hypothesis evaluation prompt requesting strict JSON array response."""
        gap_a = _extract_field(gap, ["concept_a", "concept1"], "Concept A")
        gap_b = _extract_field(gap, ["concept_b", "concept2"], "Concept B")
        gap_score = _extract_field(gap, ["gap_score", "score"], "N/A")
        gap_desc = _extract_field(
            gap, ["evidence_description", "gap_note", "description"], "Temporal-semantic literature gap"
        )
        evidence_str = _format_evidence_passages(evidence)

        hypotheses_blocks = []
        for i, h in enumerate(hypotheses, 1):
            hid = _extract_field(h, ["hypothesis_id", "id"], f"HYP-{i}")
            statement = _extract_field(
                h, ["hypothesis_text", "hypothesis_statement", "text", "statement"], "No statement provided"
            )
            iv = _extract_field(h, ["independent_variable", "iv", "intervention"], "Unspecified IV")
            dv = _extract_field(h, ["dependent_variable", "dv", "outcome"], "Unspecified DV")
            pop = _extract_field(h, ["target_population", "population"], "Unspecified population")
            mech = _extract_field(h, ["proposed_mechanism", "mechanism", "rationale"], "Unspecified mechanism")
            rel_type = _extract_field(h, ["proposed_relationship", "relationship_type", "relationship"], "Unspecified")

            hypotheses_blocks.append(
                f"--- HYPOTHESIS {i} [ID: {hid}] ---\\n"
                f"- Statement: {statement}\\n"
                f"- Independent Variable (IV): {iv}\\n"
                f"- Dependent Variable (DV): {dv}\\n"
                f"- Target Population: {pop}\\n"
                f"- Proposed Relationship: {rel_type}\\n"
                f"- Proposed Mechanism / Rationale: {mech}\\n"
            )

        joined_hypotheses = "\\n".join(hypotheses_blocks)

        prompt = f"""You are an expert scientific evaluator specializing in mobile health (mHealth) research methodology.
Assess the quality and scientific rigor of the following {len(hypotheses)} candidate mHealth research hypotheses.

### CONTEXT: CANDIDATE RESEARCH GAP
- Concept A: {gap_a}
- Concept B: {gap_b}
- Gap Score: {gap_score}
- Gap Description: {gap_desc}

### SUPPORTING SCIENTIFIC EVIDENCE (Retrieved Abstracts)
{evidence_str}

### CANDIDATE HYPOTHESES TO EVALUATE
{joined_hypotheses}

### EVALUATION RUBRIC (Rate each dimension on an integer scale from 1 to 5 for every hypothesis):
1. relevance_to_gap (1-5): Does the hypothesis meaningfully bridge Concept A ({gap_a}) and Concept B ({gap_b})?
2. clarity (1-5): Is the hypothesis syntactically unambiguous, formal, and clearly formulated?
3. logical_consistency (1-5): Are the variables, mechanism, and proposed relationship logically sound without self-contradiction?
4. testability (1-5): Can this hypothesis be empirically tested/falsified using feasible mHealth trial designs, sensors, apps, or observational cohorts?
5. evidence_alignment (1-5): Is the hypothesis consistent with the provided abstract passages without hallucinating contradictory empirical facts?
6. novelty_potential (1-5): Does this hypothesis represent an innovative scientific inquiry rather than a trivial, already-proven truism?

### INSTRUCTIONS:
Respond ONLY with a valid JSON array of objects, one object per candidate hypothesis, matching this exact schema:
[
  {{
    "hypothesis_id": "<exact hypothesis ID from above>",
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
]
Do NOT output any markdown backticks or commentary outside the JSON array.
"""
        return prompt

    def evaluate_hypotheses_batch(
        self,
        hypotheses: Sequence[Any],
        gap: Any,
        evidence_passages: Any = None,
    ) -> Dict[str, JudgeResult]:
        """Evaluate multiple candidate hypotheses in a single batch request to Gemini.

        Reduces network latency from O(N) calls to O(1) single API call while preserving
        all 6 rubric dimensions and qualitative rationales.

        Parameters
        ----------
        hypotheses : sequence of hypothesis objects or dicts
            Candidate hypotheses to evaluate.
        gap : dict or object
            Candidate research gap data.
        evidence_passages : list or dict, optional
            Retrieved supporting passages or abstracts.

        Returns
        -------
        dict of str -> JudgeResult
            Mapping from hypothesis_id to validated JudgeResult.
        """
        if not hypotheses:
            return {}

        results: Dict[str, JudgeResult] = {}

        # If only 1 hypothesis or no live client, use single evaluate_hypothesis
        if len(hypotheses) == 1 or not self.client or self.use_mock:
            for h in hypotheses:
                hid = _extract_field(h, ["hypothesis_id", "id"], "HYP-UNKNOWN")
                results[hid] = self.evaluate_hypothesis(h, gap, evidence_passages)
            return results

        prompt = self._build_batch_judge_prompt(hypotheses, gap, evidence_passages)

        try:
            logger.info(
                "Calling Gemini batch judge (%s) for %d hypotheses...",
                self.model_name,
                len(hypotheses),
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            response_text = response.text or ""
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                import re
                cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"\\s*```$", "", cleaned)

            parsed_list = json.loads(cleaned)
            if isinstance(parsed_list, dict) and "hypotheses" in parsed_list:
                parsed_list = parsed_list["hypotheses"]

            if isinstance(parsed_list, list):
                for item in parsed_list:
                    if isinstance(item, dict):
                        hid = str(item.get("hypothesis_id", ""))
                        if hid:
                            results[hid] = self._parse_judge_response(
                                response_text=json.dumps(item),
                                hypothesis_id=hid,
                                model_used=self.model_name,
                            )
            logger.info("Batch Gemini judge successfully evaluated %d/%d hypotheses.", len(results), len(hypotheses))
        except Exception as exc:
            logger.warning("Batch Gemini judge call failed (%s). Falling back to individual/mock evaluations.", exc)

        # Fallback for any hypothesis missing from batch output
        for h in hypotheses:
            hid = _extract_field(h, ["hypothesis_id", "id"], "HYP-UNKNOWN")
            if hid not in results:
                logger.info("Evaluating missing hypothesis %s individually...", hid)
                results[hid] = self.evaluate_hypothesis(h, gap, evidence_passages)

        return results

'''

target_anchor = "    def evaluate_hypothesis("
if target_anchor in content:
    content = content.replace(target_anchor, batch_methods + target_anchor)
    JUDGE_PY.write_text(content, encoding="utf-8")
    print("llm_judge.py patched successfully with evaluate_hypotheses_batch!")
else:
    print("Could not find target_anchor in llm_judge.py")
