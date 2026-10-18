"""
test_hypothesis_generation.py — Tests for LLM JSON parsing, mock generation, and missing credentials.
"""

import unittest
from pathlib import Path
from src.phase2.hypothesis_generation import HypothesisGenerator, Hypothesis

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestHypothesisGeneration(unittest.TestCase):
    def setUp(self):
        self.generator = HypothesisGenerator(use_mock=True)
        self.sample_gap = {
            "gap_id": "GAP-TEST-001",
            "concept_a": "mobile health",
            "concept_b": "rural health services",
            "gap_score": 1.15,
            "temporal_growth_avg": 0.40,
            "papers_joint": 3,
            "supporting_topic_ids": "0;1",
            "evidence_description": "Mobile health and rural health services show high semantic similarity but low co-occurrence.",
        }
        self.sample_evidence = [
            {
                "chunk_id": "CHK-100-0",
                "source_pmid": 100,
                "paper_title": "Rural mHealth",
                "text_content": "Implementing mHealth in rural communities increases access to clinics.",
            }
        ]

    def test_mock_generation_schema(self):
        hyps = self.generator.generate_hypotheses(self.sample_gap, self.sample_evidence, max_hypotheses=2)
        self.assertEqual(len(hyps), 2)
        for h in hyps:
            self.assertIsInstance(h, Hypothesis)
            self.assertEqual(h.gap_id, "GAP-TEST-001")
            self.assertTrue(len(h.hypothesis) > 10)
            self.assertTrue(len(h.independent_variable) > 0)
            self.assertTrue(len(h.dependent_variable) > 0)
            self.assertIn(h.proposed_relationship, ["association", "prediction", "causal"])
            self.assertIsInstance(h.supporting_evidence_ids, list)

    def test_json_parsing_clean(self):
        raw_json = """
        [
            {
                "hypothesis": "Testable hypothesis text.",
                "research_question": "RQ text?",
                "independent_variable": "IV",
                "dependent_variable": "DV",
                "target_population": "Adults",
                "intervention_or_exposure": "Protocol",
                "proposed_relationship": "association",
                "potential_contribution": "Contribution",
                "supporting_evidence_ids": ["CHK-100-0"],
                "contradictory_evidence_ids": []
            }
        ]
        """
        parsed = self.generator._parse_response(raw_json, "GAP-001")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["hypothesis"], "Testable hypothesis text.")

    def test_json_parsing_with_markdown_fences(self):
        raw_with_fences = """```json
        [
            {
                "hypothesis": "Fenced hypothesis statement.",
                "research_question": "RQ?",
                "independent_variable": "IV",
                "dependent_variable": "DV",
                "target_population": "Adults",
                "proposed_relationship": "prediction"
            }
        ]
        ```"""
        parsed = self.generator._parse_response(raw_with_fences, "GAP-001")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["hypothesis"], "Fenced hypothesis statement.")

    def test_malformed_json_fallback(self):
        malformed = "Here is the result: [ { 'hypothesis': 'Malformed single quote' } ] and some extra text."
        # Should gracefully return empty or best effort without raising uncaught exception
        try:
            parsed = self.generator._parse_response(malformed, "GAP-001")
            self.assertIsInstance(parsed, list)
        except Exception as e:
            self.fail(f"_parse_response raised an unexpected exception on malformed text: {e}")

    def test_missing_api_key_raises_error_when_not_mock(self):
        # When use_mock is False and client is None, generate_hypotheses must raise ValueError
        live_gen = HypothesisGenerator(use_mock=False, api_key=None)
        live_gen.client = None
        with self.assertRaises(ValueError):
            live_gen.generate_hypotheses(self.sample_gap, self.sample_evidence)


if __name__ == "__main__":
    unittest.main()
