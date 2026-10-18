"""
test_validation.py — Tests for evidence verification, NLI validation, rule checks, and judge.
"""

import unittest
from pathlib import Path
from src.phase2.hypothesis_generation import Hypothesis
from src.phase2.evidence_verification import (
    verify_evidence_references,
    compute_evidence_coverage,
    aggregate_verification,
)
from src.phase2.rule_checks import run_all_checks
from src.phase2.llm_judge import LLMJudge
from src.phase2.nli_validation import NLIValidator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.evidence_pool = [
            {"chunk_id": "CHK-100-0", "source_pmid": 100, "text_content": "Telemedicine improves rural outcomes."},
            {"chunk_id": "CHK-200-0", "source_pmid": 200, "text_content": "Mobile health clinics reduce transport barriers."},
        ]

    def test_evidence_reference_verification_valid(self):
        hyp = Hypothesis(
            hypothesis_id="H-1",
            gap_id="GAP-1",
            hypothesis="Telemedicine increases access in rural health clinics.",
            research_question="Does telemedicine help?",
            independent_variable="Telemedicine",
            dependent_variable="Clinic access",
            target_population="Rural adults",
            intervention_or_exposure="Mobile app",
            proposed_relationship="association",
            potential_contribution="Novel finding",
            supporting_evidence_ids=["CHK-100-0"],
            contradictory_evidence_ids=[],
        )
        res = verify_evidence_references(hyp, self.evidence_pool)
        self.assertEqual(res.verification_status, "passed")
        self.assertEqual(res.valid_ids, ["CHK-100-0"])
        self.assertEqual(res.fabricated_count, 0)

    def test_evidence_reference_verification_fabricated(self):
        hyp = Hypothesis(
            hypothesis_id="H-2",
            gap_id="GAP-1",
            hypothesis="Telemedicine increases access.",
            research_question="RQ?",
            independent_variable="Telemedicine",
            dependent_variable="Access",
            target_population="Adults",
            intervention_or_exposure="App",
            proposed_relationship="association",
            potential_contribution="Contribution",
            supporting_evidence_ids=["CHK-FABRICATED-999"],
            contradictory_evidence_ids=[],
        )
        res = verify_evidence_references(hyp, self.evidence_pool)
        self.assertIn(res.verification_status, ["failed", "partial"])
        self.assertEqual(res.fabricated_count, 1)
        self.assertIn("CHK-FABRICATED-999", res.invalid_ids)

    def test_rule_checks_pass(self):
        hyp = {
            "hypothesis_id": "H-3",
            "hypothesis": "Daily interactive smartphone application telemetry is positively associated with improved medication adherence and reduced systolic blood pressure among adult hypertensive patients in primary care settings.",
            "independent_variable": "Smartphone application usage",
            "dependent_variable": "Medication adherence",
            "target_population": "Hypertensive patients",
            "proposed_relationship": "association",
            "supporting_evidence_ids": ["CHK-100-0"],
        }
        res = run_all_checks(hyp, evidence_pool=self.evidence_pool)
        self.assertIn(res.overall_status, ["pass", "warn"])
        self.assertGreater(res.passed_count, 4)

    def test_rule_checks_flag_unsupported_causal(self):
        hyp = {
            "hypothesis_id": "H-4",
            "hypothesis": "Smartphone app causes cure of hypertension in elderly patients.",
            "independent_variable": "App",
            "dependent_variable": "Hypertension cure",
            "target_population": "Elderly",
            "proposed_relationship": "association",  # association claim using 'causes'
            "supporting_evidence_ids": ["CHK-100-0"],
        }
        res = run_all_checks(hyp, evidence_pool=self.evidence_pool)
        # Should flag causal language mismatch
        check_names = [c.check_name for c in res.checks if c.status == "fail" or c.status == "warning"]
        self.assertTrue(any("causal" in name.lower() for name in check_names))

    def test_llm_judge_mock_evaluation(self):
        judge = LLMJudge(use_mock=True)
        hyp = {
            "hypothesis_id": "H-5",
            "hypothesis": "Interactive mobile reminders increase adherence in diabetes management.",
            "independent_variable": "Reminders",
            "dependent_variable": "Adherence",
            "target_population": "Diabetes patients",
        }
        gap = {"concept_a": "mobile reminders", "concept_b": "diabetes"}
        res = judge.evaluate_hypothesis(hyp, gap, self.evidence_pool)
        self.assertIsNotNone(res)
        self.assertTrue(1 <= res.relevance_to_gap <= 5)
        self.assertTrue(1 <= res.testability <= 5)
        self.assertTrue(1 <= res.clarity <= 5)

    def test_nli_validator_availability(self):
        validator = NLIValidator()
        # Even if offline, is_available() should return a boolean without crashing
        avail = validator.is_available()
        self.assertIsInstance(avail, bool)


if __name__ == "__main__":
    unittest.main()
