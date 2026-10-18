"""
test_gap_preparation.py — Tests for candidate gap loading, validation, and prioritization.
"""

import unittest
from pathlib import Path
import pandas as pd
from src.phase2.gap_preparation import (
    load_phase1_gaps,
    validate_gap_record,
    prioritize_gaps,
    load_phase1_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGapPreparation(unittest.TestCase):
    def setUp(self):
        self.outputs_dir = PROJECT_ROOT / "outputs"

    def test_load_phase1_gaps(self):
        df = load_phase1_gaps(self.outputs_dir)
        self.assertGreater(len(df), 0)
        self.assertIn("gap_id", df.columns)
        self.assertTrue(df["gap_id"].str.startswith("GAP-").all())
        self.assertIn("priority_score", df.columns)
        self.assertIn("priority_rank", df.columns)

    def test_validate_gap_record_valid(self):
        sample = {
            "concept_a": "telemedicine",
            "concept_b": "rural health",
            "semantic_similarity": 0.82,
            "cooccurrence_score": 0.05,
            "temporal_growth_avg": 0.35,
            "gap_score": 1.10,
            "papers_concept_a": 100,
            "papers_concept_b": 50,
            "papers_joint": 5,
            "supporting_pmids_a": "123;456",
            "supporting_pmids_b": "789;012",
        }
        is_valid, issues = validate_gap_record(sample)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)

    def test_validate_gap_record_invalid_reflexive(self):
        sample = {
            "concept_a": "telemedicine",
            "concept_b": "telemedicine",
            "semantic_similarity": 1.0,
            "gap_score": 1.0,
        }
        is_valid, issues = validate_gap_record(sample)
        self.assertFalse(is_valid)
        self.assertTrue(any("Reflexive" in issue or "identical" in issue.lower() for issue in issues))

    def test_prioritize_gaps(self):
        df = load_phase1_gaps(self.outputs_dir)
        prioritized = prioritize_gaps(df, max_gaps=10)
        self.assertEqual(len(prioritized), 10)
        self.assertEqual(list(prioritized["priority_rank"]), list(range(1, 11)))
        # Verify scores are sorted descending
        scores = list(prioritized["priority_score"])
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_load_phase1_artifacts(self):
        artifacts = load_phase1_artifacts(PROJECT_ROOT)
        self.assertIn("papers", artifacts)
        self.assertIn("topic_assignments", artifacts)
        self.assertIn("document_concepts", artifacts)
        self.assertEqual(len(artifacts["papers"]), 12642)


if __name__ == "__main__":
    unittest.main()
