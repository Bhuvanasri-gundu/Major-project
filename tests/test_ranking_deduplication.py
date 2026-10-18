"""
test_ranking_deduplication.py — Tests for hypothesis ranking formulas and semantic deduplication.
"""

import unittest
from pathlib import Path
from src.phase2.ranking import rank_hypotheses, RankedHypothesis
from src.phase2.deduplication import SemanticDeduplicator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRankingDeduplication(unittest.TestCase):
    def setUp(self):
        self.hypotheses = [
            {
                "hypothesis_id": "H-TOP",
                "gap_id": "GAP-1",
                "hypothesis": "Digital health interventions improve patient outcomes in rural settings.",
                "independent_variable": "Digital health",
                "dependent_variable": "Patient outcomes",
                "target_population": "Rural adults",
                "supporting_evidence_ids": ["CHK-1", "CHK-2"],
            },
            {
                "hypothesis_id": "H-LOW",
                "gap_id": "GAP-1",
                "hypothesis": "Digital technology may possibly be related to something in healthcare.",
                "independent_variable": "Tech",
                "dependent_variable": "Something",
                "target_population": "People",
                "supporting_evidence_ids": [],
            },
        ]

    def test_ranking_formula_weights(self):
        # Provide simulated verifications and judge scores
        verifications = {
            "H-TOP": {"evidence_coverage": 0.8, "coverage_ratio": 0.8},
            "H-LOW": {"evidence_coverage": 0.0, "coverage_ratio": 0.0},
        }
        judge_results = {
            "H-TOP": {"mean_score": 4.8, "testability": 5},
            "H-LOW": {"mean_score": 2.0, "testability": 2},
        }
        rankings = rank_hypotheses(self.hypotheses, verifications=verifications, judge_results=judge_results)
        self.assertEqual(len(rankings), 2)
        # H-TOP should be rank position 1
        self.assertEqual(rankings[0].hypothesis_id, "H-TOP")
        self.assertGreater(rankings[0].rank_score, rankings[1].rank_score)
        self.assertEqual(rankings[0].rank_position, 1)
        self.assertEqual(rankings[1].rank_position, 2)

    def test_semantic_deduplication_merges_identical(self):
        duplicate_hyps = [
            {
                "hypothesis_id": "H-1",
                "hypothesis": "Smartphone app improves hypertension medication adherence in older adults.",
                "independent_variable": "Smartphone app",
                "dependent_variable": "Medication adherence",
                "target_population": "Older adults",
                "rank_score": 0.85,
            },
            {
                "hypothesis_id": "H-2",
                "hypothesis": "Smartphone application improves hypertension medication adherence in elderly adults.",
                "independent_variable": "Smartphone app",
                "dependent_variable": "Medication adherence",
                "target_population": "Older adults",
                "rank_score": 0.75,
            },
        ]
        deduplicator = SemanticDeduplicator(threshold=0.80)
        res = deduplicator.deduplicate(duplicate_hyps)
        self.assertEqual(res.total_input, 2)
        self.assertEqual(res.total_output, 1)
        self.assertEqual(res.kept_ids, ["H-1"])
        self.assertIn("H-2", res.removed_ids)

    def test_deduplication_preserves_substantive_differences(self):
        # Similar wording but completely different variables
        distinct_hyps = [
            {
                "hypothesis_id": "H-A",
                "hypothesis": "Mobile messaging improves pediatric asthma inhaler compliance.",
                "independent_variable": "Mobile messaging",
                "dependent_variable": "Asthma inhaler compliance",
                "target_population": "Pediatric patients",
            },
            {
                "hypothesis_id": "H-B",
                "hypothesis": "Mobile messaging improves geriatric diabetes insulin adherence.",
                "independent_variable": "Mobile messaging",
                "dependent_variable": "Diabetes insulin adherence",
                "target_population": "Geriatric patients",
            },
        ]
        deduplicator = SemanticDeduplicator(threshold=0.80)
        res = deduplicator.deduplicate(distinct_hyps)
        # Should NOT merge because populations and dependent variables differ substantively
        self.assertEqual(res.total_output, 2)
        self.assertIn("H-A", res.kept_ids)
        self.assertIn("H-B", res.kept_ids)


if __name__ == "__main__":
    unittest.main()
