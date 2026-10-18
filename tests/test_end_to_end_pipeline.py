"""
test_end_to_end_pipeline.py — Integration and End-to-End Pipeline Verification.
"""

import unittest
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.outputs_dir = PROJECT_ROOT / "outputs"

    def test_pipeline_summary_exists_and_complete(self):
        p = self.outputs_dir / "phase2_summary.json"
        self.assertTrue(p.exists(), "phase2_summary.json missing - run_phase2.py not executed")
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary.get("phase"), "Phase 2")
        self.assertGreater(summary.get("gaps_processed", 0), 0)
        self.assertGreater(summary.get("hypotheses_generated", 0), 0)
        self.assertEqual(len(summary.get("failures", [])), 0)

    def test_final_hypotheses_csv_export(self):
        p = self.outputs_dir / "final_hypotheses.csv"
        self.assertTrue(p.exists(), "final_hypotheses.csv export missing")
        df = pd.read_csv(p)
        self.assertGreater(len(df), 0)
        expected_cols = [
            "hypothesis_id", "gap_id", "hypothesis", "research_question",
            "independent_variable", "dependent_variable", "target_population",
            "proposed_relationship", "potential_contribution",
            "rank_position", "rank_score",
        ]
        for c in expected_cols:
            self.assertIn(c, df.columns, f"Missing exported column: {c}")

    def test_evidence_traceability_integrity(self):
        ev_file = self.outputs_dir / "retrieved_gap_evidence.json"
        hyp_file = self.outputs_dir / "generated_hypotheses.json"
        self.assertTrue(ev_file.exists())
        self.assertTrue(hyp_file.exists())

        with open(ev_file, "r", encoding="utf-8") as f:
            evidence = json.load(f)
        with open(hyp_file, "r", encoding="utf-8") as f:
            hyps = json.load(f)

        # Collect all valid chunk_ids
        all_chunk_ids = set()
        for gap_id, ev_list in evidence.items():
            for ev in ev_list:
                cid = ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)
                if cid:
                    all_chunk_ids.add(cid)

        # Verify that cited evidence IDs in hypotheses are traceable to the pool
        for h in hyps:
            cites = h.get("supporting_evidence_ids", [])
            for c in cites:
                self.assertIn(c, all_chunk_ids, f"Hypothesis cites unretrieved/hallucinated chunk: {c}")

    def test_evaluation_summary_exists(self):
        p = self.outputs_dir / "phase3_evaluation_summary.csv"
        self.assertTrue(p.exists(), "phase3_evaluation_summary.csv missing")
        df = pd.read_csv(p)
        self.assertGreater(len(df), 5)
        self.assertIn("metric_name", df.columns)
        self.assertIn("value", df.columns)


if __name__ == "__main__":
    unittest.main()
