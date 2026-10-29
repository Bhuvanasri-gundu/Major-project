"""
tests/test_api_endpoints.py — Comprehensive unittest suite for FastAPI backend endpoints.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("gemini_api_configured", data)
        self.assertEqual(data["corpus_screened_papers"], 12642)
        self.assertEqual(data["candidate_gaps_count"], 50)

    def test_live_suggestions_endpoint(self):
        response = self.client.get("/api/suggestions?q=rural")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "rural")
        self.assertIn("suggestions", data)
        self.assertGreater(len(data["suggestions"]), 0)
        types = {s["type"] for s in data["suggestions"]}
        self.assertTrue(bool(types.intersection({"gap", "concept", "topic"})))

    def test_suggestions_empty_query(self):
        response = self.client.get("/api/suggestions?q=a")
        self.assertEqual(response.status_code, 422)

    def test_get_gaps_endpoint(self):
        response = self.client.get("/api/gaps?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 10)
        self.assertIn("concept_a", data[0])
        self.assertIn("concept_b", data[0])
        self.assertIn("gap_score", data[0])

    def test_evaluation_endpoint(self):
        response = self.client.get("/api/evaluation")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ir_metrics", data)
        self.assertIn("grounding_audit", data)
        self.assertIn("baseline_comparisons", data)
        self.assertIn("ablation_study", data)

    def test_get_latest_run_endpoint(self):
        response = self.client.get("/api/runs/latest")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("run_id", data)
        self.assertIn("theme", data)
        self.assertIn("hypotheses", data)
        self.assertIn("matched_gaps", data)
        self.assertGreater(len(data["matched_gaps"]), 0)
        if len(data["hypotheses"]) > 0:
            first_h = data["hypotheses"][0]
            self.assertIn("hypothesis_id", first_h)
            self.assertIn("hypothesis", first_h)
            self.assertIn("target_population", first_h)
            self.assertIn("validation", first_h)

    def test_zero_hypothesis_generation_request(self):
        """Verify the 0-count hypothesis case: gap exploration and retrieval only."""
        payload = {
            "theme": "Telemedicine accessibility for remote and underserved rural communities",
            "population": "Rural elder cohort",
            "objective": "Identify structural barriers",
            "hypothesis_count": 0,
        }
        response = self.client.post("/api/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["hypothesis_count_requested"], 0)
        self.assertEqual(data["hypothesis_count_generated"], 0)
        self.assertEqual(data["status"], "no_hypotheses_requested")
        self.assertGreater(len(data["matched_gaps"]), 0)


if __name__ == "__main__":
    unittest.main()
