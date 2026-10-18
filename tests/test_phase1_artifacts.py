"""
test_phase1_artifacts.py — Integrity checks for Phase 1 baseline artifacts.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase1Artifacts(unittest.TestCase):
    def setUp(self):
        self.outputs_dir = PROJECT_ROOT / "outputs"

    def test_candidate_gaps_exist_and_schema(self):
        p = self.outputs_dir / "candidate_research_gaps.csv"
        self.assertTrue(p.exists(), "candidate_research_gaps.csv missing")
        df = pd.read_csv(p)
        self.assertEqual(len(df), 50, "Should contain exactly 50 candidate gaps")
        required_cols = [
            "concept_a", "concept_b", "semantic_similarity",
            "cooccurrence_score", "temporal_growth_avg", "gap_score",
            "papers_concept_a", "papers_concept_b", "papers_joint",
            "supporting_pmids_a", "supporting_pmids_b",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    def test_document_embeddings_shape(self):
        p = self.outputs_dir / "document_embeddings.npy"
        self.assertTrue(p.exists(), "document_embeddings.npy missing")
        embeddings = np.load(p)
        self.assertEqual(embeddings.shape, (12642, 768), "Embeddings shape should be (12642, 768)")
        self.assertEqual(embeddings.dtype, np.float32, "Embeddings dtype should be float32")

    def test_embedding_index_alignment(self):
        p = self.outputs_dir / "document_embedding_index.csv"
        self.assertTrue(p.exists(), "document_embedding_index.csv missing")
        df = pd.read_csv(p)
        self.assertEqual(len(df), 12642, "Embedding index should contain 12642 PMIDs")
        self.assertIn("pmid", df.columns)
        self.assertIn("embedding_row", df.columns)

    def test_bertopic_model_directory(self):
        p = PROJECT_ROOT / "models" / "bertopic_model"
        self.assertTrue(p.exists(), "models/bertopic_model directory missing")
        self.assertTrue((p / "config.json").exists())
        self.assertTrue((p / "topics.json").exists())

    def test_preprocessed_corpus_clean_document(self):
        p = self.outputs_dir / "mhealth_preprocessed.csv"
        self.assertTrue(p.exists(), "mhealth_preprocessed.csv missing")
        # Check first 5 rows
        df = pd.read_csv(p, nrows=5)
        self.assertIn("clean_document", df.columns)
        self.assertIn("clean_abstract", df.columns)
        self.assertIn("pmid", df.columns)
        for _, row in df.iterrows():
            self.assertGreater(len(str(row["clean_document"])), 50)


if __name__ == "__main__":
    unittest.main()
