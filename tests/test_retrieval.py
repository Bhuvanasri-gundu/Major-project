"""
test_retrieval.py — Tests for FAISS dense retrieval, diversity penalization, and duplicate pruning.
"""

import unittest
from pathlib import Path
from src.phase2.chunking import ChunkMetadata
from src.phase2.retrieval import EvidenceRetriever, RetrievedEvidence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build small test chunks for indexing
        cls.test_chunks = [
            ChunkMetadata(
                chunk_id="CHK-1-0",
                source_pmid=1,
                paper_title="mHealth Telemedicine in Rural Communities",
                publication_year=2020,
                journal="JMIR",
                doi="10.1/jmir.1",
                text_content="Rural health services benefit greatly from telemedicine and mobile phone health applications.",
                chunk_index=0,
                total_chunks=1,
                source_url="https://doi.org/10.1/jmir.1",
            ),
            ChunkMetadata(
                chunk_id="CHK-2-0",
                source_pmid=2,
                paper_title="Mobile Health Units for Underserved Populations",
                publication_year=2021,
                journal="Lancet Digit Health",
                doi="10.2/lancet.2",
                text_content="Mobile health units provide critical outreach for rural healthcare clinics and remote patients.",
                chunk_index=0,
                total_chunks=1,
                source_url="https://doi.org/10.2/lancet.2",
            ),
            ChunkMetadata(
                chunk_id="CHK-3-0",
                source_pmid=3,
                paper_title="Artificial Intelligence in Oncology Diagnosis",
                publication_year=2022,
                journal="Cancer Res",
                doi="10.3/cancer.3",
                text_content="Deep learning algorithms detect malignant melanoma from high-resolution clinical images.",
                chunk_index=0,
                total_chunks=1,
                source_url="https://doi.org/10.3/cancer.3",
            ),
        ]
        cls.retriever = EvidenceRetriever()
        cls.retriever.build_index(cls.test_chunks)

    def test_index_is_loaded(self):
        self.assertTrue(self.retriever.index_loaded)
        self.assertEqual(len(self.retriever.chunks), 3)

    def test_retrieval_relevance(self):
        query = "telemedicine rural health clinic outreach"
        results = self.retriever.retrieve(query, top_k=2, min_score=0.1)
        self.assertGreater(len(results), 0)
        # First hit should be related to rural / telemedicine
        top_pmid = results[0].source_pmid
        self.assertIn(top_pmid, [1, 2])

    def test_duplicate_passage_removal(self):
        # Two identical passages with different scores
        r1 = RetrievedEvidence(
            chunk_id="CHK-1-0",
            source_pmid=1,
            paper_title="T1",
            publication_year=2020,
            text_content="Exact identical text passage describing digital health.",
            relevance_score=0.90,
            content_type="abstract",
            journal="J1",
            doi="",
            source_url="",
            query_used="q",
        )
        r2 = RetrievedEvidence(
            chunk_id="CHK-1-1",
            source_pmid=1,
            paper_title="T1",
            publication_year=2020,
            text_content="Exact identical text passage describing digital health.",
            relevance_score=0.75,
            content_type="abstract",
            journal="J1",
            doi="",
            source_url="",
            query_used="q",
        )
        pruned = self.retriever.remove_duplicate_passages([r1, r2], sim_threshold=0.90)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].relevance_score, 0.90)


if __name__ == "__main__":
    unittest.main()
