"""
test_chunking.py — Tests for document chunking with metadata preservation.
"""

import unittest
from pathlib import Path
import pandas as pd
from src.phase2.chunking import (
    create_chunks,
    get_chunk_statistics,
    get_chunks_for_paper,
    ChunkMetadata,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestChunking(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic DataFrame mimicking mhealth_preprocessed.csv
        self.sample_papers = pd.DataFrame([
            {
                "pmid": 10001,
                "title": "Short Mobile Health Study",
                "abstract": "This is a brief abstract under 512 tokens discussing mHealth.",
                "clean_document": "Short Mobile Health Study. This is a brief abstract under 512 tokens discussing mHealth.",
                "year": 2021,
                "journal": "J Med Internet Res",
                "doi": "10.2196/10001",
            },
            {
                "pmid": 10002,
                "title": "A Longer Scientific Paper on Telemedicine Interventions",
                "abstract": " ".join([f"Sentence number {i} discussing clinical telemetry." for i in range(120)]),
                "clean_document": "A Longer Scientific Paper on Telemedicine Interventions. " + " ".join([f"Sentence number {i} discussing clinical telemetry." for i in range(120)]),
                "year": 2023,
                "journal": "Telemed J E Health",
                "doi": "10.1089/tmj.2023.10002",
            },
        ])

    def test_create_chunks_preserves_metadata(self):
        chunks = create_chunks(self.sample_papers, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertTrue(chunk.chunk_id.startswith("CHK-"))
            self.assertIn(int(chunk.source_pmid), [10001, 10002])
            self.assertEqual(chunk.content_type, "abstract")
            self.assertTrue(chunk.source_url.startswith("https://"))
            self.assertGreater(len(chunk.text_content), 20)

    def test_single_vs_multi_chunk(self):
        chunks = create_chunks(self.sample_papers, chunk_size=300, chunk_overlap=50)
        chunks_p1 = get_chunks_for_paper(10001, chunks)
        chunks_p2 = get_chunks_for_paper(10002, chunks)
        # Short paper should be 1 chunk
        self.assertEqual(len(chunks_p1), 1)
        # Long paper should be split into multiple chunks
        self.assertGreater(len(chunks_p2), 1)

    def test_chunk_statistics(self):
        chunks = create_chunks(self.sample_papers, chunk_size=300, chunk_overlap=50)
        stats = get_chunk_statistics(chunks)
        self.assertIn("total_chunks", stats)
        self.assertIn("unique_papers", stats)
        self.assertEqual(stats["unique_papers"], 2)


if __name__ == "__main__":
    unittest.main()
