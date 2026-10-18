"""
retrieval.py — FAISS-based semantic evidence retrieval for research gaps.

Retrieves grounded evidence passages from the abstract-only corpus to support
and constrain downstream hypothesis generation.

Design Note:
    We deliberately utilize 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions)
    for passage chunk retrieval rather than SPECTER2 (768 dimensions).
    While SPECTER2 is pretrained specifically for full-document citation-graph
    prediction (title + abstract pairs), all-MiniLM-L6-v2 is tuned for asymmetric
    semantic search (short natural-language queries against text passages).
    It offers 6x faster inference, compact memory usage, and robust cosine
    similarity calibration for ad-hoc topical retrieval.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch

from .chunking import ChunkMetadata

try:
    from ..config_loader import get_project_root, load_config, resolve_path
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config_loader import get_project_root, load_config, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class RetrievedEvidence:
    """A single retrieved evidence passage with provenance and relevance scores.

    Attributes
    ----------
    chunk_id : str
        Identifier of the retrieved chunk (CHK-{pmid}-{seq}).
    source_pmid : str
        PubMed identifier of the source publication.
    paper_title : str
        Title of the source publication.
    publication_year : int or None
        Publication year.
    text_content : str
        Retrieved passage text from the abstract.
    relevance_score : float
        Cosine similarity score between the query and this chunk.
    content_type : str, default='abstract'
        Type of source content. Always 'abstract' for this corpus.
    journal : str, default=''
        Publishing journal name.
    doi : str, default=''
        Digital Object Identifier.
    source_url : str, default=''
        Direct resolution link to the publication.
    query_used : str, default=''
        The search query that retrieved this evidence passage.
    """

    chunk_id: str
    source_pmid: str
    paper_title: str
    publication_year: int | None
    text_content: str
    relevance_score: float
    content_type: str = "abstract"
    journal: str = ""
    doi: str = ""
    source_url: str = ""
    query_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_pmid": str(self.source_pmid),
            "paper_title": self.paper_title,
            "publication_year": self.publication_year,
            "text_content": self.text_content,
            "relevance_score": round(float(self.relevance_score), 4),
            "content_type": self.content_type,
            "journal": self.journal,
            "doi": self.doi,
            "source_url": self.source_url,
            "query_used": self.query_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedEvidence":
        """Reconstruct from dictionary."""
        pub_year = data.get("publication_year")
        if pub_year is not None and not pd.isna(pub_year):
            try:
                pub_year = int(pub_year)
            except (ValueError, TypeError):
                pub_year = None
        else:
            pub_year = None

        return cls(
            chunk_id=str(data["chunk_id"]),
            source_pmid=str(data["source_pmid"]).strip(),
            paper_title=str(data.get("paper_title", "")),
            publication_year=pub_year,
            text_content=str(data.get("text_content", "")),
            relevance_score=float(data.get("relevance_score", 0.0)),
            content_type=str(data.get("content_type", "abstract")),
            journal=str(data.get("journal", "")),
            doi=str(data.get("doi", "")),
            source_url=str(data.get("source_url", "")),
            query_used=str(data.get("query_used", "")),
        )


class EvidenceRetriever:
    """FAISS-based dense semantic search engine for mHealth abstract chunks.

    Supports building, serializing, loading, stale-index verification,
    diversity-penalized retrieval, and multi-faceted query generation for
    candidate research gaps.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize the EvidenceRetriever.

        Parameters
        ----------
        embedding_model_name : str, default='sentence-transformers/all-MiniLM-L6-v2'
            HuggingFace / SentenceTransformers model name.
        index_path : str or Path, optional
            Path to existing FAISS index directory or file to load immediately.
        device : str, default='auto'
            Compute device: 'cpu', 'cuda', or 'auto' (detects GPU availability).
        """
        self.embedding_model_name = embedding_model_name
        self._device = self._resolve_device(device)
        self._model = None
        self.index: faiss.Index | None = None
        self.chunks: list[ChunkMetadata] = []

        if index_path is not None:
            self.load_index(index_path)

    @property
    def index_loaded(self) -> bool:
        """Return True if a valid FAISS index is initialized."""
        return self.index is not None and getattr(self.index, "ntotal", 0) > 0

    @staticmethod
    def _resolve_device(device_setting: str) -> str:
        """Resolve device string to 'cuda' or 'cpu'."""
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting

    @property
    def model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info(
                "Loading SentenceTransformer embedding model '%s' on %s...",
                self.embedding_model_name,
                self._device,
            )
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name, device=self._device)
        return self._model

    def build_index(
        self,
        chunks: list[ChunkMetadata],
        batch_size: int = 64,
    ) -> None:
        """Embed all chunk texts and construct a normalized FAISS IndexFlatIP.

        Inner product on unit-normalized vectors calculates exact cosine similarity.

        Parameters
        ----------
        chunks : list[ChunkMetadata]
            List of chunks to index.
        batch_size : int, default=64
            Inference batch size for embedding model.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty chunks list.")

        logger.info(
            "Building FAISS IndexFlatIP for %d chunks (batch_size=%d)...",
            len(chunks),
            batch_size,
        )

        texts = [c.text_content for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = list(chunks)

        logger.info(
            "FAISS index built successfully: %d vectors indexed with dimension %d.",
            self.index.ntotal,
            dim,
        )

    def _get_path_pair(self, base_path: str | Path) -> tuple[Path, Path]:
        """Resolve file paths for FAISS index and metadata files."""
        p = Path(base_path).resolve()
        if p.is_dir() or not p.suffix:
            p.mkdir(parents=True, exist_ok=True)
            index_file = p / "faiss_chunk_index.bin"
            meta_file = p / "faiss_chunk_metadata.json"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            index_file = p
            meta_file = p.with_name(f"{p.stem}_metadata.json")
        return index_file, meta_file

    def save_index(self, index_path: str | Path) -> None:
        """Save FAISS binary index and chunk metadata mapping to disk.

        Parameters
        ----------
        index_path : str or Path
            Target file or directory path.
        """
        if self.index is None or not self.chunks:
            raise ValueError("No active FAISS index to save. Call build_index() first.")

        index_file, meta_file = self._get_path_pair(index_path)

        logger.info("Saving FAISS index binary to %s", index_file)
        faiss.write_index(self.index, str(index_file))

        metadata_payload = {
            "embedding_model_name": self.embedding_model_name,
            "dimension": int(self.index.d),
            "total_chunks": len(self.chunks),
            "first_chunk_id": self.chunks[0].chunk_id if self.chunks else "",
            "last_chunk_id": self.chunks[-1].chunk_id if self.chunks else "",
            "chunks": [c.to_dict() for c in self.chunks],
        }

        logger.info("Saving chunk metadata (%d items) to %s", len(self.chunks), meta_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

        logger.info("FAISS index and metadata saved successfully.")

    def load_index(self, index_path: str | Path) -> None:
        """Load FAISS binary index and chunk metadata mapping from disk.

        Parameters
        ----------
        index_path : str or Path
            File or directory path containing saved index and metadata.
        """
        index_file, meta_file = self._get_path_pair(index_path)

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not meta_file.exists():
            raise FileNotFoundError(f"FAISS metadata file not found: {meta_file}")

        logger.info("Loading FAISS index binary from %s", index_file)
        self.index = faiss.read_index(str(index_file))

        logger.info("Loading chunk metadata from %s", meta_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata_payload = json.load(f)

        raw_chunks = metadata_payload.get("chunks", [])
        self.chunks = [ChunkMetadata.from_dict(c) for c in raw_chunks]

        if self.index.ntotal != len(self.chunks):
            logger.warning(
                "Mismatch between FAISS vector count (%d) and metadata chunk count (%d).",
                self.index.ntotal,
                len(self.chunks),
            )

        logger.info(
            "FAISS index loaded successfully: %d vectors (dimension: %d) with %d chunks.",
            self.index.ntotal,
            self.index.d,
            len(self.chunks),
        )

    def _is_index_stale(self, chunks: list[ChunkMetadata]) -> bool:
        """Check if the currently loaded index matches the given chunks.

        Compares counts and signatures of the first, middle, and last chunk IDs.

        Parameters
        ----------
        chunks : list[ChunkMetadata]
            Current list of corpus chunks.

        Returns
        -------
        bool
            True if stale or uninitialized, False if fully aligned.
        """
        if self.index is None or not self.chunks:
            return True

        if len(chunks) != len(self.chunks) or len(chunks) != self.index.ntotal:
            logger.info("Index stale: length mismatch (%d vs %d)", len(chunks), len(self.chunks))
            return True

        if not chunks:
            return False

        # Compare checkpoints
        checkpoints = [0, len(chunks) // 2, len(chunks) - 1]
        for idx in checkpoints:
            if chunks[idx].chunk_id != self.chunks[idx].chunk_id:
                logger.info("Index stale: chunk ID mismatch at index %d", idx)
                return True

        return False

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a query string into a normalized dense embedding.

        Parameters
        ----------
        query_text : str
            Query string.

        Returns
        -------
        np.ndarray
            Normalized 2D array of shape (1, dimension), dtype float32.
        """
        clean_query = str(query_text).strip()
        if not clean_query:
            clean_query = "mobile health mHealth interventions"

        emb = self.model.encode(
            [clean_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.2,
        diversity_weight: float = 0.3,
    ) -> list[RetrievedEvidence]:
        """Retrieve top-k relevant chunks with paper-level diversity penalization.

        Diversity Penalty Formulation:
        -----------------------------
        When multiple chunks belong to the same publication, a progressive penalty
        is subtracted from subsequent chunks of that paper:
          penalized_score = raw_score - (diversity_weight * paper_chunk_count)
        This ensures the retrieved evidence draws from a diverse spectrum of
        publications rather than over-indexing on multiple slices of a single paper.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int, default=20
            Maximum number of evidence chunks to return.
        min_score : float, default=0.2
            Minimum cosine similarity threshold for inclusion.
        diversity_weight : float, default=0.3
            Penalty multiplier applied for repeated chunks from the same paper.

        Returns
        -------
        list[RetrievedEvidence]
            Ranked list of diverse retrieved evidence passages.
        """
        if self.index is None or not self.chunks:
            raise ValueError("Retriever index is empty. Call build_index() or load_index().")

        q_emb = self.encode_query(query)

        # Retrieve a broader candidate pool to allow diversity selection
        fetch_k = min(len(self.chunks), max(top_k * 4, 60))
        scores, indices = self.index.search(q_emb, fetch_k)

        candidates: list[tuple[ChunkMetadata, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            raw_score = float(score)
            if raw_score >= min_score:
                candidates.append((self.chunks[idx], raw_score))

        selected: list[RetrievedEvidence] = []
        paper_counts: dict[str, int] = defaultdict(int)

        # Greedy selection loop with diversity penalization
        while len(selected) < top_k and candidates:
            best_idx = -1
            best_penalized_score = -float("inf")

            for i, (chunk, raw_score) in enumerate(candidates):
                penalty = diversity_weight * paper_counts[chunk.source_pmid]
                adj_score = raw_score - penalty
                if adj_score > best_penalized_score:
                    best_penalized_score = adj_score
                    best_idx = i

            if best_idx == -1:
                break

            best_chunk, best_raw_score = candidates.pop(best_idx)
            paper_counts[best_chunk.source_pmid] += 1

            evidence = RetrievedEvidence(
                chunk_id=best_chunk.chunk_id,
                source_pmid=best_chunk.source_pmid,
                paper_title=best_chunk.paper_title,
                publication_year=best_chunk.publication_year,
                text_content=best_chunk.text_content,
                relevance_score=round(best_raw_score, 4),
                content_type=best_chunk.content_type,
                journal=best_chunk.journal,
                doi=best_chunk.doi,
                source_url=best_chunk.source_url,
                query_used=query,
            )
            selected.append(evidence)

        logger.debug("Retrieved %d evidence passages for query: '%s'", len(selected), query)
        return selected

    def remove_duplicate_passages(
        self,
        results: list[RetrievedEvidence],
        sim_threshold: float = 0.95,
    ) -> list[RetrievedEvidence]:
        """Remove duplicate or near-identical passages from a list of retrieved evidence.

        Compares token overlap (Jaccard similarity) and exact textual identity.
        Higher-scoring evidence is preserved while near-duplicates are pruned.

        Parameters
        ----------
        results : list[RetrievedEvidence]
            List of candidate retrieved evidence passages.
        sim_threshold : float, default=0.95
            Jaccard token similarity threshold above which a passage is treated as duplicate.

        Returns
        -------
        list[RetrievedEvidence]
            Deduplicated list of evidence passages.
        """
        if not results:
            return []

        # Sort descending by relevance score
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        unique_results: list[RetrievedEvidence] = []
        seen_token_sets: list[set[str]] = []
        seen_chunk_ids: set[str] = set()

        for item in sorted_results:
            if item.chunk_id in seen_chunk_ids:
                continue

            clean_toks = set(item.text_content.lower().split())
            if not clean_toks:
                continue

            is_duplicate = False
            for prev_toks in seen_token_sets:
                intersection = len(clean_toks & prev_toks)
                union = len(clean_toks | prev_toks)
                jaccard = intersection / max(union, 1)

                if jaccard >= sim_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(item)
                seen_token_sets.append(clean_toks)
                seen_chunk_ids.add(item.chunk_id)

        return unique_results

    def retrieve_for_gap(
        self,
        gap_row: pd.Series | dict[str, Any],
        top_k: int = 20,
        min_score: float = 0.20,
    ) -> list[RetrievedEvidence]:
        """Construct a multi-faceted search query suite for a research gap and merge results.

        Constructs sub-queries exploring:
          1. Direct cross-concept synergy (concept_a AND concept_b in mHealth)
          2. Individual concept applications (concept_a digital health applications)
          3. Individual concept applications (concept_b digital health applications)
          4. Narrative evidence context (evidence_description or gap_note)

        Merges results across sub-queries, prunes near-duplicates, applies diversity
        weighting across publications, and returns top_k evidence passages.

        Parameters
        ----------
        gap_row : pd.Series or dict
            Candidate gap record.
        top_k : int, default=20
            Maximum number of evidence chunks to retrieve.

        Returns
        -------
        list[RetrievedEvidence]
            Grounded evidence passages supporting the candidate gap.
        """
        concept_a = str(gap_row.get("concept_a", "")).strip()
        concept_b = str(gap_row.get("concept_b", "")).strip()
        gap_id = str(gap_row.get("gap_id", "GAP-UNKNOWN"))

        # Formulate multifaceted queries
        sub_queries: list[str] = []
        if concept_a and concept_b:
            sub_queries.append(f"{concept_a} and {concept_b} in mobile health mHealth interventions")
            sub_queries.append(f"Integration of {concept_a} with {concept_b} in digital health")

        if concept_a:
            sub_queries.append(f"{concept_a} mobile health mHealth applications and outcomes")
        if concept_b:
            sub_queries.append(f"{concept_b} mobile health mHealth applications and outcomes")

        evidence_desc = str(gap_row.get("evidence_description", "")).strip()
        if evidence_desc and evidence_desc.lower() not in ("nan", "none"):
            # Truncate to avoid overly long queries
            truncated_desc = " ".join(evidence_desc.split()[:40])
            sub_queries.append(truncated_desc)

        logger.info(
            "Retrieving evidence for gap %s (%s <-> %s) using %d sub-queries...",
            gap_id,
            concept_a,
            concept_b,
            len(sub_queries),
        )

        # Retrieve across all sub-queries
        all_retrieved: dict[str, RetrievedEvidence] = {}

        for query in sub_queries:
            sub_results = self.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                diversity_weight=0.20,
            )
            for item in sub_results:
                if item.chunk_id not in all_retrieved:
                    all_retrieved[item.chunk_id] = item
                else:
                    # Keep version with higher relevance score
                    if item.relevance_score > all_retrieved[item.chunk_id].relevance_score:
                        all_retrieved[item.chunk_id] = item

        # Deduplicate near-identical text passages
        unique_evidence = self.remove_duplicate_passages(
            list(all_retrieved.values()),
            sim_threshold=0.95,
        )

        # Apply final diversity selection to pick top_k diverse papers
        final_selected: list[RetrievedEvidence] = []
        paper_counts: dict[str, int] = defaultdict(int)
        candidate_pool = list(unique_evidence)

        diversity_weight = 0.25
        while len(final_selected) < top_k and candidate_pool:
            best_idx = -1
            best_score = -float("inf")

            for i, cand in enumerate(candidate_pool):
                penalized = cand.relevance_score - (diversity_weight * paper_counts[cand.source_pmid])
                if penalized > best_score:
                    best_score = penalized
                    best_idx = i

            if best_idx == -1:
                break

            chosen = candidate_pool.pop(best_idx)
            paper_counts[chosen.source_pmid] += 1
            final_selected.append(chosen)

        logger.info(
            "Retrieved %d grounded evidence passages for gap %s across %d distinct papers.",
            len(final_selected),
            gap_id,
            len(paper_counts),
        )
        return final_selected


def save_retrieved_evidence(
    evidence_dict: dict[str, list[RetrievedEvidence]],
    output_path: str | Path,
) -> None:
    """Save retrieved evidence dictionary mapping gap_id to list of passages.

    Parameters
    ----------
    evidence_dict : dict[str, list[RetrievedEvidence]]
        Mapping of gap_id to list of RetrievedEvidence objects.
    output_path : str or Path
        Destination JSON filepath.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    serialized = {
        gap_id: [item.to_dict() for item in items]
        for gap_id, items in evidence_dict.items()
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    total_passages = sum(len(v) for v in evidence_dict.values())
    logger.info(
        "Saved retrieved evidence for %d gaps (%d total passages) to %s",
        len(evidence_dict),
        total_passages,
        out_file,
    )


def load_retrieved_evidence(
    input_path: str | Path,
) -> dict[str, list[RetrievedEvidence]]:
    """Load retrieved evidence dictionary from JSON file.

    Parameters
    ----------
    input_path : str or Path
        Source JSON filepath.

    Returns
    -------
    dict[str, list[RetrievedEvidence]]
        Mapping of gap_id to list of RetrievedEvidence objects.
    """
    in_file = Path(input_path).resolve()
    if not in_file.exists():
        raise FileNotFoundError(f"Retrieved evidence file not found: {in_file}")

    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    evidence_dict: dict[str, list[RetrievedEvidence]] = {}
    for gap_id, items in data.items():
        evidence_dict[str(gap_id)] = [RetrievedEvidence.from_dict(item) for item in items]

    total_passages = sum(len(v) for v in evidence_dict.values())
    logger.info(
        "Loaded retrieved evidence for %d gaps (%d total passages) from %s",
        len(evidence_dict),
        total_passages,
        in_file,
    )
    return evidence_dict
