"""
chunking.py — Document chunking with metadata preservation for evidence retrieval.

The research corpus consists strictly of PubMed publication ABSTRACTS (not full text).
Each document concatenates the title and abstract. This module partitions documents
into overlapping passages (chunks) when necessary, while preserving document provenance,
bibliographic citations, source URLs, topic assignments, and extracted concepts.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from ..config_loader import get_project_root, load_config, resolve_path
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config_loader import get_project_root, load_config, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata and text content for a single retrievable document chunk.

    Attributes
    ----------
    chunk_id : str
        Globally unique chunk identifier formatted as 'CHK-{source_pmid}-{chunk_index}'.
    source_pmid : str
        PubMed identifier of the source publication.
    paper_title : str
        Title of the source publication.
    publication_year : int or None
        Publication year of the paper.
    journal : str
        Journal or publishing venue name.
    doi : str
        Digital Object Identifier (DOI) if available.
    text_content : str
        Textual passage contained within this chunk.
    chunk_index : int
        0-indexed sequence position of the chunk within the parent document.
    total_chunks : int
        Total number of chunks produced from the parent document.
    source_url : str
        Direct URL link to the paper (DOI resolution link or PubMed link).
    content_type : str, default='abstract'
        Content type descriptor. Always 'abstract' for this abstract-only corpus.
    topic_id : int or None, optional
        BERTopic cluster ID assigned to the parent document in Phase 1.
    concepts : list[str], default_factory=list
        Normalized biomedical / mHealth concepts extracted from this document in Phase 1.
    """

    chunk_id: str
    source_pmid: str
    paper_title: str
    publication_year: int | None
    journal: str
    doi: str
    text_content: str
    chunk_index: int
    total_chunks: int
    source_url: str
    content_type: str = "abstract"
    topic_id: int | None = None
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk metadata to a serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_pmid": str(self.source_pmid),
            "paper_title": self.paper_title,
            "publication_year": self.publication_year,
            "journal": self.journal,
            "doi": self.doi,
            "text_content": self.text_content,
            "chunk_index": int(self.chunk_index),
            "total_chunks": int(self.total_chunks),
            "source_url": self.source_url,
            "content_type": self.content_type,
            "topic_id": int(self.topic_id) if self.topic_id is not None else None,
            "concepts": list(self.concepts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkMetadata":
        """Reconstruct ChunkMetadata from a dictionary."""
        pub_year = data.get("publication_year")
        if pub_year is not None and not pd.isna(pub_year):
            try:
                pub_year = int(pub_year)
            except (ValueError, TypeError):
                pub_year = None
        else:
            pub_year = None

        topic_id = data.get("topic_id")
        if topic_id is not None and not pd.isna(topic_id):
            try:
                topic_id = int(topic_id)
            except (ValueError, TypeError):
                topic_id = None
        else:
            topic_id = None

        return cls(
            chunk_id=str(data["chunk_id"]),
            source_pmid=str(data["source_pmid"]).strip(),
            paper_title=str(data.get("paper_title", "")),
            publication_year=pub_year,
            journal=str(data.get("journal", "")),
            doi=str(data.get("doi", "")),
            text_content=str(data.get("text_content", "")),
            chunk_index=int(data.get("chunk_index", 0)),
            total_chunks=int(data.get("total_chunks", 1)),
            source_url=str(data.get("source_url", "")),
            content_type=str(data.get("content_type", "abstract")),
            topic_id=topic_id,
            concepts=[str(c).strip() for c in data.get("concepts", []) if str(c).strip()],
        )


def construct_source_url(doi: Any, pmid: Any) -> str:
    """Construct canonical source URL preferring DOI and falling back to PubMed.

    Parameters
    ----------
    doi : Any
        DOI string or NaN.
    pmid : Any
        PubMed identifier or NaN.

    Returns
    -------
    str
        Canonical HTTP URL or empty string.
    """
    if doi is not None and pd.notna(doi):
        doi_str = str(doi).strip()
        if doi_str and doi_str.lower() not in ("nan", "none"):
            if doi_str.startswith("http://") or doi_str.startswith("https://"):
                return doi_str
            return f"https://doi.org/{doi_str}"

    if pmid is not None and pd.notna(pmid):
        pmid_str = str(pmid).strip()
        if pmid_str and pmid_str.lower() not in ("nan", "none"):
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/"

    return ""


def _split_text_to_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    min_chunk_length: int = 50,
) -> list[str]:
    """Split text into overlapping word chunks with min-length fragment preservation.

    Parameters
    ----------
    text : str
        Input document text.
    chunk_size : int, default=512
        Maximum number of words per chunk.
    chunk_overlap : int, default=128
        Number of overlapping words between consecutive chunks.
    min_chunk_length : int, default=50
        Minimum words required for an independent trailing chunk.
        If a trailing fragment has fewer words than this, it is merged into
        the preceding chunk to prevent fragment noise.

    Returns
    -------
    list[str]
        List of chunk text strings.
    """
    clean_text = str(text).strip()
    if not clean_text:
        return []

    words = clean_text.split()
    total_words = len(words)

    # Most abstracts fit within a single chunk
    if total_words <= chunk_size:
        return [clean_text]

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]

        # Merge trailing tiny fragment into previous chunk if too short
        if len(chunk_words) < min_chunk_length and len(chunks) > 0:
            chunks[-1] = chunks[-1] + " " + " ".join(chunk_words)
            break

        chunks.append(" ".join(chunk_words))
        if end >= total_words:
            break
        start += step

    return chunks


def create_chunks(
    papers_df: pd.DataFrame,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    min_chunk_length: int = 50,
    text_column: str = "clean_document",
    topic_assignments_df: pd.DataFrame | None = None,
    document_concepts_df: pd.DataFrame | None = None,
) -> list[ChunkMetadata]:
    """Chunk corpus documents into retrievable passages with complete provenance metadata.

    The corpus contains abstract-only records. Each paper's abstract (or title + abstract)
    is split into passages of `chunk_size` words with `chunk_overlap`. Each chunk is tagged
    with its original PMID, bibliographic info, canonical URL, and optional Phase 1 topic
    and concept mappings.

    Parameters
    ----------
    papers_df : pd.DataFrame
        Corpus DataFrame containing 'pmid', 'clean_document' (or 'document'), 'title',
        'year', 'journal', 'doi'.
    chunk_size : int, default=512
        Maximum words per chunk.
    chunk_overlap : int, default=128
        Number of overlapping words between consecutive chunks.
    min_chunk_length : int, default=50
        Minimum words for a trailing chunk before merging with previous.
    text_column : str, default='clean_document'
        Primary text column to chunk. Falls back to 'document' or 'title + abstract'.
    topic_assignments_df : pd.DataFrame, optional
        Phase 1 topic assignments (columns: pmid, topic_id).
    document_concepts_df : pd.DataFrame, optional
        Phase 1 document concepts (columns: pmid, concept or normalized_concept).

    Returns
    -------
    list[ChunkMetadata]
        List of structured ChunkMetadata objects ready for indexing and retrieval.
    """
    logger.info(
        "Creating chunks: chunk_size=%d, chunk_overlap=%d, min_chunk_length=%d, text_column=%s",
        chunk_size,
        chunk_overlap,
        min_chunk_length,
        text_column,
    )

    # Build topic lookup if provided
    topic_map: dict[str, int] = {}
    if topic_assignments_df is not None and not topic_assignments_df.empty:
        if "pmid" in topic_assignments_df.columns and "topic_id" in topic_assignments_df.columns:
            for _, r in topic_assignments_df.iterrows():
                try:
                    topic_map[str(r["pmid"]).strip()] = int(r["topic_id"])
                except (ValueError, TypeError):
                    continue

    # Build concepts lookup if provided
    concepts_map: dict[str, list[str]] = defaultdict(list)
    if document_concepts_df is not None and not document_concepts_df.empty:
        concept_col = (
            "normalized_concept"
            if "normalized_concept" in document_concepts_df.columns
            else ("concept" if "concept" in document_concepts_df.columns else None)
        )
        if concept_col and "pmid" in document_concepts_df.columns:
            for _, r in document_concepts_df.iterrows():
                c_val = str(r[concept_col]).strip()
                if c_val and c_val.lower() not in ("nan", "none"):
                    concepts_map[str(r["pmid"]).strip()].append(c_val)

    all_chunks: list[ChunkMetadata] = []
    skipped_empty = 0

    for _, row in papers_df.iterrows():
        pmid = str(row.get("pmid", "")).strip()
        if not pmid or pmid.lower() in ("nan", "none"):
            continue

        # Extract text content
        raw_text = row.get(text_column)
        if pd.isna(raw_text) or not str(raw_text).strip():
            raw_text = row.get("document")
        if pd.isna(raw_text) or not str(raw_text).strip():
            title_part = str(row.get("title", "")).strip()
            abstract_part = str(row.get("abstract", "")).strip()
            raw_text = f"{title_part} {abstract_part}".strip()

        if not str(raw_text).strip():
            skipped_empty += 1
            continue

        title = str(row.get("title", "")).strip()
        journal = str(row.get("journal", "")).strip()
        doi = str(row.get("doi", "")).strip() if pd.notna(row.get("doi")) else ""

        year_val = row.get("year")
        try:
            pub_year = int(year_val) if pd.notna(year_val) else None
        except (ValueError, TypeError):
            pub_year = None

        url = construct_source_url(doi, pmid)

        # Lookup topic and concepts
        topic_id = topic_map.get(pmid)
        if topic_id is None and "topic_id" in row and pd.notna(row["topic_id"]):
            try:
                topic_id = int(row["topic_id"])
            except (ValueError, TypeError):
                topic_id = None

        paper_concepts = concepts_map.get(pmid, [])
        if not paper_concepts and "concepts" in row and isinstance(row["concepts"], (list, tuple)):
            paper_concepts = [str(c) for c in row["concepts"]]

        # Chunk the text
        text_chunks = _split_text_to_chunks(
            str(raw_text),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_length=min_chunk_length,
        )

        total_chunks = len(text_chunks)
        for seq, chunk_text in enumerate(text_chunks):
            chunk_id = f"CHK-{pmid}-{seq}"
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_pmid=pmid,
                paper_title=title,
                publication_year=pub_year,
                journal=journal,
                doi=doi,
                text_content=chunk_text,
                chunk_index=seq,
                total_chunks=total_chunks,
                source_url=url,
                content_type="abstract",
                topic_id=topic_id,
                concepts=list(paper_concepts),
            )
            all_chunks.append(chunk_meta)

    if skipped_empty > 0:
        logger.warning("Skipped %d papers due to empty document text.", skipped_empty)

    logger.info(
        "Successfully produced %d total chunks across %d unique papers.",
        len(all_chunks),
        len(set(c.source_pmid for c in all_chunks)),
    )
    return all_chunks


def get_chunks_for_paper(pmid: str | int, chunks_list: list[ChunkMetadata]) -> list[ChunkMetadata]:
    """Retrieve all chunks belonging to a specific source paper.

    Parameters
    ----------
    pmid : str or int
        Target PubMed identifier.
    chunks_list : list[ChunkMetadata]
        Corpus chunk collection.

    Returns
    -------
    list[ChunkMetadata]
        List of matching ChunkMetadata objects, ordered by chunk_index.
    """
    target = str(pmid).strip()
    matched = [c for c in chunks_list if str(c.source_pmid).strip() == target]
    matched.sort(key=lambda x: x.chunk_index)
    return matched


def save_chunks(chunks: list[ChunkMetadata], output_path: str | Path) -> None:
    """Save chunk collection to JSON file preserving all metadata fields.

    Parameters
    ----------
    chunks : list[ChunkMetadata]
        Chunks to serialize.
    output_path : str or Path
        Destination JSON filepath.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = [c.to_dict() for c in chunks]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunks to %s", len(chunks), out_file)


def load_chunks(input_path: str | Path) -> list[ChunkMetadata]:
    """Load chunk collection from JSON file into ChunkMetadata objects.

    Parameters
    ----------
    input_path : str or Path
        Path to JSON chunks file.

    Returns
    -------
    list[ChunkMetadata]
        Reconstructed chunk list.
    """
    in_file = Path(input_path).resolve()
    if not in_file.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {in_file}")

    with open(in_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    chunks = [ChunkMetadata.from_dict(item) for item in raw_list]
    logger.info("Loaded %d chunks from %s", len(chunks), in_file)
    return chunks


def get_chunk_statistics(chunks: list[ChunkMetadata]) -> dict[str, Any]:
    """Compute comprehensive diagnostic statistics on the chunk collection.

    Parameters
    ----------
    chunks : list[ChunkMetadata]
        List of ChunkMetadata objects.

    Returns
    -------
    dict[str, Any]
        Statistical summary of chunks.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "unique_papers": 0,
            "avg_words_per_chunk": 0.0,
            "avg_chars_per_chunk": 0.0,
            "min_words_per_chunk": 0,
            "max_words_per_chunk": 0,
            "single_chunk_papers": 0,
            "multi_chunk_papers": 0,
            "chunks_per_paper_avg": 0.0,
            "chunks_per_paper_max": 0,
            "content_type_distribution": {},
        }

    word_counts = [len(c.text_content.split()) for c in chunks]
    char_counts = [len(c.text_content) for c in chunks]

    paper_chunk_counts: dict[str, int] = Counter(c.source_pmid for c in chunks)
    content_types = Counter(c.content_type for c in chunks)

    single_chunk_papers = sum(1 for cnt in paper_chunk_counts.values() if cnt == 1)
    multi_chunk_papers = sum(1 for cnt in paper_chunk_counts.values() if cnt > 1)

    stats = {
        "total_chunks": len(chunks),
        "unique_papers": len(paper_chunk_counts),
        "avg_words_per_chunk": round(float(np.mean(word_counts)), 2),
        "avg_chars_per_chunk": round(float(np.mean(char_counts)), 2),
        "min_words_per_chunk": int(np.min(word_counts)),
        "max_words_per_chunk": int(np.max(word_counts)),
        "single_chunk_papers": single_chunk_papers,
        "multi_chunk_papers": multi_chunk_papers,
        "chunks_per_paper_avg": round(float(len(chunks) / max(len(paper_chunk_counts), 1)), 2),
        "chunks_per_paper_max": int(max(paper_chunk_counts.values())),
        "content_type_distribution": dict(content_types),
    }

    logger.info(
        "Chunk statistics: %d chunks across %d papers (avg words: %.1f, single-chunk: %d, multi-chunk: %d)",
        stats["total_chunks"],
        stats["unique_papers"],
        stats["avg_words_per_chunk"],
        stats["single_chunk_papers"],
        stats["multi_chunk_papers"],
    )
    return stats
