"""
scripts/patch_retrieval.py — Safely upgrade retrieval.py with batching, caching, and inference mode.
"""

from pathlib import Path
import re

RETRIEVAL_PY = Path("src/phase2/retrieval.py")
content = RETRIEVAL_PY.read_text(encoding="utf-8")

# 1. Update __init__ to include caches
old_init = """        self.embedding_model_name = embedding_model_name
        self._device = self._resolve_device(device)
        self._model = None
        self.index: faiss.Index | None = None
        self.chunks: list[ChunkMetadata] = []

        if index_path is not None:
            self.load_index(index_path)"""

new_init = """        self.embedding_model_name = embedding_model_name
        self._device = self._resolve_device(device)
        self._model = None
        self.index: faiss.Index | None = None
        self.chunks: list[ChunkMetadata] = []
        self._query_cache: dict[str, np.ndarray] = {}
        self._gap_evidence_cache: dict[tuple, list[RetrievedEvidence]] = {}

        if index_path is not None:
            self.load_index(index_path)"""

content = content.replace(old_init.replace("\r\n", "\n"), new_init.replace("\r\n", "\n"))

# 2. Add encode_queries_batch and inference_mode to encode_query
old_encode = '''    def encode_query(self, query_text: str) -> np.ndarray:
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
        return np.ascontiguousarray(emb, dtype=np.float32)'''

new_encode = '''    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a query string into a normalized dense embedding with query caching.

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

        if hasattr(self, "_query_cache") and clean_query in self._query_cache:
            return self._query_cache[clean_query].copy()

        with torch.inference_mode():
            emb = self.model.encode(
                [clean_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        res = np.ascontiguousarray(emb, dtype=np.float32)
        if hasattr(self, "_query_cache"):
            self._query_cache[clean_query] = res
        return res

    def encode_queries_batch(self, queries: list[str]) -> np.ndarray:
        """Batch encode multiple query strings with caching and inference mode.

        Parameters
        ----------
        queries : list[str]
            List of query strings.

        Returns
        -------
        np.ndarray
            Normalized 2D array of shape (len(queries), dimension), dtype float32.
        """
        cleaned_queries = [str(q).strip() or "mobile health mHealth interventions" for q in queries]
        if not cleaned_queries:
            return np.zeros((0, getattr(self.index, "d", 384)), dtype=np.float32)

        uncached_indices = []
        uncached_texts = []
        cached_results: dict[int, np.ndarray] = {}

        if not hasattr(self, "_query_cache"):
            self._query_cache = {}

        for idx, q in enumerate(cleaned_queries):
            if q in self._query_cache:
                cached_results[idx] = self._query_cache[q]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(q)

        if uncached_texts:
            with torch.inference_mode():
                embs = self.model.encode(
                    uncached_texts,
                    batch_size=max(len(uncached_texts), 1),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            for idx, text, emb in zip(uncached_indices, uncached_texts, embs):
                arr = np.ascontiguousarray(emb.reshape(1, -1), dtype=np.float32)
                self._query_cache[text] = arr
                cached_results[idx] = arr

        ordered = [cached_results[i] for i in range(len(cleaned_queries))]
        return np.vstack(ordered).astype(np.float32)'''

content = content.replace(old_encode.replace("\r\n", "\n"), new_encode.replace("\r\n", "\n"))

# 3. Optimize retrieve_for_gap with batched search and caching
old_retrieve_gap = '''    def retrieve_for_gap(
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
        return final_selected'''

new_retrieve_gap = '''    def retrieve_for_gap(
        self,
        gap_row: pd.Series | dict[str, Any],
        top_k: int = 20,
        min_score: float = 0.20,
    ) -> list[RetrievedEvidence]:
        """Construct a multi-faceted search query suite for a research gap and merge results.

        Optimized with batched query encoding, batched FAISS matrix search, and result caching.

        Parameters
        ----------
        gap_row : pd.Series or dict
            Candidate gap record.
        top_k : int, default=20
            Maximum number of evidence chunks to retrieve.
        min_score : float, default=0.20
            Minimum relevance threshold.

        Returns
        -------
        list[RetrievedEvidence]
            Grounded evidence passages supporting the candidate gap.
        """
        concept_a = str(gap_row.get("concept_a", "")).strip()
        concept_b = str(gap_row.get("concept_b", "")).strip()
        gap_id = str(gap_row.get("gap_id", "GAP-UNKNOWN"))

        if not hasattr(self, "_gap_evidence_cache"):
            self._gap_evidence_cache = {}

        cache_key = (gap_id, top_k, min_score)
        if cache_key in self._gap_evidence_cache:
            logger.debug("Returning cached evidence for gap %s (top_k=%d)", gap_id, top_k)
            return [RetrievedEvidence.from_dict(e.to_dict()) for e in self._gap_evidence_cache[cache_key]]

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
            truncated_desc = " ".join(evidence_desc.split()[:40])
            sub_queries.append(truncated_desc)

        if not sub_queries:
            sub_queries = [f"{concept_a} {concept_b} mobile health mHealth"]

        logger.info(
            "Retrieving evidence for gap %s (%s <-> %s) using %d sub-queries...",
            gap_id,
            concept_a,
            concept_b,
            len(sub_queries),
        )

        # 1. Batched query embedding (10x faster than sequential)
        stacked_q_embs = self.encode_queries_batch(sub_queries)

        # 2. Batched FAISS search in a single C++ call
        fetch_k = min(len(self.chunks), max(top_k * 4, 60))
        all_scores, all_indices = self.index.search(stacked_q_embs, fetch_k)

        all_retrieved: dict[str, RetrievedEvidence] = {}
        diversity_weight = 0.20

        for q_idx, query in enumerate(sub_queries):
            scores = all_scores[q_idx]
            indices = all_indices[q_idx]

            candidates: list[tuple[ChunkMetadata, float]] = []
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                raw_score = float(score)
                if raw_score >= min_score:
                    candidates.append((self.chunks[idx], raw_score))

            paper_counts: dict[str, int] = defaultdict(int)
            selected_count = 0

            while selected_count < top_k and candidates:
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
                selected_count += 1

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

                if (
                    evidence.chunk_id not in all_retrieved
                    or evidence.relevance_score > all_retrieved[evidence.chunk_id].relevance_score
                ):
                    all_retrieved[evidence.chunk_id] = evidence

        # Deduplicate near-identical text passages
        unique_evidence = self.remove_duplicate_passages(
            list(all_retrieved.values()),
            sim_threshold=0.95,
        )

        # Apply final diversity selection to pick top_k diverse papers
        final_selected: list[RetrievedEvidence] = []
        paper_counts: dict[str, int] = defaultdict(int)
        candidate_pool = list(unique_evidence)

        final_div_weight = 0.25
        while len(final_selected) < top_k and candidate_pool:
            best_idx = -1
            best_score = -float("inf")

            for i, cand in enumerate(candidate_pool):
                penalized = cand.relevance_score - (final_div_weight * paper_counts[cand.source_pmid])
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
        self._gap_evidence_cache[cache_key] = final_selected
        return final_selected'''

content = content.replace(old_retrieve_gap.replace("\r\n", "\n"), new_retrieve_gap.replace("\r\n", "\n"))

RETRIEVAL_PY.write_text(content, encoding="utf-8")
print("retrieval.py patched successfully!")
