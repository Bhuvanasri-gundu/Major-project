# Comprehensive Test Execution Report

**Project:** Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research  
**Execution Date:** 2026-10-18 11:16:12  
**Test Runner:** Python Standard `unittest` Suite  
**Total Execution Time:** 41.13 seconds  

---

## 1. Test Summary Dashboard

| Metric | Result |
| :--- | :--- |
| **Total Tests Executed** | **37** |
| **Tests Passed** | **37** |
| **Tests Failed** | **0** |
| **Errors** | **0** |
| **Skipped** | **0** |
| **Overall Success Rate** | **100.0%** |
| **Overall Verification Status** | **PASSED (100%)** |

---

## 2. Test Suite Breakdown

| Module | Test Focus | Target Coverage | Status |
| :--- | :--- | :--- | :--- |
| `test_phase1_artifacts.py` | Phase 1 baseline outputs, SPECTER2 (12642x768), BERTopic model reload | Corpus & Baseline Integrity | PASS |
| `test_gap_preparation.py` | Candidate gap loading, prioritization formula, schema bounds | Gap Processing | PASS |
| `test_chunking.py` | Abstract text chunking, metadata preservation, length limits | RAG Ingestion | PASS |
| `test_retrieval.py` | FAISS index, relevance ordering, duplicate pruning | Evidence Retrieval | PASS |
| `test_hypothesis_generation.py` | Schema parsing, regex fallback, missing API key handling, mock generation | Hypothesis Generation | PASS |
| `test_validation.py` | Cited reference checks, hallucination detection, causal flag, DeBERTa NLI | Validation & Grounding | PASS |
| `test_ranking_deduplication.py` | Multi-criteria ranking formula, tier assignment, substantive difference preservation | Ranking & Deduplication | PASS |
| `test_end_to_end_pipeline.py` | Complete pipeline output, traceability verification, CSV export | System Integration | PASS |
| `test_streamlit_app.py` | Dashboard syntax compilation, page data loaders | Interactive UI | PASS |

---

## 3. Detailed Test Log

```text
test_chunk_statistics (test_chunking.TestChunking.test_chunk_statistics) ... ok
test_create_chunks_preserves_metadata (test_chunking.TestChunking.test_create_chunks_preserves_metadata) ... ok
test_single_vs_multi_chunk (test_chunking.TestChunking.test_single_vs_multi_chunk) ... ok
test_evaluation_summary_exists (test_end_to_end_pipeline.TestEndToEndPipeline.test_evaluation_summary_exists) ... ok
test_evidence_traceability_integrity (test_end_to_end_pipeline.TestEndToEndPipeline.test_evidence_traceability_integrity) ... ok
test_final_hypotheses_csv_export (test_end_to_end_pipeline.TestEndToEndPipeline.test_final_hypotheses_csv_export) ... ok
test_pipeline_summary_exists_and_complete (test_end_to_end_pipeline.TestEndToEndPipeline.test_pipeline_summary_exists_and_complete) ... ok
test_load_phase1_artifacts (test_gap_preparation.TestGapPreparation.test_load_phase1_artifacts) ... ok
test_load_phase1_gaps (test_gap_preparation.TestGapPreparation.test_load_phase1_gaps) ... ok
test_prioritize_gaps (test_gap_preparation.TestGapPreparation.test_prioritize_gaps) ... ok
test_validate_gap_record_invalid_reflexive (test_gap_preparation.TestGapPreparation.test_validate_gap_record_invalid_reflexive) ... ok
test_validate_gap_record_valid (test_gap_preparation.TestGapPreparation.test_validate_gap_record_valid) ... ok
test_json_parsing_clean (test_hypothesis_generation.TestHypothesisGeneration.test_json_parsing_clean) ... ok
test_json_parsing_with_markdown_fences (test_hypothesis_generation.TestHypothesisGeneration.test_json_parsing_with_markdown_fences) ... ok
test_malformed_json_fallback (test_hypothesis_generation.TestHypothesisGeneration.test_malformed_json_fallback) ... ok
test_missing_api_key_raises_error_when_not_mock (test_hypothesis_generation.TestHypothesisGeneration.test_missing_api_key_raises_error_when_not_mock) ... ok
test_mock_generation_schema (test_hypothesis_generation.TestHypothesisGeneration.test_mock_generation_schema) ... ok
test_bertopic_model_directory (test_phase1_artifacts.TestPhase1Artifacts.test_bertopic_model_directory) ... ok
test_candidate_gaps_exist_and_schema (test_phase1_artifacts.TestPhase1Artifacts.test_candidate_gaps_exist_and_schema) ... ok
test_document_embeddings_shape (test_phase1_artifacts.TestPhase1Artifacts.test_document_embeddings_shape) ... ok
test_embedding_index_alignment (test_phase1_artifacts.TestPhase1Artifacts.test_embedding_index_alignment) ... ok
test_preprocessed_corpus_clean_document (test_phase1_artifacts.TestPhase1Artifacts.test_preprocessed_corpus_clean_document) ... ok
test_deduplication_preserves_substantive_differences (test_ranking_deduplication.TestRankingDeduplication.test_deduplication_preserves_substantive_differences) ... ok
test_ranking_formula_weights (test_ranking_deduplication.TestRankingDeduplication.test_ranking_formula_weights) ... ok
test_semantic_deduplication_merges_identical (test_ranking_deduplication.TestRankingDeduplication.test_semantic_deduplication_merges_identical) ... ok
test_duplicate_passage_removal (test_retrieval.TestRetrieval.test_duplicate_passage_removal) ... ok
test_index_is_loaded (test_retrieval.TestRetrieval.test_index_is_loaded) ... ok
test_retrieval_relevance (test_retrieval.TestRetrieval.test_retrieval_relevance) ... ok
test_streamlit_app_compiles (test_streamlit_app.TestStreamlitApp.test_streamlit_app_compiles) ... ok
test_streamlit_loads_phase1_data (test_streamlit_app.TestStreamlitApp.test_streamlit_loads_phase1_data) ... ok
test_streamlit_loads_phase2_data (test_streamlit_app.TestStreamlitApp.test_streamlit_loads_phase2_data) ... ok
test_evidence_reference_verification_fabricated (test_validation.TestValidation.test_evidence_reference_verification_fabricated) ... ok
test_evidence_reference_verification_valid (test_validation.TestValidation.test_evidence_reference_verification_valid) ... ok
test_llm_judge_mock_evaluation (test_validation.TestValidation.test_llm_judge_mock_evaluation) ... ok
test_nli_validator_availability (test_validation.TestValidation.test_nli_validator_availability) ... ok
test_rule_checks_flag_unsupported_causal (test_validation.TestValidation.test_rule_checks_flag_unsupported_causal) ... ok
test_rule_checks_pass (test_validation.TestValidation.test_rule_checks_pass) ... ok

----------------------------------------------------------------------
Ran 37 tests in 31.003s

OK
```

---

## 4. Key Verification Findings

1. **Zero Hallucinated PMIDs**: All cited passages in generated hypotheses trace directly to verified chunks in the retrieved candidate pool.
2. **Substantive Difference Preservation**: The semantic deduplicator successfully identified and retained hypotheses with distinct target populations or clinical endpoints, while correctly clustering true semantic duplicates.
3. **Graceful Degradation**: When external LLM API credentials are not provided, the pipeline cleanly informs the user and supports offline mock verification without runtime failure.
4. **Data Provenance**: Every generated hypothesis maintains an unbroken audit trail: `Hypothesis` $\rightarrow$ `Gap ID` $\rightarrow$ `Chunk IDs` $\rightarrow$ `PMID` $\rightarrow$ `PubMed Record`.
