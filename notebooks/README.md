# Academic Research Notebook Suite: mHealth Hypothesis Generation

This directory contains the **14 end-to-end research notebooks** documenting the complete scientific discovery pipeline, from initial raw PubMed extraction to multi-tier validation, ranking, and benchmark evaluations.

The notebooks are designed for **demonstration, inspection, faculty review, and academic presentation**. They directly import and execute canonical production logic from `src/` against verified artifacts in `outputs/`.

---

## Complete Notebook Index & Execution Order

| # | Notebook | Phase | Objective & Methodology | Inputs | Primary Outputs | Compute / Key Req. |
|---|---|---|---|---|---|---|
| **01** | [`01_pubmed_data_extraction_and_cleaning.ipynb`](01_pubmed_data_extraction_and_cleaning.ipynb) | Phase 1 | PubMed Entrez query extraction (2015–2025) and XML parsing | NCBI Entrez API | `data/pubmed_mhealth_*.xls` | Internet (NCBI) |
| **02** | [`02_data_validation.ipynb`](02_data_validation.ipynb) | Phase 1 | Schema validation, deduplication, missing value audits | Raw PubMed dataset | `outputs/data_validation_report.csv` | CPU (~10s) |
| **03** | [`03_relevance_screening.ipynb`](03_relevance_screening.ipynb) | Phase 1 | Rule-based biomedical relevance screening for mHealth corpus | Validated candidate data | `outputs/mhealth_screened_corpus.csv` | CPU (~20s) |
| **04** | [`04_preprocessing_embeddings.ipynb`](04_preprocessing_embeddings.ipynb) | Phase 1 | Scientific text cleaning & dense document embedding generation | Screened corpus | `outputs/document_embeddings.npy` (SPECTER2) | GPU / CPU (~3m) |
| **05** | [`05_bertopic_topic_discovery.ipynb`](05_bertopic_topic_discovery.ipynb) | Phase 1 | Unsupervised topic clustering (63 topics + outlier) | SPECTER2 embeddings | `outputs/topic_summary.csv`, `models/` | GPU / CPU (~5m) |
| **06** | [`06_concept_mesh_mapping.ipynb`](06_concept_mesh_mapping.ipynb) | Phase 1 | MeSH biomedical keyword extraction & ontology normalization | Preprocessed text | `outputs/mesh_normalized.csv`, `concept_summary.csv` | CPU (~1m) |
| **07** | [`07_temporal_semantic_gap_detection.ipynb`](07_temporal_semantic_gap_detection.ipynb) | Phase 1 | Centroid-debiased semantic similarity & 10-year longitudinal slopes | Normalized concepts | `outputs/candidate_research_gaps.csv` | CPU (~2m) |
| **08** | [`08_evidence_chunking_and_faiss_retrieval.ipynb`](08_evidence_chunking_and_faiss_retrieval.ipynb) | Phase 2 | Passage chunking (12,776 chunks) & FAISS vector retrieval | Screened corpus & Gaps | `outputs/faiss_index/`, `outputs/chunks.json` | CPU (~30s) |
| **09** | [`09_evidence_grounded_hypothesis_generation.ipynb`](09_evidence_grounded_hypothesis_generation.ipynb) | Phase 2 | Structured hypothesis synthesis conditioned on passage evidence | Retrieved passages & gaps | `outputs/generated_hypotheses.json` | `GEMINI_API_KEY` or Mock |
| **10** | [`10_citation_verification_and_traceability.ipynb`](10_citation_verification_and_traceability.ipynb) | Phase 2 | Deterministic PMID audit verifying 0 hallucinated citations | Generated hypotheses | `outputs/verification_results.json` | CPU (~5s) |
| **11** | [`11_deberta_nli_semantic_validation.ipynb`](11_deberta_nli_semantic_validation.ipynb) | Phase 2 | Cross-encoder Natural Language Inference (DeBERTa-v3) | Hypotheses & passages | `outputs/nli_results.json` | GPU / CUDA |
| **12** | [`12_llm_judge_and_rule_checks.ipynb`](12_llm_judge_and_rule_checks.ipynb) | Phase 2 | 6-rubric qualitative judge ratings & 9 structural rule checks | Generated hypotheses | `outputs/judge_results.json`, `rule_check_results.json` | CPU / Gemini |
| **13** | [`13_multi_criteria_ranking_and_deduplication.ipynb`](13_multi_criteria_ranking_and_deduplication.ipynb) | Phase 2 | Composite rank score formula & Sentence-Transformers clustering | Validation results | `outputs/hypothesis_rankings.json`, `final_hypotheses.csv` | CPU / GPU (~15s) |
| **14** | [`14_project_evaluation_and_ablations.ipynb`](14_project_evaluation_and_ablations.ipynb) | Phase 3 | P@5, MRR, nDCG@10, Baseline comparisons & Ablation study | Ground truth concepts | `outputs/phase3_evaluation_summary.csv` | CPU (~20s) |

---

## How to Run & Present the Notebooks

### 1. Environment Preparation
Ensure your Python virtual environment is activated and dependencies are installed:
```powershell
# From project root
pip install -r requirements.txt
```

### 2. Launch JupyterLab or Notebook Server
```powershell
# Launch from project root so paths resolve cleanly
jupyter lab
# or
jupyter notebook
```

### 3. Execution Principle & Offline Reproducibility
- **Single Source of Truth**: All notebooks import production functions from `src/` rather than re-implementing algorithms.
- **Fast Demonstration**: Expensive steps (SPECTER2 document embedding, FAISS indexing, BERTopic training) automatically load verified artifacts from `outputs/` by default. You can present any notebook instantly without hours of recomputation.
- **API Cost Safety**: Notebook 09 (`hypothesis_generation.py`) and Notebook 12 (`llm_judge.py`) use an optional `GEMINI_API_KEY`. When no key is present in `.env`, they operate cleanly in deterministic offline demonstration mode with zero API expenditure.
