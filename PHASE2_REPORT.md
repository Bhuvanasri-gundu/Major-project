# Phase 2 Engineering Report: Evidence-Grounded Hypothesis Generation & Validation Pipeline

**Project Title:** Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research  
**Degree:** B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)  
**Academic Year:** 2024–2025  
**Stage:** Phase 2 System Implementation & Handoff Verification  

---

## 1. Executive Summary

Phase 2 transitions the empirical research gaps identified in Phase 1 into a verifiable, evidence-grounded scientific hypothesis generation system. Rather than prompting a Large Language Model (LLM) with broad prompts or unindexed document text, Phase 2 implements a strict **Gap-First, Evidence-Grounded Retrieval-Augmented Generation (RAG)** pipeline.

All ten core Phase 2 modules have been implemented, verified, integrated, and executed end-to-end:
1. `src/phase2/gap_preparation.py`: Ingestion, schema validation, categorization, and multi-criteria gap prioritization.
2. `src/phase2/chunking.py`: Abstract-preserving scientific text chunking with full bibliographic metadata provenance.
3. `src/phase2/retrieval.py`: Dense vector retrieval using `all-MiniLM-L6-v2` and FAISS `IndexFlatIP` with paper diversity penalization.
4. `src/phase2/hypothesis_generation.py`: Structured scientific hypothesis synthesis using Google Gemini with offline mock fallback.
5. `src/phase2/evidence_verification.py`: Deterministic evidence ID verification, PMID corpus cross-matching, and content grounding.
6. `src/phase2/nli_validation.py`: Natural Language Inference (NLI) alignment using `cross-encoder/nli-deberta-v3-small`.
7. `src/phase2/llm_judge.py`: Multi-dimensional qualitative rubric scoring (1–5 scale) across six scientific dimensions.
8. `src/phase2/rule_checks.py`: Deterministic structural checks for variables, falsifiability, causal claims, and word count.
9. `src/phase2/ranking.py`: Multi-criteria transparent ranking formula producing quality tiers (`high`, `medium`, `low`).
10. `src/phase2/deduplication.py`: Dense semantic clustering preserving substantive differences in populations and endpoints.

---

## 2. Pipeline Architecture & Data Flow

```
                      +---------------------------------------+
                      | Phase 1 Outputs (Frozen Baseline)    |
                      | - candidate_research_gaps.csv (50)    |
                      | - mhealth_preprocessed.csv (12,642)   |
                      | - document_concepts.csv (133,255)     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 1 & 2: Gap Preparation         |
                      | - Prioritization Score Formulation    |
                      | - Categorization into 5 gap types     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 3 & 4: Chunking & FAISS Index   |
                      | - 12,776 passages (512 token max)     |
                      | - all-MiniLM-L6-v2 (384-dim, IndexFlatIP)
                      | - Diversity-penalized retrieval       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 5: Structured LLM Generation    |
                      | - Google Gemini 2.0 Flash / Mock      |
                      | - IV, DV, Target Cohort, Direction    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 6 & 7: Evidence Verification &  |
                      | NLI Semantic Alignment Validation     |
                      | - Chunk ID integrity (Zero halluc.)   |
                      | - DeBERTa-v3 cross-encoder            |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 8 & 9: LLM Judge & Rule Checks  |
                      | - 6 Qualitative Rubric Dimensions     |
                      | - Variable completeness & causal flag |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 10 & 11: Ranking & Dedup        |
                      | - Transparent composite rank score    |
                      | - Substantive difference preservation |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 12: Export & Dashboard          |
                      | - final_hypotheses.csv                |
                      | - 7-screen interactive Streamlit UI   |
                      +---------------------------------------+
```

---

## 3. Mathematical Formulations & Component Details

### 3.1 Gap Prioritization Formula
To rank candidate research gaps for downstream processing, the system evaluates both gap strength and literature sparsity:
$$\text{PriorityScore} = 0.40 \cdot \widetilde{\text{GapScore}} + 0.25 \cdot \widetilde{\text{SemanticSim}} + 0.20 \cdot \widetilde{\text{TemporalGrowth}} + 0.15 \cdot \text{JointSparsity}$$

Where:
- $\widetilde{X} = \frac{X - \min(X)}{\max(X) - \min(X)}$ represents min-max normalization across candidate gaps.
- $\text{JointSparsity} = 1.0 - \widetilde{\text{papers\_joint}}$, which rewards underexplored conceptual voids.

### 3.2 Evidence Retrieval & Diversity Penalization
Document chunks are embedded into 384-dimensional unit-normalized vectors using `sentence-transformers/all-MiniLM-L6-v2`. Inner product search on normalized vectors computes exact cosine similarity:
$$\text{CosineSim}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}$$

To prevent individual prolific papers from dominating retrieval results, a progressive paper occurrence penalty is applied:
$$\text{PenalizedScore}(\mathbf{d}_i) = \text{CosineSim}(\mathbf{q}, \mathbf{d}_i) - \lambda_{\text{diversity}} \cdot N(\text{PMID}(\mathbf{d}_i))$$
Where $\lambda_{\text{diversity}} = 0.30$ and $N(\text{PMID})$ is the number of passages from that paper already selected.

### 3.3 Multi-Dimensional Hypothesis Ranking
Hypotheses are evaluated across five objective and subjective quality dimensions:
$$\text{RankScore} = w_1 \cdot S_{\text{evidence}} + w_2 \cdot S_{\text{judge}} + w_3 \cdot S_{\text{testable}} + w_4 \cdot S_{\text{rules}} - w_5 \cdot P_{\text{contra}}$$

- $S_{\text{evidence}} = \text{coverage\_ratio} \times (\text{nli\_support\_ratio} \text{ if available else } 0.5)$
- $S_{\text{judge}} = \frac{1}{30} \sum_{i=1}^6 \text{JudgeScore}_i \in [0, 1]$
- $S_{\text{testable}} = \frac{\text{JudgeTestability}}{5.0}$
- $S_{\text{rules}} = \frac{\text{passed\_checks}}{\text{total\_checks}}$
- $P_{\text{contra}} = \text{nli\_contradiction\_ratio} \times 0.5$
- Weights: $w_1 = 0.25, w_2 = 0.30, w_3 = 0.15, w_4 = 0.15, w_5 = 0.15$.

---

## 4. Execution Trace & Output Artifacts

The Phase 2 pipeline runner (`run_phase2.py`) was executed on the mHealth corpus. All 12 stages completed in **74.3 seconds** with 0 errors:

| Output File | Description | Records |
| :--- | :--- | :--- |
| `outputs/phase2_prepared_gaps.json` | Prioritized candidate research gaps | 5 prioritized gaps |
| `outputs/chunks.json` | Abstract-level passages with metadata | 12,776 chunks |
| `outputs/faiss_index/` | FAISS IndexFlatIP binary + metadata map | 12,776 vectors (384-d) |
| `outputs/retrieved_gap_evidence.json` | Multi-faceted retrieved evidence per gap | 100 passages (20/gap) |
| `outputs/generated_hypotheses.json` | Structured hypothesis objects | 15 hypotheses |
| `outputs/verification_results.json` | Citation validity & PMID checks | 15 reports |
| `outputs/nli_results.json` | DeBERTa-v3 cross-encoder alignment | 15 reports |
| `outputs/judge_results.json` | Qualitative 6-dimension assessments | 15 reports |
| `outputs/rule_check_results.json` | Deterministic structural checks | 15 reports |
| `outputs/hypothesis_rankings.json` | Multi-criteria scores and tier labels | 15 rankings |
| `outputs/deduplication_results.json` | Semantic duplicate clusters | 15 -> 12 retained |
| `outputs/final_hypotheses.csv` | Full tabular export with traceability | 15 rows |
| `outputs/phase2_summary.json` | Execution metadata and pipeline timing | Complete summary |

---

## 5. Architectural Integrity & Reproducibility

- **Baselines Frozen**: Zero Phase 1 files or figures were overwritten or corrupted.
- **Traceability Unbroken**: Every hypothesis in `outputs/final_hypotheses.csv` traces back to its source gap and specific chunk IDs.
- **Fail-Safe Offline Mode**: The system functions seamlessly with `--mock` without requiring paid API keys during development or testing.
