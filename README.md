# Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2BCUDA-red.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0-purple.svg)](https://vitejs.dev/)
[![Tests](https://img.shields.io/badge/Tests-37%2F37%20Passed-brightgreen.svg)]()

> **B.Tech Major Project** in Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
> An automated biomedical research discovery system that mines 12,642 PubMed articles (2015–2025), identifies empirical literature gaps via centroid-debiased semantic proximity and longitudinal growth velocity, and synthesizes evidence-grounded scientific hypotheses with 100% citation traceability.

---

## Table of Contents
1. [Core Scientific Novelty](#1-core-scientific-novelty)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Mathematical Formulations](#3-mathematical-formulations)
4. [Project Structure](#4-project-structure)
5. [Quickstart & Reproduction Guide](#5-quickstart--reproduction-guide)
6. [Interactive Research Platform (React 19 + FastAPI)](#6-interactive-research-platform-react-19--fastapi)
7. [Benchmark Evaluation Results](#7-benchmark-evaluation-results)
8. [Automated Testing & Quality Verification](#8-automated-testing--quality-verification)
9. [Academic Reports & Deliverables](#9-academic-reports--deliverables)

---

## 1. Core Scientific Novelty

Conventional automated scientific hypothesis generation systems prompt Large Language Models (LLMs) with broad topic names or unstructured text. This causes:
1. **Obviousness**: Generates frequent, obvious associations already saturated in literature.
2. **Citation Hallucination**: Generates plausible-sounding but completely fabricated paper citations (60–85% hallucination rate in scientific prompts).
3. **Historical Stagnation**: Recommends dormant or abandoned concept combinations due to lack of longitudinal awareness.

### Our Solution: The Gap-First Strategy
- **Empirical Gap Mining (Phase 1)**: Isolates concept pairs with high dense semantic affinity in SPECTER2 space but low co-occurrence in published literature ($J \le 0.06$), backed by positive 10-year publication slopes.
- **Passage-Level RAG (Phase 2)**: Chunks literature abstracts and indexes them in FAISS (`IndexFlatIP`). Retrieves top-$K$ diverse passages using multi-faceted queries and paper occurrence penalties.
- **Multi-Tier Hybrid Validation**: Combines deterministic citation matching against corpus PMIDs, DeBERTa-v3 cross-encoder Natural Language Inference (NLI), LLM-as-a-judge 6-dimension qualitative rubrics, and structural rule checks.
- **100% Citation Traceability**: Every hypothesis in the output maintains an unbroken provenance trail back to exact PubMed IDs.

---

## 2. End-to-End System Architecture

```text
[Raw PubMed Dataset: 14,905 papers (2015-2025)]
                        │
                        ▼
[Phase 1: Empirical Literature Mining & Gap Detection]
  ├── Validation & Cleaning: 12,642 screened mHealth records
  ├── Document Embeddings: SPECTER2 (12,642 × 768 float32)
  ├── Topic Discovery: Official BERTopic (63 topics + outlier)
  ├── Concept Normalization: 2,786 MeSH concepts
  ├── Centroid Debiasing: Restores geometric variance across concepts
  └── Gap Detection: Top 50 Candidate Research Gaps
                        │
                        ▼
[Phase 2: Evidence-Grounded Hypothesis Generation & Validation]
  ├── Gap Prioritization: Multi-criteria PriorityScore ranking
  ├── Document Chunking: 12,776 abstract-level passages with metadata
  ├── FAISS Dense Index: 384-d sentence-transformers/all-MiniLM-L6-v2
  ├── Diversity-Penalized Retrieval: Top-20 diverse evidence passages per gap
  ├── Structured Hypothesis Generation: Google Gemini 2.0 Flash / Mock engine
  ├── Evidence Verification: Strict chunk ID & PMID validation (0 hallucinations)
  ├── NLI Alignment: DeBERTa-v3 small cross-encoder entailment checks
  ├── Qualitative Rubric: LLM-as-a-Judge (relevance, clarity, testability, novelty)
  ├── Rule-Based Checks: Falsifiability, directional claims, variable checks
  ├── Ranking Engine: Multi-criteria composite rank score & quality tiers
  └── Semantic Deduplication: Clustering preserving substantive differences
                        │
                        ▼
[Phase 3: Interactive Dashboard & Comprehensive Benchmarks]
  ├── Streamlit Dashboard: 7 intuitive screens for researchers
  └── Benchmark Suite: P@K, MRR, nDCG@K, Baselines, and Ablations
```

---

## 3. Mathematical Formulations

### 3.1 Centroid-Debiased Semantic Similarity
$$\boldsymbol{\mu} = \frac{1}{|C|} \sum_{c \in C} \mathbf{v}_c$$
$$\text{SemanticSimilarity}(c_a, c_b) = \frac{(\mathbf{v}_a - \boldsymbol{\mu}) \cdot (\mathbf{v}_b - \boldsymbol{\mu})}{\|\mathbf{v}_a - \boldsymbol{\mu}\|_2 \|\mathbf{v}_b - \boldsymbol{\mu}\|_2}$$

### 3.2 Longitudinal Growth Velocity & GapScore
$$\text{GapScore}(c_a, c_b) = \text{SemanticSimilarity}(c_a, c_b) \times (1 - J(c_a, c_b)) \times (1 + \text{TemporalGrowthAvg}(c_a, c_b))$$

Where $J(c_a, c_b) = \frac{|D_a \cap D_b|}{|D_a \cup D_b|}$ is the document-level Jaccard co-occurrence, and $\text{TemporalGrowthAvg}$ is the average slope of publication counts from 2015 to 2025.

### 3.3 Retrieval Diversity Penalty
$$\text{PenalizedScore}(\mathbf{d}_i) = \text{CosineSim}(\mathbf{q}, \mathbf{d}_i) - 0.30 \cdot N(\text{PMID}(\mathbf{d}_i))$$

### 3.4 Multi-Criteria Hypothesis Ranking
$$\text{RankScore} = 0.25 \cdot S_{\text{evidence}} + 0.30 \cdot S_{\text{judge}} + 0.15 \cdot S_{\text{testable}} + 0.15 \cdot S_{\text{rules}} - 0.15 \cdot P_{\text{contra}}$$

---

## 4. Project Structure

```text
E:\Major_Project_Implementation\
├── config.yaml                   # Central parameter configuration
├── requirements.txt             # Pinned package requirements
├── .env.example                  # Template for GEMINI_API_KEY
├── run_phase1.py                 # Phase 1 master pipeline runner
├── run_phase2.py                 # Phase 2 master pipeline runner
├── run_tests.py                  # Master test suite runner (45 tests)
├── backend/                      # Production FastAPI application & RAG orchestration
│   ├── app.py                    # FastAPI entrypoint with CORS & routes
│   ├── schemas.py                # Pydantic v2 data models
│   ├── routes/api.py             # REST endpoints (health, suggestions, generate, runs)
│   └── services/                 # Theme matcher, pipeline service, evaluation service
├── frontend/                     # React 19 + TypeScript + Vite + Tailwind UI
│   ├── src/pages/                # Screen 1 (GeneratePage) & Screen 2 (WorkspacePage)
│   ├── src/components/           # Modular components (Suggestions, Cards, Validation Tabs)
│   └── src/services/api.ts       # Typed API client
├── legacy_streamlit_app.py       # Archived Streamlit prototype
│
├── src/
│   ├── phase2/                   # Phase 2 modules (RAG, Generation, Validation, Ranking)
│   │   ├── gap_preparation.py
│   │   ├── chunking.py
│   │   ├── retrieval.py
│   │   ├── hypothesis_generation.py
│   │   ├── evidence_verification.py
│   │   ├── nli_validation.py
│   │   ├── llm_judge.py
│   │   ├── rule_checks.py
│   │   ├── ranking.py
│   │   └── deduplication.py
│   └── phase3/                   # Phase 3 evaluation & benchmarking
│       ├── retrieval_eval.py
│       ├── grounding_eval.py
│       ├── baseline_comparison.py
│       ├── ablation_study.py
│       └── run_evaluation.py
│
├── tests/                        # 9 Modular test suites (100% pass)
├── outputs/                      # Verified output CSVs and JSONs
├── models/bertopic_model/        # Reusable trained BERTopic artifacts
├── figures/                      # Publication-grade figures
└── docs/                         # Detailed reports (Phase 2, Phase 3, Final, Test, Architecture)
```

---

## 5. Quickstart & Reproduction Guide

### Environment Setup
```powershell
# Clone or navigate to the repository
cd E:\Major_Project_Implementation

# Install dependencies
pip install -r requirements.txt

# (Optional) For live Gemini LLM calls, create a .env file:
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 1. Run Complete Phase 2 Pipeline (Offline Mock or Live)
```powershell
# Fast reproducible run using mock LLM (Zero API costs, ~1 minute):
python run_phase2.py --mock --max-gaps 5

# Full run with live Google Gemini API (requires GEMINI_API_KEY):
python run_phase2.py --max-gaps 10
```

### 2. Run Phase 3 Evaluation & Benchmarks
```powershell
python src/phase3/run_evaluation.py
```
Outputs will be written to `outputs/phase3_evaluation_results.json` and `outputs/phase3_evaluation_summary.csv`.

### 3. Execute Complete Automated Test Suite
```powershell
python run_tests.py
```
Executes all 37 unit and integration tests and writes `TEST_REPORT.md`.

---

## 6. Interactive Research Platform (React 19 + FastAPI)

The application provides a clean, professional academic research interface built with **React 19**, **TypeScript**, **Tailwind CSS v4**, and **FastAPI**, strictly organized into **TWO primary screens**.

### Running the Full Application:

`powershell
# Terminal 1: Launch FastAPI Backend (Port 8000)
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Launch React Frontend (Port 5173)
cd frontend
npm run dev
`

Visit http://localhost:5173 in your browser.

---

### Strict Two-Screen Architecture:

#### Screen 1 — Generate Hypotheses (/)
Combines user research input and generated hypothesis cards on a single unified screen:
- **Research Theme Input**: Required textarea accepting natural language research themes (min 3 chars, trimmed).
- **Live Debounced Theme Suggestions**: Real-time suggestions fetched from backend candidate gaps, ontology concepts, and BERTopic summaries with keyboard navigation.
- **Target Population & Objective**: Optional structured guidance inputs.
- **Hypothesis Count Selector**: Generates 0 to 3 hypotheses (0 explicitly supported for gap exploration and evidence retrieval only).
- **Hypothesis Output Cards**: Rendered directly below the input form upon generation. Displays hypothesis statement, target population, proposed mechanism, directionality, gap reference, evidence count, and validation status badge. *(Note: IV and DV are strictly omitted from cards per academic design standards).*
- **Actions**: Copy hypothesis, direct link to Complete Validation Workspace for that hypothesis, and main action to open the Complete Workspace.
- **About This Application**: Collapsible educational drawer explaining system methodology, FAISS retrieval, and LLM limitations.

#### Screen 2 — Complete Research & Validation Workspace (/workspace/:runId)
A unified academic dossier presenting all research and verification artifacts for a run:
- **Section 1: Generation Status**: Run ID, parameters, verified execution stage badges, timestamps, and warnings.
- **Section 2: Research Gap Explorer**: Matched candidate gaps, ontology concepts, joint papers, semantic distance, and alignment assessment.
- **Section 3: Hypothesis Results**: Selectable hypothesis cards allowing deep inspection of any generated hypothesis.
- **Section 4: Hypothesis Validation Workspace (5 Internal Tabs)**:
  1. *Validation Overview*: Aggregate checks passed, quality tier, limitations, and uncertainties.
  2. *Gap Alignment*: Explicit alignment score, rationale, and specific gap aspect addressed.
  3. *Evidence & Sources*: Retrieved FAISS passages with PMIDs, direct PubMed links, journal, year, relevance score, and full text.
  4. *Scientific Validation*: Deterministic citation grounding audit (100% valid), DeBERTa-v3 NLI entailment scores, 6-rubric LLM Judge evaluation, 9 rule-based checks, and Testability blueprint with study design and operationalization variables.
  5. *Ranking & Deduplication*: Multi-criteria composite rank score breakdown, priority score, grounding score, tier assignment, and deduplication status.
- **Section 5: Pipeline Evaluation (Project-Level)**: Collapsible benchmarks showing P@5, MRR, nDCG@10, citation validity rates, baseline comparisons, and ablation study.

## 7. Benchmark Evaluation Results

Evaluated on the empirical mHealth corpus against PubMed MeSH ground truth:

| Benchmark Category | Metric | Measured Value | Benchmark Significance |
| :--- | :--- | :--- | :--- |
| **Evidence Retrieval** | **Precision@5** | **0.4000** | Top 5 passages co-index target biomedical concepts |
| | **Precision@10** | **0.3800** | Top 10 passages relevant |
| | **MRR** | **0.7200** | First relevant passage returned at rank 1.39 on average |
| | **nDCG@10** | **0.2241** | Graded relevance accounting for joint concept co-indexing |
| **Evidence Grounding** | **Citation Validity Rate** | **100.0%** | Zero fabricated citations; all trace to candidate pool |
| | **Traceability Rate** | **100.0%** | Unbroken audit trail back to raw PubMed abstracts |
| | **Fabricated Citations**| **0** | Perfect hallucination resistance |
| **Baseline Contrast** | **RAG vs Direct Prompting** | **+100.0%** | Citation validity gain (+85.0% hallucination reduction) |
| **Ablation Study** | **Full Pipeline Utility**| **0.7961** | Baseline composite score |
| | **Without RAG Utility** | **0.0882** | Severe **88.9% degradation** without literature grounding |
| | **Without Temporal Utility**| **0.6804** | **14.5% degradation** (prioritizes obsolete research pairs) |

---

## 8. Automated Testing & Quality Verification

All 45 unit and integration tests execute cleanly with **100.0% success rate**:

```text
======================================================================
TEST RUN COMPLETED in 41.13 seconds
Total: 45 | Passed: 45 | Failures: 0 | Errors: 0 | Skipped: 0
Success Rate: 100.0%
======================================================================
```

Test coverage includes:
- Phase 1 baseline artifact integrity and SPECTER2 reload
- Candidate gap schema validation and prioritization formulas
- Text chunking and bibliographic metadata preservation
- FAISS dense vector retrieval and diversity penalization
- LLM JSON parsing, regex fallback, and missing API key error handling
- Deterministic reference verification and PMID cross-matching
- DeBERTa-v3 cross-encoder NLI validation
- Rule-based structural checks and unsupported causal claim flagging
- Multi-criteria composite ranking and tier assignment
- Semantic deduplication with substantive difference preservation
- Streamlit application compilation and data loaders

---

## 9. Academic Reports & Deliverables

The repository includes complete, formal academic deliverables ready for project submission and viva defense:
- **`FINAL_PROJECT_REPORT.md`**: Complete B.Tech Major Project Report with abstract, literature review, methodology, mathematical formulations, evaluation tables, and Viva Defense FAQ.
- **`PHASE2_REPORT.md`**: Comprehensive engineering report covering the 10 Phase 2 modules, data structures, and pipeline execution trace.
- **`PHASE3_EVALUATION_REPORT.md`**: Empirical benchmarking report detailing IR metrics, citation grounding, baseline comparisons, and ablation experiments.
- **`TEST_REPORT.md`**: Full test execution trace and verification findings across all 45 automated tests.
- **`ARCHITECTURE.md`**: Detailed technical specifications, component contracts, and Mermaid sequence/flowchart diagrams.
