# System Architecture & Technical Specifications

**Project Title:** Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research  
**Target Platform:** Python 3.11+ / FastAPI / React 19 / TypeScript / Vite / FAISS / Windows 11  
**Project Root:** `E:\Major_Project_Implementation`  

---

## 1. High-Level Architectural Overview

The system is organized into three decoupled phases following a strict **Gap-First, Literature-Grounded Architecture**:

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Empirical Literature Mining (Frozen Baseline)"]
        D1["Raw PubMed Corpus\n(14,905 papers)"] --> D2["Data Validation &\nRelevance Screening"]
        D2 --> D3["SPECTER2 Document\nEmbeddings (12,642 x 768)"]
        D3 --> D4["BERTopic Discovery\n(63 Topics + Outlier)"]
        D4 --> D5["Concept Extraction &\nMeSH Normalization"]
        D5 --> D6["Longitudinal Temporal\nVelocity Analysis"]
        D6 --> D7["Temporal-Semantic Gap\nScoring & Centroid Debiasing"]
        D7 --> D8["Candidate Research Gaps\n(Top 50 Gaps)"]
    end

    subgraph Phase2["Phase 2: Evidence-Grounded Generation & Validation"]
        D8 --> P1["Gap Preparation &\nMulti-Criteria Prioritization"]
        P1 --> P2["Corpus Chunking\n(12,776 Abstract Chunks)"]
        P2 --> P3["FAISS Dense Vector Index\n(all-MiniLM-L6-v2)"]
        P3 --> P4["Diversity-Penalized\nEvidence Retrieval"]
        P4 --> P5["Structured Hypothesis\nGeneration (Gemini / Mock)"]
        P5 --> P6["Evidence Verification\n& PMID Auditing"]
        P6 --> P7["NLI Semantic Alignment\n(DeBERTa-v3-small)"]
        P7 --> P8["LLM Judge Rubric &\nDeterministic Rule Checks"]
        P8 --> P9["Multi-Criteria Ranking &\nSemantic Deduplication"]
    end

    subgraph Phase3["Phase 3: Production Web Platform & Empirical Benchmarking"]
        P9 --> API["FastAPI Backend Engine\n(REST Endpoints & Async RAG)"]
        API --> UI["React 19 + TypeScript UI\n(Strict 2-Screen Academic Interface)"]
        P9 --> EV["Phase 3 Evaluation Suite\n(IR, Baselines, Ablations)"]
    end
```

---

## 2. Component Design & Module Contracts

### 2.1 Directory Structure
```text
E:\Major_Project_Implementation\
│
├── config.yaml                   # Central parameter registry (Phases 1, 2, 3)
├── requirements.txt             # Pinned package dependencies
├── .env                          # Local credentials (GEMINI_API_KEY)
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
├── legacy_streamlit_app.py       # Preserved Streamlit prototype
│
├── src/                          # Modular Python source packages
│   ├── config_loader.py          # Path resolver, YAML loader, random seeds
│   ├── data_validation.py        # Phase 1: schema checks and text validation
│   ├── relevance_screening.py    # Phase 1: rule-based mHealth scoring
│   ├── text_preprocessing.py     # Phase 1: scientific text cleaning
│   ├── embeddings.py             # Phase 1: SPECTER2 document embedding
│   ├── topic_modeling.py         # Phase 1: BERTopic cluster discovery
│   ├── concept_extraction.py     # Phase 1: MeSH keyword extraction
│   ├── mesh_normalization.py     # Phase 1: MeSH concept normalization
│   ├── temporal_analysis.py      # Phase 1: longitudinal trend slopes
│   ├── concept_quality_filter.py # Phase 1: centroid debiasing and filtering
│   ├── gap_detection.py          # Phase 1: gap scoring and candidate export
│   ├── visualization.py          # Phase 1: publication figures generator
│   │
│   ├── phase2/                   # Phase 2 Generation & Validation Package
│   │   ├── __init__.py           # Package exports
│   │   ├── gap_preparation.py    # Gap loading, schema validation, prioritization
│   │   ├── chunking.py           # Passage chunking with metadata preservation
│   │   ├── retrieval.py          # FAISS IndexFlatIP retriever with diversity
│   │   ├── hypothesis_generation.py # Structured Gemini generator & mock engine
│   │   ├── evidence_verification.py # Deterministic citation & PMID validation
│   │   ├── nli_validation.py     # DeBERTa-v3 cross-encoder NLI validation
│   │   ├── llm_judge.py          # 6-dimension LLM quality rubric evaluator
│   │   ├── rule_checks.py        # Deterministic falsifiability and language rules
│   │   ├── ranking.py            # Multi-criteria composite ranking
│   │   └── deduplication.py      # Dense semantic clustering & difference check
│   │
│   └── phase3/                   # Phase 3 Evaluation & Benchmarking Package
│       ├── __init__.py           # Package exports
│       ├── retrieval_eval.py     # Precision@K, Recall@K, MRR, nDCG@K
│       ├── grounding_eval.py     # Citation validity and traceability metrics
│       ├── baseline_comparison.py# Comparative experiments vs baselines
│       ├── ablation_study.py     # Component marginal utility benchmarks
│       └── run_evaluation.py     # Master Phase 3 experiment runner
│
├── tests/                        # 9 Modular Unit & Integration Test Suites
│   ├── test_phase1_artifacts.py
│   ├── test_gap_preparation.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_hypothesis_generation.py
│   ├── test_validation.py
│   ├── test_ranking_deduplication.py
│   ├── test_end_to_end_pipeline.py
│   ├── test_api_endpoints.py
│   └── test_react_frontend.py
│
├── outputs/                      # Intermediate and final verified artifacts
├── models/bertopic_model/        # Reusable trained BERTopic artifacts
├── figures/                      # Publication-grade figures
└── logs/                         # Execution logs with timestamps
```

---

## 3. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / User
    participant UI as React 19 Frontend (2 Screens)
    participant API as FastAPI Backend (/api/*)
    participant Pipe as PipelineService Orchestrator
    participant Ret as EvidenceRetriever (FAISS)
    participant LLM as Google Gemini (gemini-3.1-flash-lite)
    participant Ver as Citation Grounding Audit
    participant NLI as DeBERTa-v3 NLI Validator
    participant Judge as LLM Judge & 9 Rule Checks
    participant Rank as Ranking & Semantic Dedup

    User->>UI: Enter Research Theme & Target Population (Screen 1)
    UI->>API: GET /api/suggestions?query=... (Debounced)
    API-->>UI: Live Candidate Gaps & Ontology Terms
    User->>UI: Click "Generate Hypotheses" (0-3 count)
    UI->>API: POST /api/generate
    API->>Pipe: Execute Async Grounded Pipeline
    Pipe->>Ret: Retrieve Top Diverse Passages from FAISS (12,776 chunks)
    Ret-->>Pipe: Grounded Passages with PMIDs & Cosine Similarities
    Pipe->>LLM: Generate Hypotheses conditioned on Passages & Cohort
    LLM-->>Pipe: JSON Structured Hypotheses with Cited PMIDs
    Pipe->>Ver: Verify Cited PMIDs against Retrieved Chunks (100% trace)
    Pipe->>NLI: Run Entailment vs Premise Passages
    Pipe->>Judge: Run 6-Rubric Evaluation & 9 Deterministic Rule Checks
    Pipe->>Rank: Compute RankScore & Deduplicate Substantively
    Pipe-->>API: Persist Run Artifact (outputs/runs/{run_id}.json)
    API-->>UI: Return Verified Run Data & Hypotheses
    UI-->>User: Render Grounded Hypotheses on Screen 1 (No IV/DV)
    User->>UI: Click "View Complete Validation Workspace"
    UI-->>User: Render Screen 2 (5-Section Unified Workspace & Tabs)
```

---

## 4. Security, Credentials & Offline Compliance

1. **Environment Variables**: API keys (`GEMINI_API_KEY`) are accessed via `os.environ` or a local `.env` file that is strictly ignored by version control (`.gitignore`).
2. **Zero Cost During Testing**: The entire test suite and evaluation pipeline executes using offline mock engines (`use_mock=True`), eliminating accidental API spend.
3. **No Uncontrolled Execution**: No external code execution or dynamic `eval` is used. All JSON responses are strictly validated through explicit schemas with regex fallback parsers.
