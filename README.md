# Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research

## B.Tech Major Project — Phase 1 (Completed & Verified)

---

## 1. Project Overview

This project implements an intelligent mHealth research discovery system that systematically analyzes 14,905 PubMed research articles (2015–2025) to discover empirical research topics, extract biomedical concepts, model longitudinal temporal trends, evaluate semantic affinity, compute literature co-occurrence, and detect **candidate research gaps** with literature-grounded evidence.

### Core Scientific Novelty: Gap-First Hypothesis Generation

Conventional automated scientific hypothesis generation systems prompt a large language model (LLM) directly with broad topic names or unstructured abstracts. This frequently causes hallucinated associations, unfocused suggestions, and lack of empirical literature grounding.

In contrast, our proposed methodology follows a strict **Gap-First Strategy**:
1. **Mathematical Grounding**: It isolates pairs of biomedical concepts that share high dense semantic affinity in scientific representation space (SPECTER2) but exhibit disproportionately low joint occurrence in published literature (Jaccard co-occurrence $\le 0.06$).
2. **Longitudinal Validation**: It verifies that both concepts are experiencing increasing research velocity (positive temporal growth slope over 2015–2025), proving that the concepts are active research frontiers rather than obsolete terms.
3. **Structured Handoff**: By computing exact empirical gap scores, isolating topic clusters, and indexing supporting PMIDs *before* invoking any generative model, the system ensures that Phase 2 will guide the LLM using concrete literature gaps rather than open-ended prompting.

> [!IMPORTANT]
> A candidate gap represents a *potentially underexplored relationship*, not proof that no research exists. The pipeline explicitly outputs **candidate research gaps** and transparently flags potential lexical/abbreviation synonym pairs.

---

## 2. Mathematical Formulations

### Gap Score Formula

$$\text{GapScore} = \text{SemanticSimilarity} \times (1 - \text{NormalizedCoOccurrence}) \times (1 + \text{TemporalGrowthAvg})$$

Where:
- $\text{SemanticSimilarity}$: Centroid-debiased cosine similarity between dense concept vectors:
  $$\text{SemanticSimilarity}(c_a, c_b) = \frac{(\mathbf{v}_a - \boldsymbol{\mu}) \cdot (\mathbf{v}_b - \boldsymbol{\mu})}{\|\mathbf{v}_a - \boldsymbol{\mu}\| \|\mathbf{v}_b - \boldsymbol{\mu}\|}$$
- $\text{NormalizedCoOccurrence}$: Document-level Jaccard similarity across corpus:
  $$J(c_a, c_b) = \frac{|D_a \cap D_b|}{|D_a \cup D_b|}$$
- $\text{TemporalGrowthAvg}$: Average annual publication growth velocity across 2015–2025:
  $$\text{TemporalGrowthAvg} = \frac{\text{slope}(c_a) + \text{slope}(c_b)}{2}$$

---

## 3. Implemented Phase 1 Pipeline

```
Raw PubMed Dataset (14,905 records, 2015-2025)
        ↓
Data Validation & Cleaning (14,881 validated records)
        ↓
Rule-Based mHealth Relevance Screening (12,642 screened: 9,279 relevant + 3,363 borderline)
        ↓
Scientific Text Preprocessing (Clean document field)
        ↓
SPECTER2 Document Embeddings (12,642 × 768 dense vectors, allenai/specter2_base)
        ↓
Official BERTopic Topic Modeling (bertopic 0.17.4, 63 genuine topics + outlier class, saved to models/bertopic_model/)
        ↓
Biomedical Concept Extraction (2,790 concepts with ≥5 mentions)
        ↓
MeSH Normalization & Topic-Concept Mapping (2,786 normalized concepts)
        ↓
Temporal Trend Analysis (2015-2025 annual matrix, slope calculations)
        ↓
Concept Quality Filtering (737 domain concepts kept; generic/demographic terms removed)
        ↓
Centroid Debiasing (Debiased mean similarity = 0.0255, spread restored)
        ↓
Concept Co-occurrence Analysis (17,196 co-occurring pairs, top 600 concepts)
        ↓
Candidate Research Gap Detection & Scoring (Top 50 gaps scored)
        ↓
Synonym Deduplication (32 genuine substantive gaps + 18 flagged synonym variants)
        ↓
Supporting Evidence Compilation & PMID Traceability (50 evidence passages)
        ↓
Publication Figure Generation (9 figures)
```

---

## 4. Project Organization

```
E:\Major_Project_Implementation\
│
├── data/
│   └── pubmed_mhealth_2015_2025_proposed_final.xls   # Raw dataset (14,905 PubMed records, CSV utf-8-sig)
│
├── notebooks/                                # Interactive exploratory notebooks (all executed)
│   ├── 00_pubmed_data_extraction.ipynb       # Dataset extraction notebook (Entrez API, Biopython)
│   ├── 01_data_validation.ipynb              # Data validation, checks & cleaning
│   ├── 02_relevance_screening.ipynb          # Rule-based mHealth relevance screening
│   ├── 03_preprocessing_embeddings.ipynb     # Scientific text preprocessing & SPECTER2 embeddings
│   ├── 04_bertopic_topic_discovery.ipynb     # Official BERTopic topic discovery & model reload
│   ├── 05_concept_mesh_mapping.ipynb         # Biomedical concept extraction & MeSH normalization
│   └── 06_temporal_semantic_gap_detection.ipynb # Temporal trends, debiasing, co-occurrence, gap scoring
│
├── src/                                      # Modular source code (13 modules)
│   ├── __init__.py                           # Source package initializer
│   ├── config_loader.py                      # Path resolver, YAML loader, random seeds
│   ├── data_validation.py                    # Dataset validation, checks, cleaning
│   ├── relevance_screening.py                # Rule-based mHealth scoring (no LLM)
│   ├── text_preprocessing.py                 # Scientific text cleaning & normalization
│   ├── embeddings.py                         # SPECTER2 dense document embeddings
│   ├── topic_modeling.py                     # Official BERTopic topic discovery & model saving
│   ├── concept_extraction.py                 # MeSH & keyword concept extraction
│   ├── mesh_normalization.py                 # MeSH term deduplication & topic mapping
│   ├── concept_quality_filter.py             # Concept filtering, centroid debiasing, synonym dedup
│   ├── temporal_analysis.py                  # Longitudinal publication trend slopes
│   ├── gap_detection.py                      # Co-occurrence, gap scoring, evidence compilation
│   └── visualization.py                      # Publication figure generator
│
├── models/
│   └── bertopic_model/                       # Reusable trained official BERTopic model
│       ├── config.json
│       ├── ctfidf.safetensors
│       ├── ctfidf_config.json
│       ├── topics.json
│       └── topic_embeddings.safetensors
│
├── outputs/                                  # 20 verified pipeline output artifacts
│   ├── mhealth_candidate_cleaned.csv         # 14,881 records
│   ├── mhealth_screened_corpus.csv           # 12,642 records
│   ├── mhealth_preprocessed.csv              # 12,642 records
│   ├── document_embeddings.npy               # 12,642 × 768 matrix
│   ├── document_embedding_index.csv          # PMID alignment index
│   ├── topic_assignments.csv                 # 12,642 topic labels
│   ├── topic_summary.csv                     # 64 topic rows (63 topics + outlier)
│   ├── topic_top_terms.csv                   # c-TF-IDF top terms
│   ├── topic_year_matrix.csv                 # 11 × 63 longitudinal matrix
│   ├── document_concepts.csv                 # 133,255 document-concept rows
│   ├── concept_summary.csv                   # 2,790 concepts
│   ├── mesh_normalized.csv                   # MeSH normalized concepts
│   ├── topic_concept_mapping.csv             # 21,439 topic-concept mappings
│   ├── concept_summary_filtered.csv          # 737 quality-filtered concepts
│   ├── concept_quality_filter_report.json    # Filter exclusion breakdown
│   ├── concept_year_matrix.csv               # 11 × 300 annual publication matrix
│   ├── concept_trends.csv                    # Trend slopes & velocity
│   ├── semantic_similarity_pairs.csv         # Top 5,000 semantic pairs
│   ├── semantic_similarity_pairs_debiased.csv# Debiased similarity values
│   ├── concept_cooccurrence.csv              # 17,196 co-occurring concept pairs
│   ├── candidate_research_gaps.csv           # 50 ranked candidate research gaps
│   ├── gap_evidence.csv                      # Structured empirical evidence passages
│   ├── gap_quality_audit.csv                 # Substantive vs synonym quality audit
│   ├── phase1_summary.csv                    # Summary metrics CSV
│   └── phase1_summary.json                   # Summary metrics JSON
│
├── figures/                                  # 9 publication-grade figures
│   ├── papers_by_year.png
│   ├── topic_frequency.png
│   ├── topic_distribution.png
│   ├── topic_trends.png
│   ├── top_concepts.png
│   ├── concept_trends.png
│   ├── concept_cooccurrence_heatmap.png
│   ├── semantic_relationships.png
│   └── candidate_gap_ranking.png
│
├── logs/                                     # Execution logs directory
│
├── run_phase1.py                             # Master pipeline runner (reproducible end-to-end)
├── phase1_final_audit.py                     # 10-block comprehensive verification audit
├── phase1_final_report.md                    # Comprehensive Phase 1 completion and defense report
├── config.yaml                               # Central configuration parameters
├── requirements.txt                          # Python package dependencies
├── .gitignore                                # Git ignore patterns
├── .gitattributes                           # Git LFS tracking configuration
└── README.md                                 # This comprehensive documentation
```

---

## 5. Execution Instructions

### Environment Setup

```bash
pip install -r requirements.txt
```

### Running the Entire Pipeline

To run the master pipeline end-to-end (reuses precomputed embeddings and valid intermediate stages):

```bash
python run_phase1.py
```

### Running the Integrity Audit

To execute all 10 verification blocks (lineage, embeddings, official BERTopic reload, concept filtering, centroid debiasing, gap scores, evidence, and figures):

```bash
python phase1_final_audit.py
```

### Running the Notebooks

All 7 notebooks in `notebooks/` are pre-executed with visible cell outputs and can be run interactively or sequentially via Jupyter:

```bash
jupyter notebook notebooks/
```

---

## 6. Handoff to Phase 2

Phase 1 produces the following 3 structured files consumed by Phase 2:
1. `outputs/candidate_research_gaps.csv`: Ranked candidate research gaps with concept pairs, semantic similarities, co-occurrence scores, temporal velocity, and supporting PMIDs.
2. `outputs/gap_evidence.csv`: Literature-grounded evidence passages and publication volume statistics.
3. `outputs/gap_quality_audit.csv`: Quality classification distinguishing genuine substantive gaps from lexical synonym variants.

Phase 2 will implement the lightweight RAG verification, LLM hypothesis generation (guided strictly by candidate gaps), and the interactive Streamlit demonstration dashboard.
