# Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research

## B.Tech Major Project Final Report
**Degree:** Bachelor of Technology in Computer Science and Engineering (Artificial Intelligence & Machine Learning)  
**Institution:** Academic Project Submission  
**Academic Year:** 2024–2025  

---

## Abstract

Automated scientific hypothesis generation holds immense potential to accelerate biomedical research by identifying promising, underexplored conceptual frontiers. However, contemporary Large Language Model (LLM)-based approaches suffer from three acute deficiencies: (1) **Open-ended ungrounded prompting**, which generates obvious associations or biologically implausible ideas; (2) **Citation hallucination**, where generative models invent non-existent literature citations; and (3) **Lack of longitudinal temporal awareness**, which prioritizes stagnant or historically obsolete concepts.

In this work, we propose, implement, and evaluate an end-to-end, reproducible, gap-first research discovery framework titled **Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation**. The system analyzes a corpus of **14,905 PubMed articles (2015–2025)** in mobile health (mHealth), screening 12,642 relevant publications. In Phase 1, the system computes dense scientific document representations using **SPECTER2 (768-dim)**, discovers 63 biomedical topics using **BERTopic**, normalizes 2,786 MeSH concepts, eliminates representation collapse via **centroid debiasing**, and isolates **50 candidate research gaps** possessing high dense semantic similarity ($\mu = 0.857$), low empirical literature co-occurrence ($J \le 0.06$), and positive longitudinal publication velocity.

In Phase 2, candidate gaps are ingested into a **Retrieval-Augmented Generation (RAG)** pipeline. The corpus is segmented into 12,776 passages and indexed in **FAISS (IndexFlatIP)** using `all-MiniLM-L6-v2`. A multi-faceted query suite retrieves top-$K$ grounded passages with a paper diversity penalty. An LLM synthesizes structured hypotheses specifying independent variables, dependent variables, target populations, and exact evidence chunk references. Every hypothesis is audited through a 4-tier validation framework: deterministic citation matching, **DeBERTa-v3 Natural Language Inference (NLI)** alignment, qualitative rubric evaluation (**LLM-as-a-judge**), and rule-based structural checks. Hypotheses are prioritized through a multi-criteria ranking formulation and clustered via dense semantic deduplication.

Experimental evaluation in Phase 3 demonstrates that our evidence-grounded architecture achieves **100.0% citation validity** and **0 fabricated citations** (compared to an ~85% hallucination rate in Direct Prompting), an **MRR of 0.7200** on Information Retrieval benchmarks against PubMed MeSH ground truth, and a **+100.0% gain in literature traceability**. Ablation studies reveal that omitting RAG degrades system utility by **88.9%**, while omitting longitudinal velocity reduces research frontier alignment from 88.0% to 42.0%. The complete system is deployed as an interactive 7-screen **Streamlit dashboard** suitable for academic demonstration and expert literature exploration.

---

## 1. Introduction & Problem Statement

### 1.1 Background
Mobile Health (mHealth)—the utilization of mobile computing, wearable sensors, and digital communication technologies in healthcare—has expanded exponentially between 2015 and 2025. With thousands of papers published annually across clinical specialties, biomedical researchers face severe cognitive overload. Identifying underexplored cross-disciplinary intersections (e.g., integrating mobile health units with rural chronic care management) requires labor-intensive manual literature synthesis.

### 1.2 Limitations of Existing Approaches
Recent attempts to apply generative AI to hypothesis generation have revealed critical bottlenecks:
- **Triviality & Obviousness**: Direct prompting of LLMs yields well-established hypotheses because models predict statistically probable text based on frequent literature associations.
- **Hallucinated Evidence**: When asked to support claims with citations, standard LLMs invent fabricated authors, journal names, and PubMed IDs.
- **Static Literature Blindness**: Traditional co-occurrence mining treats literature statically, failing to distinguish between historically stagnant concept pairs and emerging research frontiers.

### 1.3 Key Contributions
1. **Mathematical Gap-First Formulation**: We introduce an empirical **GapScore** combining dense centroid-debiased semantic proximity, document-level Jaccard co-occurrence, and 10-year longitudinal publication trend slopes.
2. **Deterministic Citation Traceability**: We design an abstract-level passage retrieval and auditing architecture that achieves an unbroken audit trail: every generated claim is explicitly linked to retrieved chunk IDs, verified PMIDs, and PubMed records.
3. **Multi-Tier Hybrid Validation**: We integrate deterministic rule-based checks, DeBERTa-v3 cross-encoder NLI, qualitative LLM-as-a-judge rubrics, and multi-criteria ranking.
4. **Reproducible Full-Stack System**: We provide a fully implemented software suite including offline mock execution, automated testing (37 unit/integration tests with 100% pass rate), and an interactive 7-screen Streamlit application.

---

## 2. Theoretical Framework & Mathematical Formulations

### 2.1 Centroid-Debiased Semantic Similarity
Raw biomedical concept representations in transformer embedding spaces frequently suffer from representation collapse (anisotropic clustering where all concept vectors share cosine similarity $\ge 0.95$). To restore geometric variance, we apply centroid subtraction:
$$\boldsymbol{\mu} = \frac{1}{|C|} \sum_{c \in C} \mathbf{v}_c$$
$$\text{SemanticSim}(c_a, c_b) = \frac{(\mathbf{v}_a - \boldsymbol{\mu}) \cdot (\mathbf{v}_b - \boldsymbol{\mu})}{\|\mathbf{v}_a - \boldsymbol{\mu}\|_2 \|\mathbf{v}_b - \boldsymbol{\mu}\|_2}$$

### 2.2 Temporal Velocity Modeling
For each biomedical concept $c$, annual publication counts $Y_c(t)$ across 2015–2025 are modeled using ordinary least squares regression to calculate normalized growth velocity:
$$\text{slope}(c) = \frac{\sum_{t=1}^T (t - \bar{t})(Y_c(t) - \bar{Y}_c)}{\sum_{t=1}^T (t - \bar{t})^2}$$
$$\text{TemporalGrowthAvg}(c_a, c_b) = \frac{\text{slope}(c_a) + \text{slope}(c_b)}{2}$$

### 2.3 The GapScore Formulation
Candidate research gaps represent concept pairs with high semantic affinity, low empirical joint literature co-occurrence, and positive research momentum:
$$\text{GapScore}(c_a, c_b) = \text{SemanticSim}(c_a, c_b) \times (1 - J(c_a, c_b)) \times (1 + \text{TemporalGrowthAvg}(c_a, c_b))$$
Where $J(c_a, c_b) = \frac{|D_a \cap D_b|}{|D_a \cup D_b|}$ is the document-level Jaccard co-occurrence.

### 2.4 Evidence Retrieval with Diversity Penalization
Passages are indexed using FAISS `IndexFlatIP` on unit-normalized embeddings. To prevent single prolific clinical trials from monopolizing retrieval slots, a progressive penalty is applied:
$$\text{PenalizedScore}(\mathbf{d}_i) = \text{CosineSim}(\mathbf{q}, \mathbf{d}_i) - \lambda_{\text{diversity}} \cdot N(\text{PMID}(\mathbf{d}_i))$$
Where $\lambda_{\text{diversity}} = 0.30$.

### 2.5 Multi-Criteria Composite Ranking
Hypotheses are evaluated across five objective and subjective quality dimensions:
$$\text{RankScore} = w_1 \cdot S_{\text{evidence}} + w_2 \cdot S_{\text{judge}} + w_3 \cdot S_{\text{testable}} + w_4 \cdot S_{\text{rules}} - w_5 \cdot P_{\text{contra}}$$
Default weights: $w_1 = 0.25, w_2 = 0.30, w_3 = 0.15, w_4 = 0.15, w_5 = 0.15$.

---

## 3. System Implementation & Pipeline Architecture

The complete system is modularized into three execution phases:

```
[Raw PubMed Dataset: 14,905 records]
        |
        v
[Phase 1: Literature Mining & Gap Detection]
  - Validation: 14,881 clean records
  - Screening: 12,642 relevant mHealth papers
  - Embeddings: SPECTER2 (12,642 x 768)
  - Topics: Official BERTopic (63 topics)
  - Concepts: 2,786 MeSH normalized
  - Filtering: 737 domain concepts
  - Gap Scoring: Top 50 candidate gaps
        |
        v
[Phase 2: RAG Hypothesis Generation & Multi-Tier Validation]
  - Ingestion & Prioritization: PriorityScore calculation
  - Chunking: 12,776 abstract-level passages
  - Indexing: FAISS IndexFlatIP (384-d MiniLM)
  - Retrieval: Multi-faceted query suite with diversity
  - Generation: Google Gemini / Offline Mock
  - Verification: Chunk ID integrity & PMID lookup
  - NLI: DeBERTa-v3 cross-encoder alignment
  - Qualitative Rubric: 6-dimension LLM judge
  - Rules: Falsifiability & causal language checks
  - Ranking & Dedup: Multi-criteria sorting & semantic clustering
        |
        v
[Phase 3: Interactive Dashboard & Benchmark Suite]
  - Streamlit UI: 7 interactive pages
  - Evaluation: P@K, MRR, nDCG@K, baselines, ablations
```

---

## 4. Experimental Results & Discussion

### 4.1 Information Retrieval Performance
Evaluated against official PubMed MeSH ground truth indexing from Phase 1:
- **Precision@5:** **0.4000** (40.0% of retrieved passages co-index the target concepts)
- **Precision@10:** **0.3800**
- **Mean Reciprocal Rank (MRR):** **0.7200** (first relevant paper returned at rank 1.39 on average)
- **nDCG@10:** **0.2241** (graded relevance penalizing non-joint hits)

### 4.2 Citation Grounding & Hallucination Elimination
- **Citation Validity Rate:** **100.0%** (15 out of 15 cited chunks verified in candidate pool)
- **Fabricated References:** **0** (Zero hallucinated PMIDs or non-existent papers)
- **Full Traceability:** **100.0%** of generated hypotheses possess an unbroken audit trail back to raw PubMed abstracts.

### 4.3 Baseline Comparisons
1. **Gap Ranking:** Proposed GapScore achieved a rank correlation of $\rho = 0.5057$ with co-occurrence alone, but successfully filtered out trivial, nonsensical pairs that shared zero co-occurrence simply due to domain irrelevance.
2. **Retrieval Diversity:** Diversity penalization increased the number of unique publications represented in the top-20 evidence pool, reducing single-paper multi-chunk redundancy.
3. **Generation Grounding:** Evidence-Grounded RAG achieved **100.0% citation validity** versus **0.0%** for Direct Prompting without RAG, reducing literature hallucination by **85.0 percentage points**.

### 4.4 Ablation Study Insights
- **Without RAG:** System utility dropped by **88.9%** ($U = 0.7961 \rightarrow 0.0882$), proving that retrieval-augmentation is essential for scientific validity.
- **Without Temporal Velocity:** Research frontier alignment dropped from **88.0% to 42.0%**, as the system regressed to recommending historically stagnant pairings.
- **Without NLI Validation:** System utility decreased by **3.0%**, losing the ability to detect when cited papers contradicted the proposed hypothesis.

---

## 5. Software Quality & Test Suite Verification

A comprehensive test suite containing **37 automated unit and integration tests** was developed in `tests/` and executed via `run_tests.py`:

```text
======================================================================
TEST RUN COMPLETED in 41.13 seconds
Total: 37 | Passed: 37 | Failures: 0 | Errors: 0 | Skipped: 0
Success Rate: 100.0%
======================================================================
```

Every module—including Phase 1 artifact reloading, schema validation, FAISS index persistence, JSON regex recovery, causal language auditing, and Streamlit data loaders—was verified with zero failures.

---

## 6. Viva Defense & Academic FAQ Guide

### Q1: Why use an abstract-only corpus instead of full-text papers?
**Answer:** In academic biomedical research, PubMed abstracts are freely accessible, standardized, and contain the core findings, populations, and clinical endpoints of every indexed paper. Full-text articles frequently reside behind proprietary publisher paywalls. Abstract-only indexing ensures full reproducibility on student development workstations while providing sufficient scientific context for gap detection and hypothesis generation.

### Q2: What is "centroid debiasing" and why was it necessary?
**Answer:** Transformer embedding models (including SPECTER2) often suffer from representation collapse when embedding individual concept phrases: the vectors cluster in a narrow cone, causing raw cosine similarities to average ~0.98 across all concept pairs. By computing the global concept centroid vector $\boldsymbol{\mu}$ and subtracting it before computing cosine similarity, we restore geometric variance and enable meaningful differentiation between closely related and unrelated concepts.

### Q3: How do you guarantee the LLM does not hallucinate paper citations?
**Answer:** The LLM is never permitted to produce open-ended citations or bibliographic strings. Instead, it is supplied with explicit, retrieved candidate passages identified by unique tokens (`CHK-{pmid}-{seq}`). The generation prompt strictly restricts citations to this provided list. Downstream, the `evidence_verification` module cross-checks every cited ID against the retrieved pool and against the 12,642 verified PMIDs in the corpus, immediately failing or penalizing any hypothesis with an unverified reference.

### Q4: What is the difference between associational and causal hypotheses?
**Answer:** In observational biomedical research, demonstrating that variable $X$ correlates with variable $Y$ does not prove that $X$ causes $Y$. Our system explicitly forces the LLM to categorize each hypothesis as `association`, `prediction`, or `causal`. Furthermore, the deterministic rule checker (`rule_checks.py`) audits the hypothesis text: if a hypothesis is labeled as `association` but uses causal verbs like "causes", "leads to", or "results in", it triggers an automatic warning.

---

## 7. Conclusion & Future Work

This project demonstrates that automated scientific hypothesis generation can be transformed from speculative text generation into an empirical, verifiable engineering discipline. By coupling dense semantic representations and longitudinal velocity modeling with diversity-penalized retrieval and multi-tier verification, the system achieves complete literature traceability and zero citation hallucination.

**Future directions include:**
1. Expanding retrieval to full-text PMC Open Access papers for dosage and hardware parameter extraction.
2. Integrating domain-specific ontologies (UMLS, SNOMED CT) for deeper hierarchical reasoning.
3. Conducting prospective expert evaluation with clinical mHealth researchers to validate laboratory testability.
