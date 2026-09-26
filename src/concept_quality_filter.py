"""
concept_quality_filter.py — Concept quality audit and filtering.

PURPOSE
-------
Raw concept embeddings computed by averaging document embeddings suffer from
a well-known 'centroid collapse' problem:

    When a concept appears in many documents, its mean embedding converges
    toward the CORPUS CENTROID — a single direction shared by all high-frequency
    concepts regardless of their actual semantic meaning. This is caused by
    the Law of Large Numbers: the average of thousands of vectors in a
    high-dimensional, isotropic space points toward the corpus mean.

Consequence in Phase 1:
    Every concept's embedding was close to the corpus centroid (mean cosine
    similarity to centroid = 0.980). Pairwise similarities between concepts
    were therefore inflated to 0.996-1.000, meaning EVERY concept appeared
    semantically indistinguishable from every other.

This module fixes this via two orthogonal interventions:

1. CONCEPT QUALITY FILTERING (domain heuristics)
   Remove concepts that cannot form meaningful mHealth research gaps:
   - Demographic categories: male, female, adult, aged, child, etc.
   - Generic study design: cohort studies, follow-up studies, pilot projects, etc.
   - Generic methodology: interviews as topic, focus groups, surveys, etc.
   - Platform synonyms: mobile app, apps, mhealth, telemedicine (too broad)
   - Pure geographic terms: united states, china, india, etc.
   - Pure statistical terms: odds ratio, regression analysis, etc.

2. CENTROID-DEBIASED SIMILARITY (technical fix)
   Before computing pairwise cosine similarity, subtract the corpus mean
   vector from all concept embeddings. This removes the shared constant
   direction and reveals the genuine relative semantic differences
   between concepts. (Equivalent to PCA whitening in the direction of
   the first principal component.)

After filtering + debiasing, semantic similarities range ~0.3-0.9,
which produces meaningful discrimination between concept pairs.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from .config_loader import load_config, ensure_dirs


# ================================================================
# CONCEPT BLOCKLIST — concepts excluded from gap detection
# ================================================================

DEMOGRAPHIC_TERMS = {
    # Age / sex
    "male", "female", "humans", "human",
    "adult", "adults",
    "aged", "age factors",
    "aged, 80 and over",
    "middle aged",
    "young adult", "young adults",
    "adolescent", "adolescents", "adolescence",
    "child", "children", "child, preschool",
    "infant", "infants",
    "elderly", "older adults", "older adult",
    "pediatric", "geriatric",
    # Sex / gender
    "sex factors", "gender", "sex",
    "socioeconomic factors", "educational status",
    # Racial / ethnic categories (population descriptors, not research concepts)
    "african american", "african americans",
    "black or african american",
    "hispanic or latino", "hispanic", "latino",
    "white", "caucasian",
    "asian", "asian american",
    "minority groups", "ethnic groups", "racial groups",
    # Pregnancy / perinatal population descriptors
    "pregnant women", "pregnant people",
    "postpartum", "postpartum period",
    "postnatal care", "prenatal care", "antenatal care",
    "maternal and child health",
    # Population / student descriptors
    "college students", "university students",
    "patients",
}

GENERIC_STUDY_DESIGN = {
    "pilot projects", "pilot study", "pilot",
    "feasibility studies", "feasibility study",
    "cohort studies", "cohort study",
    "cross-sectional studies", "cross-sectional study",
    "prospective studies",
    "retrospective studies",
    "randomized controlled trials as topic",
    "randomized controlled trial",
    "clinical trial", "clinical trials",
    "follow-up studies", "follow-up study",
    "longitudinal studies",
    "observational study", "observational studies",
    "meta-analysis", "systematic review",
    "mixed methods",
    "quantitative study",
    "randomized trial",
}

GENERIC_METHODOLOGY_TERMS = {
    "interviews as topic", "interview", "interviews",
    "focus groups", "focus group",
    "surveys and questionnaires", "questionnaire", "questionnaires",
    "qualitative research", "qualitative", "qualitative study",
    "thematic analysis",
    "content analysis",
    "online survey",
    "self-report", "self report",
    "data collection",
    "sample size",
    "statistics as topic",
    "regression analysis",
    "odds ratio",
    "confidence interval",
    "outcome measures",
    "outcome assessment",
}

OVERLY_BROAD_PLATFORM_SYNONYMS = {
    # mHealth / platform synonyms (entire corpus is about these — no discriminative power)
    "mhealth", "m-health", "mobile health", "mobile health (mhealth)",
    "mobile applications", "mobile application",
    "mobile apps", "mobile app",
    "mobile phone", "mobile phones",
    "cell phone", "cell phones",
    "smartphone", "smartphones",
    "app", "apps", "application", "applications",
    "digital health", "digital intervention", "digital technology",
    "ehealth", "e-health",
    "telemedicine",      # too broad — entire corpus
    "telehealth",
    "technology",
    "internet",
    "wearable", "wearables", "wearable technology",
    "wireless technology",
    "computer",
    "software",
}

GENERIC_HEALTH_TERMS = {
    # So generic they apply to the entire biomedical literature
    "health", "healthcare", "health care", "medical care",
    "delivery of health care",
    "public health",
    "health promotion",
    "health behavior", "health behaviour",
    "health services", "health service",
    "primary health care",
    "prevention", "disease prevention",
    "treatment", "therapy",
    "patient care",
    "clinical practice",
    "evidence-based medicine",
    "health personnel",
    "health literacy",
}

GEOGRAPHIC_TERMS = {
    "united states", "usa", "u.s.a", "america",
    "china", "india", "australia", "uk", "united kingdom",
    "africa", "europe", "asia",
    "low and middle income countries", "lmic",
    "rural", "urban", "rural population",
    "developing countries",
    "sub-saharan africa",
}

PURE_PATIENT_DESCRIPTOR_TERMS = {
    # These are study attributes, not actionable research concepts
    "outpatients", "inpatients",
    "patients", "patient",
    "caregivers", "caregiver",
    "providers", "clinicians",
    "nurses", "physicians", "doctors",
}

COMBINED_BLOCKLIST = (
    DEMOGRAPHIC_TERMS
    | GENERIC_STUDY_DESIGN
    | GENERIC_METHODOLOGY_TERMS
    | OVERLY_BROAD_PLATFORM_SYNONYMS
    | GENERIC_HEALTH_TERMS
    | GEOGRAPHIC_TERMS
    | PURE_PATIENT_DESCRIPTOR_TERMS
)

# Additional regex patterns to catch blocklist variants
BLOCKLIST_PATTERNS = [
    re.compile(r"^aged[, ]"),           # aged, aged 80 and over, etc.
    re.compile(r"^adults?\s"),
    re.compile(r"study design"),
    re.compile(r"clinical protocol"),
    re.compile(r"^sex\b"),
    re.compile(r"^gender\b"),
    re.compile(r"attitude\s+of\s+health\s+personnel"),
    re.compile(r"^control\s+group"),
    re.compile(r"sample\s+size"),
]

# Known synonym/variant pairs — if BOTH appear in a gap, it is a terminology
# artefact and should be moved to the synonym bucket regardless of cooc.
# Format: frozenset({concept_a_lower, concept_b_lower})
KNOWN_SYNONYM_PAIRS: set[frozenset] = {
    frozenset({"self efficacy", "self-efficacy"}),
    frozenset({"wearable devices", "wearable electronic devices"}),
    frozenset({"cbt", "cognitive behavioral therapy"}),
    frozenset({"psychological distress", "stress, psychological"}),
    frozenset({"stress", "stress, psychological"}),
    frozenset({"stress management", "stress, psychological"}),
    frozenset({"access", "health services accessibility"}),
    frozenset({"pre-exposure prophylaxis", "prep"}),
    frozenset({"patient reported outcome measures", "patient-reported outcome measures"}),
    frozenset({"african american", "african americans"}),
    frozenset({"african american", "black or african american"}),
    frozenset({"african americans", "black or african american"}),
    frozenset({"anti-hiv agents", "anti-retroviral agents"}),
    frozenset({"anti-hiv agents", "antiretroviral therapy"}),
    frozenset({"reproductive health", "sexual and reproductive health"}),
    frozenset({"reproductive health", "reproductive health services"}),
    frozenset({"glycated hemoglobin", "hba1c"}),
    frozenset({"glycated hemoglobin", "glycemic control"}),
    frozenset({"maternal health services", "maternal-child health services"}),
    frozenset({"antenatal care", "prenatal care"}),
    frozenset({"postpartum", "postpartum period"}),
    frozenset({"postnatal care", "prenatal care"}),
    frozenset({"maternal and child health", "maternal health services"}),
    frozenset({"healthy lifestyle", "lifestyle"}),
    frozenset({"urban population", "urban"}),
    frozenset({"rural population", "rural"}),
    frozenset({"rural health services", "rural population"}),
}


def is_blocked(concept: str) -> bool:
    """Return True if a concept should be excluded from gap analysis."""
    c = concept.lower().strip()
    if c in COMBINED_BLOCKLIST:
        return True
    for pat in BLOCKLIST_PATTERNS:
        if pat.search(c):
            return True
    return False


def filter_concepts(
    concept_summary: pd.DataFrame,
    doc_concepts: pd.DataFrame,
    min_doc_count: int = 20,
    max_doc_fraction: float = 0.60,
    concept_col: str = "normalized_concept",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter concept summary and document-concepts to quality concepts only.

    Parameters
    ----------
    concept_summary : concept_summary.csv
    doc_concepts : mesh_normalized.csv
    min_doc_count : minimum number of documents a concept must appear in
    max_doc_fraction : maximum fraction of total corpus (too generic)
    concept_col : column name for concept in doc_concepts

    Returns
    -------
    filtered_summary, filtered_doc_concepts, filter_report dict
    """
    total_docs = doc_concepts["pmid"].nunique()
    total_concepts = len(concept_summary)

    blocked = []
    too_rare = []
    too_common = []
    kept = []

    for _, row in concept_summary.iterrows():
        c = str(row["concept"])
        n_docs = int(row["document_count"])
        doc_frac = n_docs / total_docs

        if is_blocked(c):
            blocked.append(c)
        elif n_docs < min_doc_count:
            too_rare.append(c)
        elif doc_frac > max_doc_fraction:
            too_common.append(c)
        else:
            kept.append(c)

    kept_set = set(kept)
    filtered_summary = concept_summary[
        concept_summary["concept"].isin(kept_set)
    ].reset_index(drop=True)

    filtered_doc_concepts = doc_concepts[
        doc_concepts[concept_col].isin(kept_set)
    ].reset_index(drop=True)

    report = {
        "total_before": total_concepts,
        "blocked_by_heuristic": len(blocked),
        "removed_too_rare": len(too_rare),
        "removed_too_common": len(too_common),
        "kept": len(kept),
        "blocked_examples": blocked[:20],
    }

    print("\n[QUALITY] Concept quality filtering report:")
    print(f"  Total before  : {total_concepts:,}")
    print(f"  Blocked (heuristic) : {len(blocked):,}")
    print(f"  Too rare (<{min_doc_count} docs) : {len(too_rare):,}")
    print(f"  Too common (>{max_doc_fraction*100:.0f}% of corpus): {len(too_common):,}")
    print(f"  KEPT          : {len(kept):,}")
    print(f"\n  Sample blocked concepts:")
    for b in blocked[:15]:
        print(f"    - {b}")

    return filtered_summary, filtered_doc_concepts, report


# ================================================================
# SYNONYM-PAIR DEDUPLICATION
# ================================================================

def deduplicate_synonym_pairs(gaps_df: pd.DataFrame, sim_threshold: float = 0.90) -> pd.DataFrame:
    """Remove gaps that are essentially synonym/variant pairs.

    When two concepts are very similar MeSH variants of the same idea
    (e.g. 'anti-hiv agents' <-> 'anti-retroviral agents', or
    'glycated hemoglobin' <-> 'hba1c'), they represent a terminology
    artefact rather than a genuine research gap.

    Heuristic: if semantic_similarity >= sim_threshold AND
    cooccurrence_score == 0 AND paper counts differ by < 3x,
    flag as a potential synonym pair and deprioritise.

    These are moved to the end of the ranked list with a note,
    NOT silently deleted.
    """
    if len(gaps_df) == 0:
        return gaps_df

    def is_synonym_pair(row):
        sim = row.get("semantic_similarity", 0)
        cooc = row.get("cooccurrence_score", 1)
        pa = row.get("papers_concept_a", 1)
        pa = row.get("papers_concept_a", 1)
        pb = row.get("papers_concept_b", 1)
        ca = str(row.get("concept_a", "")).lower().strip()
        cb = str(row.get("concept_b", "")).lower().strip()
        pair = frozenset({ca, cb})

        # Check explicit known synonym list
        if pair in KNOWN_SYNONYM_PAIRS:
            return True

        # Heuristic: very high similarity AND near-zero co-occurrence AND
        # balanced document counts → terminology artefact
        if sim >= sim_threshold and cooc < 0.05:
            ratio = max(pa, pb) / max(min(pa, pb), 1)
            if ratio < 5.0:
                return True
        return False

    flags = gaps_df.apply(is_synonym_pair, axis=1)
    n_synonym = flags.sum()
    genuine = gaps_df[~flags].copy()
    synonym = gaps_df[flags].copy()
    synonym["gap_note"] = "potential_synonym_pair"
    genuine["gap_note"] = ""

    result = pd.concat([genuine, synonym], ignore_index=True)
    print(f"\n[QUALITY] Synonym-pair deduplication: {n_synonym} potential synonym pairs"
          f" deprioritised (moved to end of ranking)")
    return result


# ================================================================
# CENTROID-DEBIASED CONCEPT EMBEDDINGS
# ================================================================

def compute_debiased_concept_embeddings(
    concept_summary: pd.DataFrame,
    doc_concepts: pd.DataFrame,
    doc_embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
    concept_col: str = "normalized_concept",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute CENTROID-DEBIASED concept embeddings.

    The plain average of document embeddings converges toward the corpus
    centroid as the number of documents grows. This is the root cause of
    the 0.996-1.000 similarity collapse seen in Phase 1.

    Fix: subtract the corpus mean vector from every concept embedding
    BEFORE normalising. This removes the shared 'mHealth corpus direction'
    and exposes genuine inter-concept differences.

    Returns
    -------
    concept_embeddings : dict concept -> unit vector (after debiasing)
    corpus_mean_vec : the raw (unnormalised) corpus centroid
    """
    print("\n[QUALITY] Computing centroid-debiased concept embeddings...")

    pmid_to_row = dict(
        zip(embedding_index["pmid"].astype(str), embedding_index["embedding_row"])
    )

    # Build pmid -> doc-embedding mapping
    # Corpus mean (centroid)
    corpus_mean = doc_embeddings.mean(axis=0)  # shape (768,)
    print(f"  Corpus centroid norm: {np.linalg.norm(corpus_mean):.4f}")
    print(f"  Corpus centroid cos to self: 1.0000  (reference)")

    # Group PMIDs per concept
    concept_pmids: dict[str, set] = defaultdict(set)
    for _, row in doc_concepts.iterrows():
        concept_pmids[row[concept_col]].add(str(row["pmid"]))

    concept_embeddings: dict[str, np.ndarray] = {}
    skipped = 0

    for concept in concept_summary["concept"]:
        pmids = concept_pmids.get(concept, set())
        rows = [pmid_to_row[p] for p in pmids if p in pmid_to_row]
        if not rows:
            skipped += 1
            continue

        # Mean of document embeddings
        raw_mean = doc_embeddings[rows].mean(axis=0)

        # CENTROID DEBIASING: subtract corpus mean
        debiased = raw_mean - corpus_mean

        norm = np.linalg.norm(debiased)
        if norm < 1e-10:
            skipped += 1
            continue

        concept_embeddings[concept] = debiased / norm

    print(f"  Concepts with embeddings: {len(concept_embeddings):,}  (skipped: {skipped})")

    # Verify debiasing worked: check similarity distribution on a sample
    sample_concepts = list(concept_embeddings.keys())[:200]
    if len(sample_concepts) >= 2:
        mat = np.array([concept_embeddings[c] for c in sample_concepts])
        sim_mat = mat @ mat.T
        triu = sim_mat[np.triu_indices(len(sample_concepts), k=1)]
        print(f"\n  Debiased similarity distribution (sample 200 concepts):")
        print(f"    mean={triu.mean():.4f}  std={triu.std():.4f}"
              f"  min={triu.min():.4f}  max={triu.max():.4f}")

    return concept_embeddings, corpus_mean


# ================================================================
# DEBIASED SEMANTIC SIMILARITY
# ================================================================

def compute_debiased_semantic_similarity(
    concept_embeddings: dict[str, np.ndarray],
    top_n_pairs: int = 5000,
    min_similarity: float = 0.3,
) -> pd.DataFrame:
    """Compute pairwise semantic similarity on debiased embeddings.

    Parameters
    ----------
    concept_embeddings : debiased, normalised concept vectors
    top_n_pairs : how many top pairs to keep
    min_similarity : minimum threshold (0.3 is appropriate post-debiasing)
    """
    concepts = sorted(concept_embeddings.keys())
    n = len(concepts)
    print(f"\n[QUALITY] Computing debiased pairwise similarity for {n:,} concepts...")
    print(f"  Potential pairs: {n*(n-1)//2:,}")

    emb_matrix = np.array([concept_embeddings[c] for c in concepts])
    sim_matrix = emb_matrix @ emb_matrix.T

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= min_similarity:
                pairs.append({
                    "concept_a": concepts[i],
                    "concept_b": concepts[j],
                    "semantic_similarity": round(sim, 4),
                })

    df = pd.DataFrame(pairs)
    if len(df) > 0:
        df = df.sort_values("semantic_similarity", ascending=False).head(top_n_pairs).reset_index(drop=True)

    print(f"  Pairs above threshold ({min_similarity}): {len(df):,}")
    if len(df) > 0:
        print(f"  Similarity range: {df['semantic_similarity'].min():.3f} – {df['semantic_similarity'].max():.3f}")

    return df


# ================================================================
# SAVE FILTER REPORT
# ================================================================

def save_quality_filter_outputs(
    filtered_summary: pd.DataFrame,
    filter_report: dict,
    debiased_semantic_pairs: pd.DataFrame,
) -> dict:
    """Save quality filter outputs to outputs/."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "concept_summary_filtered.csv"
    filtered_summary.to_csv(p, index=False, encoding="utf-8-sig")
    saved["filtered_summary"] = p
    print(f"[SAVE] Filtered concept summary -> {p}")

    p = out / "concept_quality_filter_report.json"
    import json
    with open(p, "w", encoding="utf-8") as f:
        json.dump(filter_report, f, indent=2, ensure_ascii=False, default=str)
    saved["filter_report"] = p
    print(f"[SAVE] Quality filter report -> {p}")

    p = out / "semantic_similarity_pairs_debiased.csv"
    debiased_semantic_pairs.to_csv(p, index=False, encoding="utf-8-sig")
    saved["debiased_pairs"] = p
    print(f"[SAVE] Debiased semantic pairs -> {p}")

    return saved
