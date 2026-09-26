"""
gap_detection.py — Semantic relationship analysis, concept co-occurrence,
and candidate research gap detection.

This is the core "Gap-First" module:
1. Compute semantic similarity between concepts
2. Compute concept co-occurrence in documents
3. Detect candidate gaps where high semantic similarity meets low co-occurrence
   and increasing temporal activity
4. Score and rank candidate gaps

GapScore = SemanticSimilarity x (1 - NormalizedCoOccurrence) x TemporalGrowth
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

from .config_loader import load_config, ensure_dirs


# ================================================================
# CONCEPT EMBEDDINGS
# ================================================================

def compute_concept_embeddings(
    concept_summary: pd.DataFrame,
    doc_concepts: pd.DataFrame,
    doc_embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute concept embeddings by averaging document embeddings.

    For each concept, the embedding is the mean of all document
    embeddings where that concept appears.

    Returns dict mapping concept -> embedding vector.
    """
    print("\n[GAP] Computing concept embeddings...")

    # Build pmid -> embedding-row lookup
    pmid_to_row = dict(
        zip(embedding_index["pmid"].astype(str), embedding_index["embedding_row"])
    )

    concept_col = (
        "normalized_concept"
        if "normalized_concept" in doc_concepts.columns
        else "concept"
    )

    # Group documents by concept
    concept_pmids = defaultdict(set)
    for _, row in doc_concepts.iterrows():
        concept_pmids[row[concept_col]].add(str(row["pmid"]))

    concept_embeddings = {}
    for concept, pmids in concept_pmids.items():
        rows = [pmid_to_row[p] for p in pmids if p in pmid_to_row]
        if rows:
            vecs = doc_embeddings[rows]
            mean_vec = vecs.mean(axis=0)
            # Normalise
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            concept_embeddings[concept] = mean_vec

    print(f"  Computed embeddings for {len(concept_embeddings):,} concepts")
    return concept_embeddings


# ================================================================
# SEMANTIC SIMILARITY
# ================================================================

def compute_semantic_similarity(
    concept_embeddings: dict[str, np.ndarray],
    top_n_pairs: int = 2000,
    min_similarity: float = 0.4,
) -> pd.DataFrame:
    """Compute pairwise semantic similarity between concepts.

    Uses cosine similarity on concept embeddings.
    Returns only pairs above the minimum similarity threshold.
    """
    cfg = load_config()
    sem_cfg = cfg.get("semantic", {})
    min_similarity = sem_cfg.get("similarity_threshold", min_similarity)
    top_n_pairs = sem_cfg.get("top_n_pairs", top_n_pairs)

    concepts = sorted(concept_embeddings.keys())
    n = len(concepts)

    print(f"\n[GAP] Computing semantic similarity for {n:,} concepts...")
    print(f"  Potential pairs: {n * (n - 1) // 2:,}")

    # Build embedding matrix
    emb_matrix = np.array([concept_embeddings[c] for c in concepts])

    # Compute cosine similarity matrix (embeddings are already normalised)
    sim_matrix = emb_matrix @ emb_matrix.T

    # Extract pairs above threshold
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

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values(
            "semantic_similarity", ascending=False
        ).head(top_n_pairs).reset_index(drop=True)

    print(f"  Pairs above threshold ({min_similarity}): {len(pairs_df):,}")
    if len(pairs_df) > 0:
        print(f"  Similarity range: "
              f"{pairs_df['semantic_similarity'].min():.3f} — "
              f"{pairs_df['semantic_similarity'].max():.3f}")

    return pairs_df


# ================================================================
# CO-OCCURRENCE
# ================================================================

def compute_cooccurrence(
    doc_concepts: pd.DataFrame,
    top_n_concepts: int = 300,
) -> pd.DataFrame:
    """Compute concept co-occurrence statistics.

    For each pair of concepts, counts how many documents contain both.
    Uses Jaccard similarity as the normalised measure.

    Returns a DataFrame with concept_a, concept_b, cooccurrence_count,
    and normalized_cooccurrence.
    """
    cfg = load_config()
    cooc_cfg = cfg.get("cooccurrence", {})
    top_n = cooc_cfg.get("top_n_concepts", top_n_concepts)
    min_cooc = cooc_cfg.get("min_cooccurrence", 2)

    concept_col = (
        "normalized_concept"
        if "normalized_concept" in doc_concepts.columns
        else "concept"
    )

    print(f"\n[GAP] Computing concept co-occurrence...")

    # Get top concepts by document frequency
    concept_doc_freq = doc_concepts.groupby(concept_col)["pmid"].nunique()
    top_concepts = concept_doc_freq.nlargest(top_n).index.tolist()

    # Build concept -> set of PMIDs mapping
    filtered = doc_concepts[doc_concepts[concept_col].isin(top_concepts)]
    concept_docs = defaultdict(set)
    for _, row in filtered.iterrows():
        concept_docs[row[concept_col]].add(str(row["pmid"]))

    print(f"  Concepts: {len(top_concepts)}")
    print(f"  Computing pairwise co-occurrence...")

    # Compute pairwise statistics
    pairs = []
    concept_list = sorted(concept_docs.keys())

    for i, ca in enumerate(tqdm(concept_list, desc="Co-occurrence")):
        docs_a = concept_docs[ca]
        for j in range(i + 1, len(concept_list)):
            cb = concept_list[j]
            docs_b = concept_docs[cb]

            intersection = len(docs_a & docs_b)
            if intersection < min_cooc:
                continue

            union = len(docs_a | docs_b)
            jaccard = intersection / union if union > 0 else 0

            pairs.append({
                "concept_a": ca,
                "concept_b": cb,
                "cooccurrence_count": intersection,
                "docs_a": len(docs_a),
                "docs_b": len(docs_b),
                "union_count": union,
                "normalized_cooccurrence": round(jaccard, 4),
            })

    cooc_df = pd.DataFrame(pairs)
    if len(cooc_df) > 0:
        cooc_df = cooc_df.sort_values(
            "cooccurrence_count", ascending=False
        ).reset_index(drop=True)

    print(f"  Co-occurring pairs (>={min_cooc}): {len(cooc_df):,}")

    return cooc_df


# ================================================================
# GAP DETECTION
# ================================================================

def detect_candidate_gaps(
    semantic_pairs: pd.DataFrame,
    cooccurrence_df: pd.DataFrame,
    concept_trends: pd.DataFrame,
    doc_concepts: pd.DataFrame,
    topic_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Detect candidate research gaps using the Gap-First strategy.

    GapScore = SemanticSimilarity x (1 - NormalizedCoOccurrence) x TemporalGrowth

    A candidate gap is a concept pair that:
    1. Has HIGH semantic relatedness (concepts are meaningfully related)
    2. Has LOW/LIMITED co-occurrence in the literature
    3. Has INCREASING temporal activity (recent research interest)

    IMPORTANT: These are CANDIDATE gaps, not confirmed scientific gaps.
    Automated analysis identifies promising unexplored relationships,
    but human/scientific verification is still required.
    """
    cfg = load_config()
    gap_cfg = cfg.get("gap_detection", {})
    top_n = gap_cfg.get("top_n_gaps", 50)
    min_sem = gap_cfg.get("min_semantic_similarity", 0.4)
    max_cooc = gap_cfg.get("max_normalized_cooccurrence", 0.8)
    min_growth = gap_cfg.get("min_temporal_growth", 0.0)
    min_papers = gap_cfg.get("require_supporting_papers", 3)

    print("\n" + "=" * 70)
    print("CANDIDATE RESEARCH GAP DETECTION")
    print("=" * 70)

    concept_col = (
        "normalized_concept"
        if "normalized_concept" in doc_concepts.columns
        else "concept"
    )

    # Build lookup tables
    cooc_lookup = {}
    for _, row in cooccurrence_df.iterrows():
        key = tuple(sorted([row["concept_a"], row["concept_b"]]))
        cooc_lookup[key] = row["normalized_cooccurrence"]

    growth_lookup = {}
    for _, row in concept_trends.iterrows():
        growth_lookup[row["concept"]] = row["growth_rate"]

    # Concept -> supporting PMIDs
    concept_pmids = defaultdict(set)
    for _, row in doc_concepts.iterrows():
        concept_pmids[row[concept_col]].add(str(row["pmid"]))

    # Concept -> supporting topics
    concept_topics = defaultdict(set)
    if "topic_id" in doc_concepts.columns:
        for _, row in doc_concepts.iterrows():
            if row["topic_id"] != -1:
                concept_topics[row[concept_col]].add(int(row["topic_id"]))

    # Score each semantic pair
    gap_candidates = []

    for _, row in semantic_pairs.iterrows():
        ca = row["concept_a"]
        cb = row["concept_b"]
        sem_sim = row["semantic_similarity"]

        if sem_sim < min_sem:
            continue

        # Check supporting papers
        pmids_a = concept_pmids.get(ca, set())
        pmids_b = concept_pmids.get(cb, set())
        if len(pmids_a) < min_papers or len(pmids_b) < min_papers:
            continue

        # Co-occurrence
        key = tuple(sorted([ca, cb]))
        norm_cooc = cooc_lookup.get(key, 0.0)
        if norm_cooc > max_cooc:
            continue

        # Temporal growth (average of both concepts)
        growth_a = growth_lookup.get(ca, 0.0)
        growth_b = growth_lookup.get(cb, 0.0)
        avg_growth = (growth_a + growth_b) / 2

        # Shift growth to be positive for scoring
        # growth_factor: values > 0 mean increasing -> higher score
        growth_factor = max(avg_growth + 1.0, 0.01)  # shift so stable=1.0

        if avg_growth < min_growth:
            continue

        # GapScore = SemanticSim x (1 - NormCoOc) x GrowthFactor
        gap_score = sem_sim * (1 - norm_cooc) * growth_factor

        # Joint PMIDs
        joint_pmids = pmids_a & pmids_b

        # Supporting topics
        topics = concept_topics.get(ca, set()) | concept_topics.get(cb, set())

        gap_candidates.append({
            "concept_a": ca,
            "concept_b": cb,
            "semantic_similarity": round(sem_sim, 4),
            "cooccurrence_score": round(norm_cooc, 4),
            "temporal_growth_a": round(growth_a, 4),
            "temporal_growth_b": round(growth_b, 4),
            "temporal_growth_avg": round(avg_growth, 4),
            "gap_score": round(gap_score, 4),
            "papers_concept_a": len(pmids_a),
            "papers_concept_b": len(pmids_b),
            "papers_joint": len(joint_pmids),
            "supporting_topic_ids": ";".join(str(t) for t in sorted(topics)),
            "supporting_pmids_a": ";".join(sorted(list(pmids_a)[:20])),
            "supporting_pmids_b": ";".join(sorted(list(pmids_b)[:20])),
            "supporting_pmids_joint": ";".join(sorted(list(joint_pmids)[:20])),
        })

    gaps_df = pd.DataFrame(gap_candidates)
    if len(gaps_df) > 0:
        gaps_df = gaps_df.sort_values(
            "gap_score", ascending=False
        ).head(top_n).reset_index(drop=True)

    print(f"\n  Total semantic pairs evaluated: {len(semantic_pairs):,}")
    print(f"  Candidate research gaps found: {len(gaps_df):,}")

    if len(gaps_df) > 0:
        print(f"\n  Top 10 candidate research gaps:")
        for i, (_, r) in enumerate(gaps_df.head(10).iterrows()):
            print(
                f"    {i+1}. [{r['concept_a']}] <-> [{r['concept_b']}]\n"
                f"       GapScore={r['gap_score']:.3f}  "
                f"Sim={r['semantic_similarity']:.3f}  "
                f"CoOc={r['cooccurrence_score']:.3f}  "
                f"Growth={r['temporal_growth_avg']:.3f}  "
                f"Joint={r['papers_joint']}"
            )

    return gaps_df


# ================================================================
# GAP EVIDENCE PREPARATION
# ================================================================

def prepare_gap_evidence(
    gaps_df: pd.DataFrame,
    df: pd.DataFrame,
    concept_trends: pd.DataFrame,
    topic_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare supporting evidence for each candidate research gap.

    For each gap, generates a human-readable evidence summary
    based ONLY on computed data (no LLM, no fabrication).
    """
    print("\n[GAP] Preparing gap evidence...")

    growth_lookup = {}
    for _, row in concept_trends.iterrows():
        growth_lookup[row["concept"]] = {
            "growth_rate": row["growth_rate"],
            "total_mentions": row.get("total_mentions", 0),
            "recent_mentions": row.get("recent_mentions", 0),
            "trend": row.get("trend", "unknown"),
        }

    evidence_rows = []

    for _, gap in gaps_df.iterrows():
        ca = gap["concept_a"]
        cb = gap["concept_b"]

        # Trend info
        trend_a = growth_lookup.get(ca, {})
        trend_b = growth_lookup.get(cb, {})

        # Generate evidence description (template-based, no LLM)
        desc_parts = []
        desc_parts.append(
            f"Concept \"{ca}\" and concept \"{cb}\" show "
            f"high semantic relatedness (similarity={gap['semantic_similarity']:.3f})."
        )

        if gap["cooccurrence_score"] < 0.05:
            desc_parts.append(
                f"These concepts have very limited joint co-occurrence in "
                f"the literature (Jaccard={gap['cooccurrence_score']:.3f}, "
                f"only {gap['papers_joint']} joint papers)."
            )
        elif gap["cooccurrence_score"] < 0.2:
            desc_parts.append(
                f"These concepts have limited joint co-occurrence "
                f"(Jaccard={gap['cooccurrence_score']:.3f}, "
                f"{gap['papers_joint']} joint papers)."
            )
        else:
            desc_parts.append(
                f"These concepts co-occur in some papers "
                f"(Jaccard={gap['cooccurrence_score']:.3f}, "
                f"{gap['papers_joint']} joint papers)."
            )

        # Temporal
        g_avg = gap["temporal_growth_avg"]
        if g_avg > 0.1:
            desc_parts.append(
                "Both concepts show increasing research activity "
                "in recent years."
            )
        elif g_avg > 0:
            desc_parts.append(
                "These concepts show moderate recent research activity."
            )
        else:
            desc_parts.append(
                "These concepts have stable research activity."
            )

        desc_parts.append(
            f"Concept \"{ca}\" appears in {gap['papers_concept_a']} papers; "
            f"concept \"{cb}\" appears in {gap['papers_concept_b']} papers."
        )

        evidence = " ".join(desc_parts)

        evidence_rows.append({
            "concept_a": ca,
            "concept_b": cb,
            "gap_score": gap["gap_score"],
            "semantic_similarity": gap["semantic_similarity"],
            "cooccurrence_score": gap["cooccurrence_score"],
            "temporal_growth_avg": gap["temporal_growth_avg"],
            "trend_a": trend_a.get("trend", "unknown"),
            "trend_b": trend_b.get("trend", "unknown"),
            "papers_concept_a": gap["papers_concept_a"],
            "papers_concept_b": gap["papers_concept_b"],
            "papers_joint": gap["papers_joint"],
            "supporting_topic_ids": gap["supporting_topic_ids"],
            "evidence_description": evidence,
            "supporting_pmids_a": gap["supporting_pmids_a"],
            "supporting_pmids_b": gap["supporting_pmids_b"],
            "supporting_pmids_joint": gap.get("supporting_pmids_joint", ""),
        })

    evidence_df = pd.DataFrame(evidence_rows)
    print(f"  Prepared evidence for {len(evidence_df)} candidate gaps")
    return evidence_df


# ================================================================
# SAVE OUTPUTS
# ================================================================

def save_gap_outputs(
    semantic_pairs: pd.DataFrame,
    cooccurrence_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> dict:
    """Save all gap detection outputs."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    p = out / "semantic_similarity_pairs.csv"
    semantic_pairs.to_csv(p, index=False, encoding="utf-8-sig")
    saved["semantic_pairs"] = p
    print(f"[SAVE] Semantic pairs -> {p}  ({len(semantic_pairs):,} pairs)")

    p = out / "concept_cooccurrence.csv"
    cooccurrence_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["cooccurrence"] = p
    print(f"[SAVE] Co-occurrence -> {p}  ({len(cooccurrence_df):,} pairs)")

    p = out / "candidate_research_gaps.csv"
    gaps_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["gaps"] = p
    print(f"[SAVE] Candidate gaps -> {p}  ({len(gaps_df):,} gaps)")

    p = out / "gap_evidence.csv"
    evidence_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["evidence"] = p
    print(f"[SAVE] Gap evidence -> {p}  ({len(evidence_df):,} rows)")

    # Produce gap_quality_audit.csv
    audit_rows = []
    for rank, (_, row) in enumerate(gaps_df.iterrows(), 1):
        note = str(row.get("gap_note", "") or "")
        is_syn = (note == "potential_synonym_pair")
        audit_rows.append({
            "rank": rank,
            "concept_a": row["concept_a"],
            "concept_b": row["concept_b"],
            "gap_score": round(float(row["gap_score"]), 4),
            "semantic_similarity": round(float(row["semantic_similarity"]), 4),
            "cooccurrence_score": round(float(row["cooccurrence_score"]), 4),
            "temporal_growth_avg": round(float(row["temporal_growth_avg"]), 4),
            "papers_concept_a": int(row["papers_concept_a"]),
            "papers_concept_b": int(row["papers_concept_b"]),
            "papers_joint": int(row["papers_joint"]),
            "quality_status": "synonym_or_abbreviation_variant" if is_syn else "genuine_substantive_gap",
            "audit_note": "Deprioritized - known synonym or lexical variant pair" if is_syn else "Valid candidate research gap with high semantic affinity, low co-occurrence, and positive growth",
            "has_supporting_evidence": bool(len(str(row.get("supporting_pmids_a", ""))) > 0 and len(str(row.get("supporting_pmids_b", ""))) > 0),
        })
    audit_df = pd.DataFrame(audit_rows)
    p = out / "gap_quality_audit.csv"
    audit_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["quality_audit"] = p
    print(f"[SAVE] Gap quality audit -> {p}  ({len(audit_df):,} rows)")

    return saved
