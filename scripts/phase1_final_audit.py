"""
phase1_final_audit.py
=====================
Comprehensive Phase 1 integrity audit.
Run from project root directory:
    python phase1_final_audit.py
"""
import sys, json, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from src.config_loader import ensure_dirs, load_config, get_project_root

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
issues = []

def check(cond, label, detail=""):
    if cond:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}  {detail}")
        issues.append(label)

def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

dirs = ensure_dirs()
OUT = dirs["outputs"]

# ============================================================
# BLOCK 1 — DATA LINEAGE
# ============================================================
section("BLOCK 1 — DATA LINEAGE")

cfg = load_config()
root = get_project_root()
raw_path = root / cfg["paths"].get("dataset", "data/pubmed_mhealth_2015_2025_proposed_final.xls")
if not raw_path.exists():
    raw_path = root.parent / "pubmed_mhealth_2015_2025_proposed_final.xls"
raw = pd.read_csv(raw_path, encoding="utf-8-sig", low_memory=False)
check(len(raw) == 14905, f"Raw record count == 14,905", f"got {len(raw)}")
check(raw["pmid"].nunique() == 14905, "Zero duplicate PMIDs in raw dataset")

cleaned = pd.read_csv(OUT / "mhealth_candidate_cleaned.csv", low_memory=False)
check(len(cleaned) == 14881, f"Cleaned count == 14,881", f"got {len(cleaned)}")
check(cleaned["pmid"].nunique() == 14881, "No duplicate PMIDs after cleaning")

screened = pd.read_csv(OUT / "mhealth_screened_corpus.csv", low_memory=False)
check(len(screened) == 12642, f"Screened count == 12,642", f"got {len(screened)}")
relevant_cnt   = (screened["relevance_label"] == "relevant").sum()
borderline_cnt = (screened["relevance_label"] == "borderline").sum()
check(relevant_cnt == 9279,   f"Relevant count == 9,279 (got {relevant_cnt})")
check(borderline_cnt == 3363, f"Borderline count == 3,363 (got {borderline_cnt})")
check("relevance_score" in screened.columns, "'relevance_score' column present")
check("relevance_reason" in screened.columns, "'relevance_reason' column present")

prep = pd.read_csv(OUT / "mhealth_preprocessed.csv", low_memory=False)
check(len(prep) == 12642, f"Preprocessed count == 12,642 (got {len(prep)})")

flog = pd.read_csv(OUT / "cleaning_filter_log.csv")
check(len(flog) >= 4, f"Filter log has >= 4 steps (got {len(flog)})")
print(f"\n  Filter log:")
for _, r in flog.iterrows():
    print(f"    {r['step']}: removed={r['removed']}, after={r['after']}")

# ============================================================
# BLOCK 2 — EMBEDDING INTEGRITY
# ============================================================
section("BLOCK 2 — EMBEDDING INTEGRITY")

emb = np.load(OUT / "document_embeddings.npy")
idx = pd.read_csv(OUT / "document_embedding_index.csv")
info_txt = (OUT / "embedding_model_info.txt").read_text(encoding="utf-8").strip()

check(emb.shape[0] == 12642, f"Embedding rows == 12,642 (got {emb.shape[0]})")
check(emb.shape[1] == 768,   f"Embedding dim  == 768 (got {emb.shape[1]})")
check(not np.isnan(emb).any(), "No NaN values in embeddings")
check(not np.isinf(emb).any(), "No Inf values in embeddings")

norms = np.linalg.norm(emb, axis=1)
check(np.allclose(norms, 1.0, atol=1e-5), f"All embeddings are unit-normalised (mean norm={norms.mean():.6f})")

check(len(idx) == 12642, f"Embedding index rows == 12,642")
check("pmid" in idx.columns and "embedding_row" in idx.columns, "Index has 'pmid' and 'embedding_row' columns")
check(list(idx["embedding_row"]) == list(range(12642)), "Embedding rows are contiguous 0..12641")

# PMID alignment: every PMID in screened should be in index
screened_pmids = set(screened["pmid"].astype(str))
idx_pmids = set(idx["pmid"].astype(str))
missing = screened_pmids - idx_pmids
check(len(missing) == 0, f"All screened PMIDs have embedding rows ({len(missing)} missing)")

print(f"\n  Embedding model: {info_txt}")

# ============================================================
# BLOCK 3 — TOPIC MODEL QUALITY
# ============================================================
section("BLOCK 3 — TOPIC MODEL QUALITY")

ta = pd.read_csv(OUT / "topic_assignments.csv")
ts = pd.read_csv(OUT / "topic_summary.csv")
tt = pd.read_csv(OUT / "topic_top_terms.csv")

check(len(ta) == 12642, f"Topic assignments count == 12,642 (got {len(ta)})")
n_topics = ta["topic_id"].nunique()
outlier_cnt = (ta["topic_id"] == -1).sum()
check(n_topics >= 2, f"At least 2 distinct topic IDs exist (got {n_topics})")
print(f"\n  Topic IDs: {n_topics} (including -1 outlier class)")
print(f"  Outlier docs: {outlier_cnt:,} ({100*outlier_cnt/len(ta):.1f}%)")

genuine_topics = ta[ta["topic_id"] >= 0]
print(f"  Docs in genuine topics: {len(genuine_topics):,}")

topic_sizes = ta.groupby("topic_id").size().sort_values(ascending=False)
print(f"\n  Top 10 topic sizes:")
for tid, sz in topic_sizes.head(10).items():
    print(f"    Topic {tid}: {sz} docs")

# Check top terms are present and non-empty
check("topic_id" in tt.columns and "term" in tt.columns, "Topic top-terms file has expected columns")
empty_terms = (tt["term"].fillna("").str.strip() == "").sum()
check(empty_terms == 0, f"No empty top-terms rows (found {empty_terms})")

# Check saved official BERTopic model artifact & reload verification
model_dir = OUT.parent / "models" / "bertopic_model"
check(model_dir.exists(), "BERTopic model directory exists at models/bertopic_model")
check((model_dir / "config.json").exists(), "BERTopic model has config.json")
check((model_dir / "topics.json").exists(), "BERTopic model has topics.json")
check((model_dir / "ctfidf.safetensors").exists(), "BERTopic model has ctfidf.safetensors")

try:
    from bertopic import BERTopic
    reloaded_model = BERTopic.load(str(model_dir))
    reloaded_n_topics = len(reloaded_model.get_topics()) - 1
    check(reloaded_n_topics == 63, f"Reloaded model has 63 genuine topics (got {reloaded_n_topics})")
    check(len(reloaded_model.get_topic(0)) > 0, "Reloaded model can retrieve topic words")
except Exception as e:
    check(False, f"BERTopic model reload failed: {e}")

# ============================================================
# BLOCK 4 — CONCEPT EXTRACTION INTEGRITY
# ============================================================
section("BLOCK 4 — CONCEPT EXTRACTION INTEGRITY")

cs = pd.read_csv(OUT / "concept_summary.csv")
dc = pd.read_csv(OUT / "document_concepts.csv")
mn = pd.read_csv(OUT / "mesh_normalized.csv")
cf = pd.read_csv(OUT / "concept_summary_filtered.csv")

check(len(cs) == 2790, f"Total concepts == 2,790 (got {len(cs)})")
check("concept" in cs.columns, "Concept summary has 'concept' column")
check("document_count" in cs.columns, "Concept summary has 'document_count' column")

check(len(dc) > 100000, f"Document-concept rows > 100,000 (got {len(dc):,})")

check(len(mn) > 0, f"Mesh normalized has rows (got {len(mn)})")
check("normalized_concept" in mn.columns, "Mesh normalized has 'normalized_concept' column")
check("pmid" in mn.columns, "Mesh normalized has 'pmid' column")

cf_count = len(cf)
check(700 <= cf_count <= 800, f"Filtered concepts in range 700-800 (got {cf_count})")

# Cross-check PMID coverage in mesh_normalized
mesh_pmids = set(mn["pmid"].astype(str))
emb_pmids = set(idx["pmid"].astype(str))
mesh_not_in_emb = mesh_pmids - emb_pmids
check(len(mesh_not_in_emb) == 0, f"All MeSH PMIDs have embeddings ({len(mesh_not_in_emb)} without)")

# ============================================================
# BLOCK 5 — TEMPORAL ANALYSIS
# ============================================================
section("BLOCK 5 — TEMPORAL ANALYSIS")

tym = pd.read_csv(OUT / "topic_year_matrix.csv", index_col=0)
cym = pd.read_csv(OUT / "concept_year_matrix.csv", index_col=0)
ct  = pd.read_csv(OUT / "concept_trends.csv")

years = list(range(2015, 2026))
tym_idx = [int(x) for x in tym.index.tolist()]
cym_idx = [int(x) for x in cym.index.tolist()]

check(all(y in tym_idx for y in years), f"Topic-year matrix index spans 2015-2025 (got {tym_idx})")
check(all(y in cym_idx for y in years), f"Concept-year matrix index spans 2015-2025 (got {cym_idx})")

# Verify counts are non-negative integers
check((tym.values >= 0).all(), "Topic-year matrix: all counts >= 0")
check((cym.values >= 0).all(), "Concept-year matrix: all counts >= 0")

check("growth_rate" in ct.columns, "Concept trends has 'growth_rate' column")
check("concept" in ct.columns, "Concept trends has 'concept' column")
check(ct["growth_rate"].between(-2, 2).all(), "All growth rates in reasonable range [-2, 2]")

# ============================================================
# BLOCK 6 — CENTROID DEBIASING VERIFICATION
# ============================================================
section("BLOCK 6 — CENTROID DEBIASING VERIFICATION")

# Recompute a sample of debiased embeddings to verify math
corpus_mean = emb.mean(axis=0)
corpus_mean_norm = np.linalg.norm(corpus_mean)
check(corpus_mean_norm > 0.8, f"Corpus centroid has significant norm: {corpus_mean_norm:.4f}")

# Verify cos(concept_vec, corpus_centroid) drops after debiasing
# Take concept "telemedicine" (common) and "wearable devices" (specific)
# Read pmid->row map
pmid_to_row = dict(zip(idx["pmid"].astype(str), idx["embedding_row"]))
corpus_centroid_n = corpus_mean / corpus_mean_norm

concept_cosines_raw = []
concept_cosines_debiased = []
sample_concepts = cf["concept"].head(50).tolist()

for concept in sample_concepts:
    pmids_c = set(mn[mn["normalized_concept"] == concept]["pmid"].astype(str))
    rows = [pmid_to_row[p] for p in pmids_c if p in pmid_to_row]
    if not rows:
        continue
    raw_mean = emb[rows].mean(axis=0)
    rn = np.linalg.norm(raw_mean)
    if rn < 1e-10:
        continue
    raw_n = raw_mean / rn
    cos_raw = float(raw_n @ corpus_centroid_n)
    concept_cosines_raw.append(cos_raw)

    debiased = raw_mean - corpus_mean
    dn = np.linalg.norm(debiased)
    if dn < 1e-10:
        continue
    deb_n = debiased / dn
    # cos to centroid after debiasing — should be much lower
    concept_cosines_debiased.append(float(deb_n @ corpus_centroid_n))

mean_raw   = np.mean(concept_cosines_raw)
mean_deb   = np.mean(concept_cosines_debiased)
print(f"\n  Raw concept-centroid cosine (sample {len(concept_cosines_raw)}): mean={mean_raw:.4f}")
print(f"  Debiased concept-centroid cosine (sample): mean={mean_deb:.4f}")
check(mean_raw > 0.95,   f"Raw embeddings were close to centroid (mean cos={mean_raw:.4f} > 0.95)")
check(abs(mean_deb) < 0.3, f"Debiasing reduced centroid proximity (mean cos={mean_deb:.4f})")

# Now verify pairwise similarity distribution is discriminative
debiased_pairs = pd.read_csv(OUT / "semantic_similarity_pairs_debiased.csv")
check(len(debiased_pairs) > 0, "Debiased similarity pairs file has rows")
if len(debiased_pairs) > 0:
    lo = debiased_pairs["semantic_similarity"].min()
    hi = debiased_pairs["semantic_similarity"].max()
    me = debiased_pairs["semantic_similarity"].mean()
    check(lo < 0.95, f"Min similarity < 0.95 (got {lo:.4f}) — not collapsed")
    check(me < 0.90,  f"Mean similarity < 0.90 (got {me:.4f}) — discriminative spread")
    print(f"\n  Debiased pairwise sim range: {lo:.4f} – {hi:.4f}  mean={me:.4f}")

# ============================================================
# BLOCK 7 — GAP SCORE FORMULA VERIFICATION
# ============================================================
section("BLOCK 7 — GAP SCORE FORMULA VERIFICATION")

gaps = pd.read_csv(OUT / "candidate_research_gaps.csv")
check(len(gaps) == 50, f"50 candidate-gap rows saved (got {len(gaps)})")

# Verify GapScore = Sim * (1 - CoOc) * (1 + Growth)
n_wrong = 0
for i, row in gaps.head(30).iterrows():
    expected = round(
        row["semantic_similarity"]
        * (1 - row["cooccurrence_score"])
        * (1 + row["temporal_growth_avg"]),
        4,
    )
    actual = round(row["gap_score"], 4)
    if abs(expected - actual) > 0.001:
        n_wrong += 1
        print(f"  {FAIL} Row {i}: expected={expected}  actual={actual}")

check(n_wrong == 0, f"GapScore formula is correct for all checked rows ({n_wrong} mismatches)")

# Check value ranges
check(gaps["semantic_similarity"].between(0, 1).all(), "All semantic_similarity in [0,1]")
check(gaps["cooccurrence_score"].between(0, 1).all(), "All cooccurrence_score in [0,1]")
check((gaps["gap_score"] > 0).all(), "All gap_scores > 0")

# Verify synonym flagging
if "gap_note" in gaps.columns:
    syn_count = (gaps["gap_note"] == "potential_synonym_pair").sum()
    genuine_count = len(gaps) - syn_count
    check(syn_count <= 30, f"Synonym-flagged pairs <= 30 (got {syn_count})")
    check(genuine_count >= 20, f"Genuine candidate gaps >= 20 (got {genuine_count})")
    print(f"\n  Genuine gaps: {genuine_count}  |  Synonym-flagged: {syn_count}")
else:
    issues.append("gap_note column missing from candidate_research_gaps.csv")
    print(f"  {FAIL} gap_note column missing")

# Check supporting PMID traceability
check("supporting_pmids_a" in gaps.columns, "supporting_pmids_a column present")
check("supporting_pmids_b" in gaps.columns, "supporting_pmids_b column present")

# Check gap_quality_audit.csv
gqa_file = OUT / "gap_quality_audit.csv"
check(gqa_file.exists(), "outputs/gap_quality_audit.csv exists")
if gqa_file.exists():
    gqa = pd.read_csv(gqa_file)
    check(len(gqa) == 50, f"gap_quality_audit has 50 rows (got {len(gqa)})")
    check("quality_status" in gqa.columns, "gap_quality_audit has 'quality_status' column")
    check("has_supporting_evidence" in gqa.columns, "gap_quality_audit has 'has_supporting_evidence' column")
    substantive_cnt = (gqa["quality_status"] == "genuine_substantive_gap").sum()
    check(substantive_cnt >= 20, f"Substantive candidate gaps >= 20 (got {substantive_cnt})")

# ============================================================
# BLOCK 8 — CO-OCCURRENCE INTEGRITY
# ============================================================
section("BLOCK 8 — CO-OCCURRENCE INTEGRITY")

cooc = pd.read_csv(OUT / "concept_cooccurrence.csv")
check("concept_a" in cooc.columns, "Co-occurrence has concept_a column")
check("concept_b" in cooc.columns, "Co-occurrence has concept_b column")
check("normalized_cooccurrence" in cooc.columns or "cooccurrence_count" in cooc.columns,
      "Co-occurrence has normalized or count column")

if "normalized_cooccurrence" in cooc.columns:
    check(cooc["normalized_cooccurrence"].between(0, 1).all(),
          "All normalized_cooccurrence values in [0,1]")
    check((cooc["normalized_cooccurrence"] > 0).all(),
          "All normalized_cooccurrence > 0 (filtered to joint>=2)")
    
print(f"\n  Co-occurrence pairs: {len(cooc):,}")

# ============================================================
# BLOCK 9 — EVIDENCE TRACEABILITY
# ============================================================
section("BLOCK 9 — EVIDENCE TRACEABILITY")

ev = pd.read_csv(OUT / "gap_evidence.csv")
check(len(ev) == 50, f"Evidence has 50 rows (got {len(ev)})")
check("evidence_description" in ev.columns, "evidence_description column present")

# Every genuine gap should have non-empty evidence
genuine_ev = ev.head(26)  # first 26 are genuine (before synonym rows)
empty_ev = (genuine_ev["evidence_description"].fillna("").str.strip() == "").sum()
check(empty_ev == 0, f"All genuine gaps have non-empty evidence ({empty_ev} empty)")

# ============================================================
# BLOCK 10 — OUTPUT FILE CHECKLIST
# ============================================================
section("BLOCK 10 — OUTPUT FILE CHECKLIST")

required_outputs = [
    "data_validation_report.csv",
    "data_validation_summary.json",
    "cleaning_filter_log.csv",
    "mhealth_candidate_cleaned.csv",
    "screening_statistics.csv",
    "mhealth_screened_corpus.csv",
    "mhealth_preprocessed.csv",
    "document_embeddings.npy",
    "document_embedding_index.csv",
    "embedding_model_info.txt",
    "topic_assignments.csv",
    "topic_summary.csv",
    "topic_top_terms.csv",
    "topic_year_matrix.csv",
    "concept_year_matrix.csv",
    "concept_trends.csv",
    "document_concepts.csv",
    "concept_summary.csv",
    "mesh_normalized.csv",
    "topic_concept_mapping.csv",
    "concept_summary_filtered.csv",
    "concept_quality_filter_report.json",
    "semantic_similarity_pairs.csv",
    "semantic_similarity_pairs_debiased.csv",
    "concept_cooccurrence.csv",
    "candidate_research_gaps.csv",
    "gap_evidence.csv",
    "gap_quality_audit.csv",
    "phase1_summary.csv",
    "phase1_summary.json",
]

required_figures = [
    "papers_by_year.png",
    "topic_frequency.png",
    "topic_distribution.png",
    "topic_trends.png",
    "top_concepts.png",
    "concept_trends.png",
    "concept_cooccurrence_heatmap.png",
    "semantic_relationships.png",
    "candidate_gap_ranking.png",
]

for fname in required_outputs:
    p = OUT / fname
    check(p.exists() and p.stat().st_size > 0, f"outputs/{fname}")

fig_dir = OUT.parent / "figures"
for fname in required_figures:
    p = fig_dir / fname
    check(p.exists() and p.stat().st_size > 0, f"figures/{fname}")

# ============================================================
# SUMMARY
# ============================================================
section("AUDIT SUMMARY")
total_checks = len(issues) + (30 - len(issues))  # rough estimate
if issues:
    print(f"\n  Issues found: {len(issues)}")
    for iss in issues:
        print(f"    {FAIL} {iss}")
else:
    print(f"\n  ALL CHECKS PASSED — No issues found.")

print("\n  Issue count:", len(issues))
