"""
streamlit_app.py — Interactive Research Assistant Dashboard

Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation in mHealth Research

Launch: streamlit run streamlit_app.py
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="mHealth Research Gap Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HELPERS
# ============================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"


@st.cache_data
def load_json(path):
    """Load a JSON file with caching."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return None


@st.cache_data
def load_csv(path, **kwargs):
    """Load a CSV file with caching."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def load_phase1_data():
    """Load Phase 1 artifacts."""
    data = {}
    data["gaps"] = load_csv(OUTPUTS_DIR / "candidate_research_gaps.csv")
    data["gap_evidence"] = load_csv(OUTPUTS_DIR / "gap_evidence.csv")
    data["gap_audit"] = load_csv(OUTPUTS_DIR / "gap_quality_audit.csv")
    data["topic_summary"] = load_csv(OUTPUTS_DIR / "topic_summary.csv")
    data["concept_summary"] = load_csv(OUTPUTS_DIR / "concept_summary.csv")
    data["phase1_summary"] = load_json(OUTPUTS_DIR / "phase1_summary.json")
    data["topic_assignments"] = load_csv(OUTPUTS_DIR / "topic_assignments.csv")
    return data


def load_phase2_data():
    """Load Phase 2 outputs if available."""
    data = {}
    data["prepared_gaps"] = load_json(OUTPUTS_DIR / "phase2_prepared_gaps.json")
    data["retrieved_evidence"] = load_json(OUTPUTS_DIR / "retrieved_gap_evidence.json")
    data["hypotheses"] = load_json(OUTPUTS_DIR / "generated_hypotheses.json")
    data["verification"] = load_json(OUTPUTS_DIR / "verification_results.json")
    data["nli_results"] = load_json(OUTPUTS_DIR / "nli_results.json")
    data["judge_results"] = load_json(OUTPUTS_DIR / "judge_results.json")
    data["rule_results"] = load_json(OUTPUTS_DIR / "rule_check_results.json")
    data["rankings"] = load_json(OUTPUTS_DIR / "hypothesis_rankings.json")
    data["dedup_results"] = load_json(OUTPUTS_DIR / "deduplication_results.json")
    data["phase2_summary"] = load_json(OUTPUTS_DIR / "phase2_summary.json")
    data["final_csv"] = load_csv(OUTPUTS_DIR / "final_hypotheses.csv")
    return data


def safe_get(d, key, default=""):
    """Safely get from dict or object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🔬 mHealth Research")
st.sidebar.caption("Gap-Guided Hypothesis Generation")

pages = [
    "📋 Project Overview",
    "🔍 Research Gap Explorer",
    "📄 Evidence Explorer",
    "⚡ Hypothesis Generation",
    "✅ Hypothesis Validation",
    "🏆 Ranked Results",
    "📥 Export",
]

page = st.sidebar.radio("Navigate", pages, label_visibility="collapsed")

# AI disclaimer
st.sidebar.divider()
st.sidebar.warning(
    "⚠️ **Important**: AI-generated hypotheses are computational "
    "suggestions requiring expert review and empirical testing. "
    "They are NOT validated scientific claims."
)

# Pipeline status
p2_summary = load_json(OUTPUTS_DIR / "phase2_summary.json")
if p2_summary:
    st.sidebar.divider()
    st.sidebar.subheader("Pipeline Status")
    st.sidebar.success(f"✅ Phase 2 Complete")
    st.sidebar.metric("Hypotheses Generated", p2_summary.get("hypotheses_generated", 0))
    st.sidebar.metric("Gaps Processed", p2_summary.get("gaps_processed", 0))
    if p2_summary.get("mock_mode"):
        st.sidebar.info("🔧 Mock Mode (demo data)")
else:
    st.sidebar.divider()
    st.sidebar.info("ℹ️ Run Phase 2 pipeline first:\n`python run_phase2.py --mock`")


# ============================================================
# PAGE: PROJECT OVERVIEW
# ============================================================

if page == "📋 Project Overview":
    st.title("🧬 Temporal-Semantic Gap-Guided Evidence-Grounded Hypothesis Generation")
    st.subheader("in mHealth Research")

    st.markdown("""
    ### Project Objective
    Develop an AI-powered research assistant that analyzes mHealth literature, identifies
    potentially underexplored research relationships, retrieves supporting scientific evidence,
    generates testable research hypotheses using an LLM, validates them against the retrieved
    evidence, and presents the results through this interactive application.
    """)

    col1, col2, col3 = st.columns(3)

    p1 = load_json(OUTPUTS_DIR / "phase1_summary.json")
    if p1:
        with col1:
            st.metric("📚 PubMed Papers", f"{p1.get('screened_records', 0):,}")
            st.metric("🏷️ Topics Discovered", p1.get("num_topics", 0))
        with col2:
            st.metric("🧪 Biomedical Concepts", f"{p1.get('num_concepts_quality_filtered', 0):,}")
            st.metric("📊 Candidate Gaps", p1.get("num_candidate_gaps_genuine", 0))
        with col3:
            if p2_summary:
                st.metric("💡 Hypotheses Generated", p2_summary.get("hypotheses_generated", 0))
                st.metric("✅ Validated", p2_summary.get("hypotheses_verified", 0))
            else:
                st.metric("💡 Hypotheses", "—")
                st.metric("✅ Validated", "—")

    st.markdown("### Pipeline Architecture")
    st.markdown("""
    ```
    Phase 1 (Completed)              Phase 2 (Evidence-Grounded Generation)
    ═══════════════════              ══════════════════════════════════════
    PubMed Collection                Gap Preparation & Prioritization
         ↓                                      ↓
    Data Validation                  Document Chunking for RAG
         ↓                                      ↓
    Relevance Screening              FAISS Evidence Retrieval
         ↓                                      ↓
    SPECTER2 Embeddings              LLM Hypothesis Generation (Gemini)
         ↓                                      ↓
    BERTopic Topics                  Evidence Verification
         ↓                                      ↓
    Concept Extraction               NLI Validation
         ↓                                      ↓
    MeSH Normalization               LLM-as-a-Judge
         ↓                                      ↓
    Temporal Analysis                Rule-Based Checks
         ↓                                      ↓
    Gap Detection ──────────────→    Ranking & Deduplication
         ↓                                      ↓
    50 Candidate Gaps                Final Hypotheses + Dashboard
    ```
    """)

    st.markdown("### Data Summary")
    st.markdown("""
    | Aspect | Detail |
    |--------|--------|
    | **Corpus** | PubMed abstracts (2015–2025), NOT full-text papers |
    | **Records** | 12,642 screened mHealth papers |
    | **Embeddings** | SPECTER2 (768-dim) for documents; MiniLM for RAG chunks |
    | **Gap Formula** | GapScore = Semantic Similarity × (1 - CoOccurrence) × (1 + TemporalGrowth) |
    | **LLM** | Google Gemini for generation and evaluation |
    """)

    # Show Phase 1 figures
    st.markdown("### Phase 1 Visualizations")
    fig_cols = st.columns(3)
    figures = [
        ("papers_by_year.png", "Papers by Year"),
        ("topic_frequency.png", "Topic Frequency"),
        ("candidate_gap_ranking.png", "Gap Ranking"),
    ]
    for i, (fname, title) in enumerate(figures):
        fpath = FIGURES_DIR / fname
        if fpath.exists():
            with fig_cols[i]:
                st.image(str(fpath), caption=title, use_container_width=True)

    # Show Phase 3 Benchmark Evaluation Table
    eval_csv_path = OUTPUTS_DIR / "phase3_evaluation_summary.csv"
    if eval_csv_path.exists():
        st.markdown("### 📊 Phase 3 Empirical Benchmark Performance")
        eval_df = load_csv(eval_csv_path)
        if eval_df is not None:
            st.dataframe(eval_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: RESEARCH GAP EXPLORER
# ============================================================

elif page == "🔍 Research Gap Explorer":
    st.title("🔍 Research Gap Explorer")
    st.caption("Browse and filter candidate research gaps identified in Phase 1")

    p1_data = load_phase1_data()
    gaps = p1_data["gaps"]
    audit = p1_data["gap_audit"]

    if gaps is None:
        st.error("Gap data not found. Run Phase 1 pipeline first.")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        gap_type = st.selectbox("Gap Type", ["All", "Genuine Substantive", "Synonym/Variant"])
    with col2:
        min_score = st.slider("Min Gap Score", 0.0, 2.0, 0.5, 0.05)
    with col3:
        search_term = st.text_input("Search concepts", "")

    # Apply filters
    display_gaps = gaps.copy()
    if audit is not None:
        display_gaps = display_gaps.merge(
            audit[["concept_a", "concept_b", "quality_status"]],
            on=["concept_a", "concept_b"], how="left"
        )
    else:
        display_gaps["quality_status"] = "unknown"

    if gap_type == "Genuine Substantive":
        display_gaps = display_gaps[display_gaps["quality_status"] == "genuine_substantive_gap"]
    elif gap_type == "Synonym/Variant":
        display_gaps = display_gaps[display_gaps["quality_status"] == "synonym_or_abbreviation_variant"]

    display_gaps = display_gaps[display_gaps["gap_score"] >= min_score]

    if search_term:
        mask = (display_gaps["concept_a"].str.contains(search_term, case=False, na=False) |
                display_gaps["concept_b"].str.contains(search_term, case=False, na=False))
        display_gaps = display_gaps[mask]

    st.subheader(f"Showing {len(display_gaps)} gaps")

    for idx, (_, gap) in enumerate(display_gaps.iterrows()):
        is_genuine = gap.get("quality_status", "") == "genuine_substantive_gap"
        icon = "✅" if is_genuine else "⚠️"
        label = "Genuine Gap" if is_genuine else "Synonym/Variant"

        with st.expander(
            f"{icon} [{gap['concept_a']}] ↔ [{gap['concept_b']}] — "
            f"Score: {gap['gap_score']:.3f} ({label})",
            expanded=(idx < 3)
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gap Score", f"{gap['gap_score']:.3f}")
            c2.metric("Semantic Similarity", f"{gap['semantic_similarity']:.3f}")
            c3.metric("Co-occurrence", f"{gap['cooccurrence_score']:.3f}")
            c4.metric("Temporal Growth", f"{gap['temporal_growth_avg']:.3f}")

            c5, c6, c7 = st.columns(3)
            c5.metric(f"Papers ({gap['concept_a'][:20]}...)", gap['papers_concept_a'])
            c6.metric(f"Papers ({gap['concept_b'][:20]}...)", gap['papers_concept_b'])
            c7.metric("Joint Papers", gap['papers_joint'])

            # Topic IDs
            topics = str(gap.get("supporting_topic_ids", ""))
            if topics and topics != "nan":
                st.markdown(f"**Supporting Topics**: {topics}")

            # Gap evidence description
            if p1_data["gap_evidence"] is not None:
                ev_row = p1_data["gap_evidence"][
                    (p1_data["gap_evidence"]["concept_a"] == gap["concept_a"]) &
                    (p1_data["gap_evidence"]["concept_b"] == gap["concept_b"])
                ]
                if len(ev_row) > 0:
                    st.info(ev_row.iloc[0].get("evidence_description", ""))


# ============================================================
# PAGE: EVIDENCE EXPLORER
# ============================================================

elif page == "📄 Evidence Explorer":
    st.title("📄 Evidence Explorer")
    st.caption("View retrieved evidence passages for each research gap")

    p2_data = load_phase2_data()
    retrieved = p2_data.get("retrieved_evidence")

    if not retrieved:
        st.warning("No retrieved evidence found. Run Phase 2 pipeline first.")
        st.info("```\npython run_phase2.py --mock\n```")
        st.stop()

    # Gap selector
    gap_ids = list(retrieved.keys())
    selected_gap = st.selectbox("Select Research Gap", gap_ids)

    if selected_gap:
        evidence_list = retrieved[selected_gap]
        st.subheader(f"Evidence for {selected_gap}: {len(evidence_list)} passages")

        for i, ev in enumerate(evidence_list):
            score = safe_get(ev, "relevance_score", 0)
            title = safe_get(ev, "paper_title", "Unknown")
            pmid = safe_get(ev, "source_pmid", "")
            year = safe_get(ev, "publication_year", "")
            content_type = safe_get(ev, "content_type", "abstract")
            text = safe_get(ev, "text_content", "")

            with st.expander(
                f"📄 [{i+1}] {title[:80]}... (Score: {score:.3f}, {year})",
                expanded=(i < 3)
            ):
                cols = st.columns([3, 1, 1, 1])
                cols[0].write(f"**Paper**: {title}")
                cols[1].write(f"**PMID**: {pmid}")
                cols[2].write(f"**Year**: {year}")
                cols[3].write(f"**Type**: {content_type}")

                if content_type == "abstract":
                    st.caption("📝 Source: Paper abstract (not full text)")

                st.markdown(f"> {text[:500]}..." if len(text) > 500 else f"> {text}")

                doi = safe_get(ev, "doi", "")
                if doi and str(doi) != "nan":
                    st.markdown(f"[View Paper (DOI)](https://doi.org/{doi})")
                elif pmid:
                    st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")


# ============================================================
# PAGE: HYPOTHESIS GENERATION
# ============================================================

elif page == "⚡ Hypothesis Generation":
    st.title("⚡ Hypothesis Generation")
    st.caption("Generate evidence-grounded research hypotheses from identified gaps")

    p1_data = load_phase1_data()
    p2_data = load_phase2_data()

    # Show existing hypotheses if available
    if p2_data["hypotheses"]:
        st.success(f"✅ {len(p2_data['hypotheses'])} hypotheses already generated")

        # Group by gap
        gap_groups = {}
        for h in p2_data["hypotheses"]:
            gid = safe_get(h, "gap_id", "unknown")
            gap_groups.setdefault(gid, []).append(h)

        for gap_id, hyps in gap_groups.items():
            st.subheader(f"📌 {gap_id}")
            for h in hyps:
                hyp_id = safe_get(h, "hypothesis_id", "")
                statement = safe_get(h, "hypothesis", "No hypothesis text")

                with st.expander(f"💡 {hyp_id}: {statement[:100]}..."):
                    st.markdown(f"**Hypothesis**: {statement}")
                    st.markdown(f"**Research Question**: {safe_get(h, 'research_question', 'N/A')}")

                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**Independent Variable**: {safe_get(h, 'independent_variable', 'N/A')}")
                        st.markdown(f"**Dependent Variable**: {safe_get(h, 'dependent_variable', 'N/A')}")
                        st.markdown(f"**Target Population**: {safe_get(h, 'target_population', 'N/A')}")
                    with cols[1]:
                        st.markdown(f"**Proposed Relationship**: {safe_get(h, 'proposed_relationship', 'N/A')}")
                        st.markdown(f"**Intervention/Exposure**: {safe_get(h, 'intervention_or_exposure', 'N/A')}")
                        st.markdown(f"**Contribution**: {safe_get(h, 'potential_contribution', 'N/A')}")

                    # Evidence
                    supp = safe_get(h, "supporting_evidence_ids", [])
                    contra = safe_get(h, "contradictory_evidence_ids", [])
                    if supp:
                        st.markdown(f"**Supporting Evidence**: {len(supp)} passages")
                    if contra:
                        st.warning(f"**Contradictory Evidence**: {len(contra)} passages")

                    # Uncertainties and limitations
                    uncertainties = safe_get(h, "uncertainties", [])
                    limitations = safe_get(h, "limitations", [])
                    if uncertainties:
                        st.markdown("**Uncertainties**:")
                        for u in uncertainties:
                            st.markdown(f"  - {u}")
                    if limitations:
                        st.markdown("**Limitations**:")
                        for l in limitations:
                            st.markdown(f"  - {l}")

                    st.caption(f"Model: {safe_get(h, 'generation_model', 'N/A')} | "
                             f"Generated: {safe_get(h, 'generation_timestamp', 'N/A')}")
    # Interactive On-Demand Generation Section
    st.divider()
    st.subheader("🧪 Generate New Hypotheses On-Demand")
    st.caption("Select any candidate research gap and synthesize evidence-grounded hypotheses interactively.")

    gaps = p1_data.get("gaps")
    if gaps is not None and len(gaps) > 0:
        gap_options = [
            f"{row['concept_a']} ↔ {row['concept_b']} (Score: {row['gap_score']:.3f})"
            for _, row in gaps.head(30).iterrows()
        ]
        selected_idx = st.selectbox("Select Research Gap to Bridge", range(len(gap_options)),
                                    format_func=lambda i: gap_options[i], key="gen_gap_select")
        selected_row = gaps.iloc[selected_idx].to_dict()

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            use_mock = st.checkbox("Use mock generator (fast offline demo)", value=True, key="use_mock_box")
        with col_g2:
            num_hyp = st.slider("Number of Hypotheses", 1, 3, 2, key="num_hyp_slider")

        if st.button("🚀 Synthesize Hypotheses Now", type="primary", key="btn_gen_now"):
            with st.spinner("Retrieving grounded evidence from FAISS index and generating hypotheses..."):
                try:
                    from src.phase2.retrieval import EvidenceRetriever
                    from src.phase2.hypothesis_generation import HypothesisGenerator

                    index_dir = OUTPUTS_DIR / "faiss_index"
                    retriever = EvidenceRetriever(index_path=index_dir) if index_dir.exists() else None
                    evidence = retriever.retrieve_for_gap(selected_row, top_k=10) if retriever else []

                    generator = HypothesisGenerator(use_mock=use_mock)
                    live_hyps = generator.generate_hypotheses(selected_row, evidence, max_hypotheses=num_hyp)

                    st.success(f"Successfully generated {len(live_hyps)} hypotheses for [{selected_row.get('concept_a')}] ↔ [{selected_row.get('concept_b')}]!")

                    for nh_idx, nh in enumerate(live_hyps, 1):
                        nh_dict = nh.to_dict() if hasattr(nh, "to_dict") else nh
                        with st.expander(f"✨ New Hypothesis #{nh_idx}: {safe_get(nh_dict, 'hypothesis')[:80]}...", expanded=True):
                            st.markdown(f"**Hypothesis**: {safe_get(nh_dict, 'hypothesis')}")
                            st.markdown(f"**Research Question**: {safe_get(nh_dict, 'research_question')}")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"- **Independent Variable**: {safe_get(nh_dict, 'independent_variable')}")
                                st.markdown(f"- **Dependent Variable**: {safe_get(nh_dict, 'dependent_variable')}")
                                st.markdown(f"- **Target Population**: {safe_get(nh_dict, 'target_population')}")
                            with c2:
                                st.markdown(f"- **Relationship**: {safe_get(nh_dict, 'proposed_relationship')}")
                                st.markdown(f"- **Intervention**: {safe_get(nh_dict, 'intervention_or_exposure')}")
                                st.markdown(f"- **Cited Evidence Chunks**: `{safe_get(nh_dict, 'supporting_evidence_ids')}`")
                            st.markdown(f"**Testability**: {safe_get(nh_dict, 'testability_explanation')}")
                except Exception as ex:
                    st.error(f"Generation error: {ex}")
                    logger.error(f"Interactive generation error: {ex}", exc_info=True)


# ============================================================
# PAGE: HYPOTHESIS VALIDATION
# ============================================================

elif page == "✅ Hypothesis Validation":
    st.title("✅ Hypothesis Validation")
    st.caption("View evidence verification, NLI, LLM-judge, and rule-based validation results")

    p2_data = load_phase2_data()

    if not p2_data["hypotheses"]:
        st.warning("No hypotheses to validate. Run Phase 2 pipeline first.")
        st.stop()

    # Select hypothesis
    hyp_options = [f"{safe_get(h, 'hypothesis_id', 'N/A')}: {safe_get(h, 'hypothesis', '')[:60]}..."
                   for h in p2_data["hypotheses"]]
    selected_idx = st.selectbox("Select Hypothesis", range(len(hyp_options)),
                                format_func=lambda i: hyp_options[i])

    hyp = p2_data["hypotheses"][selected_idx]
    hyp_id = safe_get(hyp, "hypothesis_id", "")

    st.subheader(f"Validation Results: {hyp_id}")

    tabs = st.tabs(["📋 Evidence Check", "🧪 NLI", "⚖️ LLM Judge", "📝 Rule Checks"])

    # Tab 1: Evidence Verification
    with tabs[0]:
        verif = (p2_data.get("verification") or {}).get(hyp_id, {})
        if verif:
            status = safe_get(verif, "verification_status", "unknown")
            icon = "✅" if status == "passed" else "⚠️" if status == "partial" else "❌"
            st.markdown(f"### {icon} Verification Status: **{status}**")

            cols = st.columns(3)
            cols[0].metric("Valid References", len(safe_get(verif, "valid_ids", [])))
            cols[1].metric("Invalid References", len(safe_get(verif, "invalid_ids", [])))
            cols[2].metric("Coverage Ratio", f"{safe_get(verif, 'coverage_ratio', 0):.2%}")

            if safe_get(verif, "invalid_ids", []):
                st.error(f"Invalid evidence IDs: {safe_get(verif, 'invalid_ids', [])}")
            if safe_get(verif, "fabricated_count", 0) > 0:
                st.error(f"⚠️ {safe_get(verif, 'fabricated_count', 0)} potentially fabricated references detected!")
        else:
            st.info("Evidence verification not run for this hypothesis")

    # Tab 2: NLI
    with tabs[1]:
        nli = (p2_data.get("nli_results") or {}).get(hyp_id, {})
        if nli:
            alignment = safe_get(nli, "overall_alignment", "unknown")
            score = safe_get(nli, "alignment_score", 0)

            color = "🟢" if alignment == "supported" else "🔴" if alignment == "contradicted" else "🟡"
            st.markdown(f"### {color} NLI Alignment: **{alignment}** (score: {score:.3f})")

            cols = st.columns(3)
            cols[0].metric("Support", safe_get(nli, "support_count", 0))
            cols[1].metric("Contradiction", safe_get(nli, "contradiction_count", 0))
            cols[2].metric("Neutral", safe_get(nli, "neutral_count", 0))

            passage_results = safe_get(nli, "passage_results", [])
            if passage_results:
                st.markdown("**Passage-level results:**")
                for pr in passage_results:
                    label = safe_get(pr, "label", "unknown")
                    eid = safe_get(pr, "evidence_id", "")
                    icon = "✅" if label == "entailment" else "❌" if label == "contradiction" else "➖"
                    st.markdown(f"  {icon} `{eid}`: {label}")

            st.caption(f"Model: {safe_get(nli, 'model_used', 'N/A')}")
            st.caption("⚠️ NLI results are automated alignment signals, not proof of scientific truth.")
        else:
            st.info("NLI validation not available for this hypothesis")

    # Tab 3: LLM Judge
    with tabs[2]:
        judge = (p2_data.get("judge_results") or {}).get(hyp_id, {})
        if judge:
            st.markdown("### LLM Quality Assessment")
            st.caption("⚠️ These are automated LLM assessments, not expert evaluations.")

            dims = [
                ("relevance_to_gap", "Relevance to Gap"),
                ("clarity", "Clarity"),
                ("logical_consistency", "Logical Consistency"),
                ("testability", "Testability"),
                ("evidence_alignment", "Evidence Alignment"),
                ("novelty_potential", "Novelty Potential"),
            ]

            cols = st.columns(3)
            for i, (key, label) in enumerate(dims):
                score = safe_get(judge, key, 0)
                with cols[i % 3]:
                    st.metric(label, f"{score}/5")

            # Justifications
            st.markdown("**Justifications:**")
            just_keys = [
                ("relevance_justification", "Relevance"),
                ("clarity_justification", "Clarity"),
                ("consistency_justification", "Consistency"),
                ("testability_justification", "Testability"),
                ("evidence_justification", "Evidence"),
                ("novelty_justification", "Novelty"),
            ]
            for key, label in just_keys:
                just = safe_get(judge, key, "")
                if just:
                    st.markdown(f"  - **{label}**: {just}")

            overall = safe_get(judge, "overall_assessment", "")
            if overall:
                st.info(f"**Overall**: {overall}")
        else:
            st.info("LLM judge evaluation not available for this hypothesis")

    # Tab 4: Rule Checks
    with tabs[3]:
        rules = (p2_data.get("rule_results") or {}).get(hyp_id, {})
        if rules:
            status = safe_get(rules, "overall_status", "unknown")
            icon = "✅" if status == "pass" else "⚠️" if status == "warn" else "❌"
            st.markdown(f"### {icon} Rule Check Status: **{status}**")

            checks = safe_get(rules, "checks", [])
            passed = sum(1 for c in checks if safe_get(c, "status") == "pass")
            failed = sum(1 for c in checks if safe_get(c, "status") == "fail")
            warned = sum(1 for c in checks if safe_get(c, "status") == "warning")

            cols = st.columns(3)
            cols[0].metric("Passed", passed)
            cols[1].metric("Failed", failed)
            cols[2].metric("Warnings", warned)

            for check in checks:
                s = safe_get(check, "status", "unknown")
                icon = "✅" if s == "pass" else "❌" if s == "fail" else "⚠️"
                st.markdown(f"  {icon} **{safe_get(check, 'check_name', '')}**: "
                          f"{safe_get(check, 'message', '')}")
        else:
            st.info("Rule checks not available for this hypothesis")


# ============================================================
# PAGE: RANKED RESULTS
# ============================================================

elif page == "🏆 Ranked Results":
    st.title("🏆 Ranked Hypothesis Results")
    st.caption("Final ranked and deduplicated hypotheses with full traceability")

    p2_data = load_phase2_data()
    hypotheses = p2_data.get("hypotheses", [])
    rankings = p2_data.get("rankings", [])
    final_csv = p2_data.get("final_csv")

    if not hypotheses:
        st.warning("No hypotheses available. Run Phase 2 pipeline first.")
        st.stop()

    # Build lookup
    hyp_lookup = {safe_get(h, "hypothesis_id"): h for h in hypotheses}
    rank_lookup = {safe_get(r, "hypothesis_id"): r for r in (rankings or [])}

    # Sort by rank
    sorted_hyps = sorted(hypotheses,
                         key=lambda h: safe_get(rank_lookup.get(safe_get(h, "hypothesis_id"), {}),
                                                "rank_score", 0),
                         reverse=True)

    # Summary metrics
    st.subheader("Summary")
    cols = st.columns(4)
    cols[0].metric("Total Hypotheses", len(hypotheses))

    dedup = p2_data.get("dedup_results")
    if dedup:
        cols[1].metric("After Deduplication", safe_get(dedup, "total_output", len(hypotheses)))
    else:
        cols[1].metric("After Deduplication", len(hypotheses))

    if rankings:
        high = sum(1 for r in rankings if safe_get(r, "quality_tier") == "high")
        cols[2].metric("High Quality", high)
        avg_score = np.mean([safe_get(r, "rank_score", 0) for r in rankings])
        cols[3].metric("Avg Rank Score", f"{avg_score:.3f}")

    # Display ranked hypotheses
    st.divider()
    for i, hyp in enumerate(sorted_hyps):
        hyp_id = safe_get(hyp, "hypothesis_id", f"H-{i}")
        rank = rank_lookup.get(hyp_id, {})
        tier = safe_get(rank, "quality_tier", "unranked")
        score = safe_get(rank, "rank_score", 0)
        position = safe_get(rank, "rank_position", i + 1)

        tier_icon = "🥇" if tier == "high" else "🥈" if tier == "medium" else "🥉"
        statement = safe_get(hyp, "hypothesis", "No text")

        with st.expander(
            f"{tier_icon} #{position} | {hyp_id} | Score: {score:.3f} | {statement[:80]}...",
            expanded=(i < 3)
        ):
            st.markdown(f"**Hypothesis**: {statement}")
            st.markdown(f"**Research Question**: {safe_get(hyp, 'research_question', 'N/A')}")

            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Variables**")
                st.markdown(f"- IV: {safe_get(hyp, 'independent_variable', 'N/A')}")
                st.markdown(f"- DV: {safe_get(hyp, 'dependent_variable', 'N/A')}")
            with cols[1]:
                st.markdown("**Context**")
                st.markdown(f"- Population: {safe_get(hyp, 'target_population', 'N/A')}")
                st.markdown(f"- Relationship: {safe_get(hyp, 'proposed_relationship', 'N/A')}")
            with cols[2]:
                st.markdown("**Quality Dimensions**")
                st.markdown(f"- Evidence Support: {safe_get(rank, 'evidence_support_score', 'N/A')}")
                st.markdown(f"- Judge Quality: {safe_get(rank, 'judge_quality_score', 'N/A')}")
                st.markdown(f"- Contradiction Penalty: {safe_get(rank, 'contradiction_penalty', 'N/A')}")

            # Traceability
            st.markdown("---")
            st.markdown("**📍 Traceability Path**")
            gap_id = safe_get(hyp, "gap_id", "")
            st.markdown(f"Hypothesis `{hyp_id}` → Gap `{gap_id}` → Evidence passages: "
                       f"{safe_get(hyp, 'supporting_evidence_ids', [])}")

            # Warnings
            warnings = safe_get(rank, "validation_warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(w)


# ============================================================
# PAGE: EXPORT
# ============================================================

elif page == "📥 Export":
    st.title("📥 Export Results")
    st.caption("Download hypotheses, evidence, and validation results")

    p2_data = load_phase2_data()

    st.subheader("Available Exports")

    # CSV exports
    final_csv_path = OUTPUTS_DIR / "final_hypotheses.csv"
    if final_csv_path.exists():
        with open(final_csv_path, "r", encoding="utf-8-sig") as f:
            csv_data = f.read()
        st.download_button(
            "📥 Download Final Hypotheses (CSV)",
            csv_data,
            file_name="mhealth_hypotheses.csv",
            mime="text/csv",
            type="primary",
        )

    eval_csv_path = OUTPUTS_DIR / "phase3_evaluation_summary.csv"
    if eval_csv_path.exists():
        with open(eval_csv_path, "r", encoding="utf-8-sig") as f:
            eval_csv_data = f.read()
        st.download_button(
            "📊 Download Phase 3 Evaluation Metrics (CSV)",
            eval_csv_data,
            file_name="phase3_evaluation_summary.csv",
            mime="text/csv",
            key="eval_csv_dl",
        )

    # JSON exports
    json_files = [
        ("generated_hypotheses.json", "Generated Hypotheses"),
        ("retrieved_gap_evidence.json", "Retrieved Evidence"),
        ("verification_results.json", "Verification Results"),
        ("nli_results.json", "NLI Results"),
        ("judge_results.json", "LLM Judge Results"),
        ("hypothesis_rankings.json", "Rankings"),
        ("phase2_summary.json", "Pipeline Summary"),
        ("phase3_evaluation_results.json", "Phase 3 Evaluation Benchmarks"),
    ]

    for fname, label in json_files:
        fpath = OUTPUTS_DIR / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                data = f.read()
            st.download_button(
                f"📥 {label} (JSON)",
                data,
                file_name=fname,
                mime="application/json",
                key=fname,
            )
        else:
            st.caption(f"  ⏳ {label} — not yet generated")

    st.divider()
    st.warning(
        "⚠️ Exported results are AI-generated research candidates. "
        "They require expert review and empirical validation before "
        "use in any scientific publication or clinical application."
    )
