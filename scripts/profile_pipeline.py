"""
scripts/profile_pipeline.py — Comprehensive Profiling Benchmark for Hypothesis Pipeline.
Measures cold-start vs warm-request latencies, stage-by-stage timings, GPU/CPU utilization, and memory.
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure dotenv is loaded
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

import torch

def get_mem():
    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return ram_mb, vram_mb

print("=" * 80)
print("mHealth HYPOTHESIS PIPELINE: DETAILED PERFORMANCE PROFILER")
print("=" * 80)
print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
ram_init, vram_init = get_mem()
print(f"Initial Memory: RAM = {ram_init:.1f} MB, VRAM = {vram_init:.1f} MB")
print("-" * 80)

timings = {}

# ---------------------------------------------------------------------------
# 1. COLD-START COMPONENT INITIALIZATION
# ---------------------------------------------------------------------------
print("\n[STAGE 1: COLD-START INITIALIZATION]")

t0 = time.perf_counter()
from backend.services.theme_matcher import ThemeMatcher
theme_matcher = ThemeMatcher()
timings["init_theme_matcher"] = time.perf_counter() - t0
print(f"  - ThemeMatcher data loading: {timings['init_theme_matcher']*1000:.2f} ms")

t0 = time.perf_counter()
from src.phase2.retrieval import EvidenceRetriever
index_path = PROJECT_ROOT / "outputs" / "faiss_index"
retriever = EvidenceRetriever(index_path=index_path)
timings["init_faiss_load"] = time.perf_counter() - t0
print(f"  - FAISS index & metadata load: {timings['init_faiss_load']*1000:.2f} ms ({retriever.index.ntotal} vectors)")

t0 = time.perf_counter()
st_model = retriever.model
timings["init_sentence_transformer"] = time.perf_counter() - t0
ram_st, vram_st = get_mem()
print(f"  - SentenceTransformer model load ({retriever.embedding_model_name}): {timings['init_sentence_transformer']*1000:.2f} ms (RAM: {ram_st:.1f} MB, VRAM: {vram_st:.1f} MB)")

t0 = time.perf_counter()
from src.phase2.hypothesis_generation import HypothesisGenerator
generator = HypothesisGenerator(use_mock=False)
timings["init_hypothesis_generator"] = time.perf_counter() - t0
print(f"  - HypothesisGenerator init ({generator.model_name}): {timings['init_hypothesis_generator']*1000:.2f} ms")

t0 = time.perf_counter()
from src.phase2.llm_judge import LLMJudge
judge = LLMJudge(use_mock=False)
timings["init_llm_judge"] = time.perf_counter() - t0
print(f"  - LLMJudge init ({judge.model_name}): {timings['init_llm_judge']*1000:.2f} ms")

t0 = time.perf_counter()
from src.phase2.nli_validation import NLIValidator
nli_validator = NLIValidator(device="cuda" if torch.cuda.is_available() else "cpu")
timings["init_deberta_nli"] = time.perf_counter() - t0
ram_nli, vram_nli = get_mem()
print(f"  - DeBERTa-v3 NLI CrossEncoder load: {timings['init_deberta_nli']*1000:.2f} ms (RAM: {ram_nli:.1f} MB, VRAM: {vram_nli:.1f} MB)")

t0 = time.perf_counter()
from src.phase2.deduplication import SemanticDeduplicator
deduplicator = SemanticDeduplicator()
deduplicator.model = retriever._model
timings["init_deduplicator"] = time.perf_counter() - t0
print(f"  - SemanticDeduplicator init (shared ST model): {timings['init_deduplicator']*1000:.2f} ms")

cold_start_total = sum(timings.values())
print(f"\n>> Total Cold-Start Initialization Time: {cold_start_total:.3f} s")

# ---------------------------------------------------------------------------
# 2. GRANULAR STAGE-BY-STAGE EXECUTION BREAKDOWN
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STAGE 2: STEP-BY-STEP PROFILING ON ACTIVE THEME]")
print("=" * 80)

test_theme = "Telemedicine accessibility for remote and underserved rural communities"
test_count = 2

stage_timings = {}

# Step 2.1: Research Gap Matching
t0 = time.perf_counter()
matched_gaps = theme_matcher.match_theme_to_gaps(test_theme, top_k=5)
stage_timings["gap_matching"] = time.perf_counter() - t0
primary_gap = matched_gaps[0]
print(f"2.1 Gap Matching: {stage_timings['gap_matching']*1000:.2f} ms -> Selected {primary_gap['gap_id']} ({primary_gap['concept_a']} <-> {primary_gap['concept_b']})")

# Step 2.2: Subquery formulation
t0 = time.perf_counter()
ca = primary_gap["concept_a"]
cb = primary_gap["concept_b"]
sub_queries = [
    f"{ca} and {cb} in mobile health mHealth interventions",
    f"Integration of {ca} with {cb} in digital health",
    f"{ca} mobile health mHealth applications and outcomes",
    f"{cb} mobile health mHealth applications and outcomes",
]
desc = primary_gap.get("description", "")
if desc:
    sub_queries.append(" ".join(desc.split()[:40]))
stage_timings["query_formulation"] = time.perf_counter() - t0
print(f"2.2 Subquery Formulation: {stage_timings['query_formulation']*1000:.4f} ms ({len(sub_queries)} queries)")

# Step 2.3: Query Embedding Generation (Sequential as in current retrieval.py)
t0 = time.perf_counter()
query_embeddings = []
for q in sub_queries:
    t_q = time.perf_counter()
    emb = retriever.encode_query(q)
    query_embeddings.append(emb)
stage_timings["query_embedding_sequential"] = time.perf_counter() - t0
print(f"2.3 Query Embedding Generation (Current Sequential): {stage_timings['query_embedding_sequential']*1000:.2f} ms (Avg {(stage_timings['query_embedding_sequential']/len(sub_queries))*1000:.2f} ms per query)")

# Test Batched Query Embedding for comparison
t0 = time.perf_counter()
batched_embs = retriever.model.encode(sub_queries, normalize_embeddings=True, convert_to_numpy=True)
stage_timings["query_embedding_batched"] = time.perf_counter() - t0
print(f"    [COMPARE] Batched Query Embedding: {stage_timings['query_embedding_batched']*1000:.2f} ms ({stage_timings['query_embedding_sequential']/max(stage_timings['query_embedding_batched'], 1e-6):.2f}x faster)")

# Step 2.4: FAISS Search & Search Latency
t0 = time.perf_counter()
raw_hits = []
top_k_retrieve = 10
fetch_k = min(len(retriever.chunks), max(top_k_retrieve * 4, 60))
for q_emb in query_embeddings:
    scores, indices = retriever.index.search(q_emb, fetch_k)
    raw_hits.append((scores, indices))
stage_timings["faiss_search_sequential"] = time.perf_counter() - t0
print(f"2.4 FAISS Search (Sequential): {stage_timings['faiss_search_sequential']*1000:.2f} ms for {len(sub_queries)} searches")

# Test Batched FAISS Search for comparison
t0 = time.perf_counter()
stacked_q = np.vstack(query_embeddings)
b_scores, b_indices = retriever.index.search(stacked_q, fetch_k)
stage_timings["faiss_search_batched"] = time.perf_counter() - t0
print(f"    [COMPARE] Batched FAISS Search: {stage_timings['faiss_search_batched']*1000:.2f} ms ({stage_timings['faiss_search_sequential']/max(stage_timings['faiss_search_batched'], 1e-6):.2f}x faster)")

# Step 2.5: Full retrieval.retrieve_for_gap (End-to-End existing retrieval)
t0 = time.perf_counter()
evidence_passages = retriever.retrieve_for_gap(primary_gap, top_k=top_k_retrieve)
stage_timings["retrieval_for_gap_full"] = time.perf_counter() - t0
print(f"2.5 Full Retrieval for Gap (Existing Code): {stage_timings['retrieval_for_gap_full']*1000:.2f} ms (Yielded {len(evidence_passages)} passages)")

# Step 2.6: Evidence context construction & Prompt Building
t0 = time.perf_counter()
prompt = generator._build_prompt(
    primary_gap,
    evidence_passages,
    max_hypotheses=test_count,
    user_theme=test_theme,
    target_population="",
    research_objective="",
)
stage_timings["prompt_construction"] = time.perf_counter() - t0
prompt_chars = len(prompt)
prompt_approx_tokens = prompt_chars / 4
print(f"2.6 Context & Prompt Construction: {stage_timings['prompt_construction']*1000:.2f} ms (Prompt: {prompt_chars} chars, ~{prompt_approx_tokens:.0f} tokens)")

# Step 2.7: Gemini Hypothesis Generation (Live API)
t0 = time.perf_counter()
raw_hypotheses = generator.generate_hypotheses(
    gap=primary_gap,
    evidence_passages=evidence_passages,
    max_hypotheses=test_count,
    user_theme=test_theme,
    target_population="",
    research_objective="",
)
stage_timings["gemini_generation_api"] = time.perf_counter() - t0
print(f"2.7 Gemini Hypothesis Generation API: {stage_timings['gemini_generation_api']:.3f} s (Generated {len(raw_hypotheses)} hypotheses)")

# Step 2.8: Citation Verification
t0 = time.perf_counter()
from src.phase2.evidence_verification import verify_evidence_references
evidence_pool = {
    (ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)): ev
    for ev in evidence_passages
}
verification_results = {}
for h in raw_hypotheses:
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    verification_results[hid] = verify_evidence_references(h, evidence_pool)
stage_timings["citation_verification"] = time.perf_counter() - t0
print(f"2.8 Citation Verification: {stage_timings['citation_verification']*1000:.2f} ms for {len(raw_hypotheses)} hypotheses")

# Step 2.9: Rule-Based Validation Checks
t0 = time.perf_counter()
from src.phase2.rule_checks import run_all_checks
rule_results = {}
for h in raw_hypotheses:
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    rule_results[hid] = run_all_checks(h, evidence_pool=evidence_pool)
stage_timings["rule_checks"] = time.perf_counter() - t0
print(f"2.9 Rule Validation Checks (9 rules each): {stage_timings['rule_checks']*1000:.2f} ms for {len(raw_hypotheses)} hypotheses")

# Step 2.10: DeBERTa NLI Validation (Per-Hypothesis Sequential vs Batched)
t0 = time.perf_counter()
nli_results_seq = {}
evidence_texts = [ev.text_content for ev in evidence_passages]
evidence_ids = [ev.chunk_id for ev in evidence_passages]

for h in raw_hypotheses:
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    htext = h.hypothesis if hasattr(h, "hypothesis") else h.get("hypothesis")
    nli_results_seq[hid] = nli_validator.validate_hypothesis(htext, evidence_texts, evidence_ids, hypothesis_id=hid)
stage_timings["nli_validation_sequential"] = time.perf_counter() - t0
total_nli_pairs = len(raw_hypotheses) * len(evidence_texts)
print(f"2.10 DeBERTa NLI Validation (Sequential): {stage_timings['nli_validation_sequential']*1000:.2f} ms for {total_nli_pairs} pairs ({stage_timings['nli_validation_sequential']*1000/total_nli_pairs:.2f} ms/pair)")

# Benchmark Batched NLI across ALL hypotheses at once
t0 = time.perf_counter()
all_pairs = []
pair_metadata = []
for h in raw_hypotheses:
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    htext = h.hypothesis if hasattr(h, "hypothesis") else h.get("hypothesis")
    for ev in evidence_passages:
        all_pairs.append((ev.text_content, htext))
        pair_metadata.append((hid, ev.chunk_id))

batched_nli_preds = nli_validator.predict_batch_nli(all_pairs)
stage_timings["nli_validation_batched"] = time.perf_counter() - t0
print(f"     [COMPARE] Batched NLI (batch_size={nli_validator.batch_size}): {stage_timings['nli_validation_batched']*1000:.2f} ms ({stage_timings['nli_validation_sequential']/max(stage_timings['nli_validation_batched'], 1e-6):.2f}x speedup)")

# Step 2.11: LLM Judge
t0 = time.perf_counter()
judge_results = {}
from concurrent.futures import ThreadPoolExecutor

def _eval_single(h):
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    return hid, judge.evaluate_hypothesis(h, primary_gap, evidence_passages)

with ThreadPoolExecutor(max_workers=min(len(raw_hypotheses), 4)) as executor:
    for hid, j_res in executor.map(_eval_single, raw_hypotheses):
        judge_results[hid] = j_res
stage_timings["llm_judge_parallel"] = time.perf_counter() - t0
print(f"2.11 LLM Judge Evaluation (Concurrent Gemini API calls): {stage_timings['llm_judge_parallel']:.3f} s for {len(raw_hypotheses)} hypotheses")

# Step 2.12: Multi-Criteria Ranking
t0 = time.perf_counter()
from src.phase2.ranking import rank_hypotheses
ranked_list = rank_hypotheses(
    hypotheses=raw_hypotheses,
    verifications=verification_results,
    nli_results=nli_results_seq,
    judge_results=judge_results,
    rule_results=rule_results,
)
rankings = {r.hypothesis_id: r for r in ranked_list}
stage_timings["ranking"] = time.perf_counter() - t0
print(f"2.12 Multi-Criteria Ranking: {stage_timings['ranking']*1000:.2f} ms")

# Step 2.13: Deduplication
t0 = time.perf_counter()
dedup_res = deduplicator.deduplicate(raw_hypotheses)
stage_timings["deduplication"] = time.perf_counter() - t0
print(f"2.13 Semantic Deduplication: {stage_timings['deduplication']*1000:.2f} ms")

# Step 2.14: Packaging & Serialization
t0 = time.perf_counter()
from backend.services.pipeline_service import PipelineService
pipe_srv = PipelineService.__new__(PipelineService)
pipe_srv._retriever = retriever
pipe_srv._generator = generator
pipe_srv._judge = judge
pipe_srv._deduplicator = deduplicator
pipe_srv.matcher = theme_matcher
packaged = pipe_srv._package_hypotheses(
    raw_hypotheses,
    primary_gap,
    evidence_passages,
    user_population="",
    verification_dict=verification_results,
    rule_dict=rule_results,
    judge_dict=judge_results,
    rank_dict=rankings,
)
stage_timings["packaging"] = time.perf_counter() - t0
print(f"2.14 Response Packaging: {stage_timings['packaging']*1000:.2f} ms")

# Step 2.15: Disk Persistence
t0 = time.perf_counter()
test_run_file = PROJECT_ROOT / "outputs" / "runs" / "_profile_temp_run.json"
with open(test_run_file, "w", encoding="utf-8") as f:
    json.dump([h.model_dump() for h in packaged], f, indent=2)
stage_timings["disk_persistence"] = time.perf_counter() - t0
if test_run_file.exists():
    test_run_file.unlink()
print(f"2.15 Disk Persistence: {stage_timings['disk_persistence']*1000:.2f} ms")

# Total Pipeline Latency
total_warm_latency = (
    stage_timings["gap_matching"]
    + stage_timings["retrieval_for_gap_full"]
    + stage_timings["prompt_construction"]
    + stage_timings["gemini_generation_api"]
    + stage_timings["citation_verification"]
    + stage_timings["rule_checks"]
    + stage_timings["nli_validation_sequential"]
    + stage_timings["llm_judge_parallel"]
    + stage_timings["ranking"]
    + stage_timings["deduplication"]
    + stage_timings["packaging"]
    + stage_timings["disk_persistence"]
)
print("\n" + "=" * 80)
print(f"TOTAL PIPELINE EXECUTION TIME (SINGLE WARM RUN): {total_warm_latency:.3f} s")
print("=" * 80)

# Save baseline profile to JSON
profile_report = {
    "cold_start": {k: round(v, 4) for k, v in timings.items()},
    "cold_start_total_sec": round(cold_start_total, 3),
    "stage_latencies_sec": {k: round(v, 4) for k, v in stage_timings.items()},
    "total_warm_latency_sec": round(total_warm_latency, 3),
}
out_path = PROJECT_ROOT / "outputs" / "baseline_profiling_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(profile_report, f, indent=2)
print(f"\nSaved profiling report to: {out_path}")
