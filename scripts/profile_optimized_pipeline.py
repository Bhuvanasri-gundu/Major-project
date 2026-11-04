"""
scripts/profile_optimized_pipeline.py — Comprehensive Benchmarking of the Optimized Hypothesis Pipeline.
Measures:
1. Optimized cold-start component initialization.
2. Optimized warm-request stage-by-stage timings (batched retrieval, batched Gemini judge, batched DeBERTa NLI, parallel validation).
3. Optimized cached request timing.
4. Direct comparison with original baseline profiling metrics.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
import psutil
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

def get_mem():
    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return ram_mb, vram_mb

print("=" * 80)
print("mHealth HYPOTHESIS PIPELINE: OPTIMIZED PERFORMANCE BENCHMARK")
print("=" * 80)
device_name = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
print(f"Device: {device_name}")
ram_init, vram_init = get_mem()
print(f"Initial Memory: RAM = {ram_init:.1f} MB, VRAM = {vram_init:.1f} MB")
print("-" * 80)

# Load baseline numbers for direct comparison
baseline_path = PROJECT_ROOT / "outputs" / "baseline_profiling_results.json"
baseline_data = {}
if baseline_path.exists():
    try:
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        print("Loaded baseline profiling data for comparison.")
    except Exception as e:
        print(f"Could not load baseline profiling data: {e}")

# ---------------------------------------------------------------------------
# 1. OPTIMIZED COLD-START COMPONENT INITIALIZATION
# ---------------------------------------------------------------------------
print("\n[STAGE 1: OPTIMIZED COLD-START INITIALIZATION]")
cold_timings = {}

t0 = time.perf_counter()
from backend.services.theme_matcher import ThemeMatcher
theme_matcher = ThemeMatcher()
cold_timings["theme_matcher"] = time.perf_counter() - t0
print(f"  - ThemeMatcher data load: {cold_timings['theme_matcher']*1000:.2f} ms")

t0 = time.perf_counter()
from src.phase2.retrieval import EvidenceRetriever
index_path = PROJECT_ROOT / "outputs" / "faiss_index"
retriever = EvidenceRetriever(index_path=index_path)
cold_timings["faiss_index_load"] = time.perf_counter() - t0
print(f"  - FAISS index & metadata load: {cold_timings['faiss_index_load']*1000:.2f} ms ({retriever.index.ntotal} vectors)")

t0 = time.perf_counter()
st_model = retriever.model
cold_timings["sentence_transformer"] = time.perf_counter() - t0
ram_st, vram_st = get_mem()
print(f"  - SentenceTransformer model ({retriever.embedding_model_name}): {cold_timings['sentence_transformer']*1000:.2f} ms (RAM: {ram_st:.1f} MB, VRAM: {vram_st:.1f} MB)")

t0 = time.perf_counter()
from src.phase2.hypothesis_generation import HypothesisGenerator
generator = HypothesisGenerator(use_mock=False)
cold_timings["hypothesis_generator"] = time.perf_counter() - t0
print(f"  - HypothesisGenerator init ({generator.model_name}): {cold_timings['hypothesis_generator']*1000:.2f} ms")

t0 = time.perf_counter()
from src.phase2.llm_judge import LLMJudge
judge = LLMJudge(use_mock=False)
cold_timings["llm_judge"] = time.perf_counter() - t0
print(f"  - LLMJudge init ({judge.model_name}): {cold_timings['llm_judge']*1000:.2f} ms")

t0 = time.perf_counter()
from src.phase2.nli_validation import NLIValidator
nli_validator = NLIValidator(device="cuda" if torch.cuda.is_available() else "cpu")
cold_timings["deberta_nli"] = time.perf_counter() - t0
ram_nli, vram_nli = get_mem()
print(f"  - DeBERTa-v3 NLI CrossEncoder ({nli_validator.device}): {cold_timings['deberta_nli']*1000:.2f} ms (RAM: {ram_nli:.1f} MB, VRAM: {vram_nli:.1f} MB)")

t0 = time.perf_counter()
from src.phase2.deduplication import SemanticDeduplicator
deduplicator = SemanticDeduplicator()
deduplicator.model = retriever._model
cold_timings["deduplicator"] = time.perf_counter() - t0
print(f"  - SemanticDeduplicator init (shared ST): {cold_timings['deduplicator']*1000:.2f} ms")

cold_start_total = sum(cold_timings.values())
print(f"\n>> Total Optimized Cold-Start Initialization: {cold_start_total:.3f} s")

# ---------------------------------------------------------------------------
# 2. GRANULAR STAGE-BY-STAGE BENCHMARK ON IDENTICAL TEST INPUT
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STAGE 2: OPTIMIZED STAGE-BY-STAGE EXECUTION BREAKDOWN]")
print("=" * 80)

test_theme = "Telemedicine accessibility for remote and underserved rural communities"
test_count = 2

opt_stage_timings = {}

# 2.1 Gap Matching
t0 = time.perf_counter()
matched_gaps = theme_matcher.match_theme_to_gaps(test_theme, top_k=5)
opt_stage_timings["gap_matching"] = time.perf_counter() - t0
primary_gap = matched_gaps[0]
print(f"2.1 Gap Matching: {opt_stage_timings['gap_matching']*1000:.2f} ms")

# 2.2 Batched & Cached Retrieval for Gap
t0 = time.perf_counter()
evidence_passages = retriever.retrieve_for_gap(primary_gap, top_k=10)
opt_stage_timings["retrieval_first_call"] = time.perf_counter() - t0
print(f"2.2 Optimized FAISS Retrieval (Batched subqueries): {opt_stage_timings['retrieval_first_call']*1000:.2f} ms ({len(evidence_passages)} passages)")

# Test cached retrieval latency
t0 = time.perf_counter()
_ = retriever.retrieve_for_gap(primary_gap, top_k=10)
opt_stage_timings["retrieval_cached_call"] = time.perf_counter() - t0
print(f"    [CACHED] FAISS Retrieval: {opt_stage_timings['retrieval_cached_call']*1000:.4f} ms ({opt_stage_timings['retrieval_first_call']/max(opt_stage_timings['retrieval_cached_call'], 1e-6):.1f}x faster)")

# 2.3 Gemini Hypothesis Generation
t0 = time.perf_counter()
raw_hypotheses = generator.generate_hypotheses(
    gap=primary_gap,
    evidence_passages=evidence_passages,
    max_hypotheses=test_count,
    user_theme=test_theme,
    target_population="",
    research_objective="",
)
opt_stage_timings["gemini_generation_api"] = time.perf_counter() - t0
print(f"2.3 Gemini Hypothesis Generation API: {opt_stage_timings['gemini_generation_api']:.3f} s ({len(raw_hypotheses)} hypotheses)")

# 2.4 Multi-Layered Validation: Sequential vs Concurrent Execution
print("\n  [Validation Stage Comparison]")
evidence_pool = {
    (ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)): ev
    for ev in evidence_passages
}

# 2.4a Citation Verification
t0 = time.perf_counter()
from src.phase2.evidence_verification import verify_evidence_references
verification_results = {}
for h in raw_hypotheses:
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    verification_results[hid] = verify_evidence_references(h, evidence_pool)
opt_stage_timings["citation_verification"] = time.perf_counter() - t0
print(f"    - Citation Verification: {opt_stage_timings['citation_verification']*1000:.2f} ms")

# 2.4b Rule Checks
t0 = time.perf_counter()
from src.phase2.rule_checks import run_all_checks
rule_results = {}
for h in raw_hypotheses:
    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
    rule_results[hid] = run_all_checks(h, evidence_pool=evidence_pool)
opt_stage_timings["rule_checks"] = time.perf_counter() - t0
print(f"    - Rule Checks (9 rules each): {opt_stage_timings['rule_checks']*1000:.2f} ms")

# 2.4c Batched DeBERTa NLI Validation
t0 = time.perf_counter()
nli_results = nli_validator.validate_hypotheses_batch(raw_hypotheses, evidence_passages)
opt_stage_timings["deberta_nli_batched"] = time.perf_counter() - t0
print(f"    - DeBERTa NLI Batched: {opt_stage_timings['deberta_nli_batched']*1000:.2f} ms ({len(raw_hypotheses)*len(evidence_passages)} pairs)")

# 2.4d Batched LLM Judge (1 unified call)
t0 = time.perf_counter()
judge_results = judge.evaluate_hypotheses_batch(raw_hypotheses, primary_gap, evidence_passages)
opt_stage_timings["llm_judge_batched"] = time.perf_counter() - t0
print(f"    - LLM Judge Unified Batch API: {opt_stage_timings['llm_judge_batched']:.3f} s (Evaluated all {len(raw_hypotheses)} hypotheses)")

# 2.4e Parallel Validation (All 3 Concurrent)
t0 = time.perf_counter()
from concurrent.futures import ThreadPoolExecutor

def _t_judge():
    return judge.evaluate_hypotheses_batch(raw_hypotheses, primary_gap, evidence_passages)

def _t_nli():
    return nli_validator.validate_hypotheses_batch(raw_hypotheses, evidence_passages)

def _t_cit_rules():
    v = {h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id"): verify_evidence_references(h, evidence_pool) for h in raw_hypotheses}
    r = {h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id"): run_all_checks(h, evidence_pool=evidence_pool) for h in raw_hypotheses}
    return v, r

with ThreadPoolExecutor(max_workers=3) as executor:
    fj = executor.submit(_t_judge)
    fn = executor.submit(_t_nli)
    fcr = executor.submit(_t_cit_rules)
    j_res_p = fj.result()
    n_res_p = fn.result()
    v_res_p, r_res_p = fcr.result()

opt_stage_timings["validation_parallel_total"] = time.perf_counter() - t0
print(f"    >> Concurrent Validation Total (Judge + NLI + Citations + Rules): {opt_stage_timings['validation_parallel_total']:.3f} s")

# 2.5 Multi-Criteria Ranking
t0 = time.perf_counter()
from src.phase2.ranking import rank_hypotheses
ranked_list = rank_hypotheses(
    hypotheses=raw_hypotheses,
    verifications=v_res_p,
    nli_results=n_res_p,
    judge_results=j_res_p,
    rule_results=r_res_p,
)
rankings = {r.hypothesis_id: r for r in ranked_list}
opt_stage_timings["ranking"] = time.perf_counter() - t0
print(f"2.5 Ranking: {opt_stage_timings['ranking']*1000:.2f} ms")

# 2.6 Deduplication
t0 = time.perf_counter()
dedup_res = deduplicator.deduplicate(raw_hypotheses)
opt_stage_timings["deduplication"] = time.perf_counter() - t0
print(f"2.6 Deduplication: {opt_stage_timings['deduplication']*1000:.2f} ms")

# ---------------------------------------------------------------------------
# 3. END-TO-END PIPELINESERVICE BENCHMARK (WARM & CACHED)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STAGE 3: END-TO-END PIPELINE SERVICE BENCHMARK]")
print("=" * 80)

from backend.services.pipeline_service import get_pipeline_service
from backend.schemas import GenerateRequest

pipeline_service = get_pipeline_service()
req = GenerateRequest(
    theme=test_theme,
    population="Underserved rural populations",
    objective="Evaluate telemedicine accessibility and efficacy",
    hypothesis_count=test_count,
)

# Run 1: Warm request (Cold caches for this exact prompt)
t0 = time.perf_counter()
run_resp_1 = pipeline_service.execute_run(req)
time_warm_run = time.perf_counter() - t0
print(f"3.1 Warm Pipeline Request: {time_warm_run:.3f} s (Status: {run_resp_1.status}, Hypotheses: {len(run_resp_1.hypotheses)})")

# Run 2: Cached identical request
t0 = time.perf_counter()
run_resp_2 = pipeline_service.execute_run(req)
time_cached_run = time.perf_counter() - t0
print(f"3.2 Cached Pipeline Request: {time_cached_run*1000:.2f} ms (Speedup: {time_warm_run/max(time_cached_run, 1e-6):.1f}x)")

# ---------------------------------------------------------------------------
# 4. MEMORY AND HARDWARE UTILIZATION
# ---------------------------------------------------------------------------
ram_final, vram_final = get_mem()
print(f"\nFinal Memory: RAM = {ram_final:.1f} MB, VRAM = {vram_final:.1f} MB")
print(f"RAM Growth: +{ram_final - ram_init:.1f} MB | VRAM Allocated: {vram_final:.1f} MB")

# ---------------------------------------------------------------------------
# 5. SUMMARY COMPARISON TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("BENCHMARK COMPARISON: BASELINE vs OPTIMIZED")
print("=" * 80)

base_warm_total = baseline_data.get("warm_request_total_seconds", 43.178)
base_judge = baseline_data.get("stage_timings", {}).get("llm_judge_parallel", 20.041)
base_nli = baseline_data.get("stage_timings", {}).get("nli_validation_sequential", 2.835)
base_query_emb = baseline_data.get("stage_timings", {}).get("query_embedding_sequential", 0.550)
base_faiss_search = baseline_data.get("stage_timings", {}).get("faiss_search_sequential", 0.048)

print(f"{'Metric / Stage':<35} | {'Baseline':<12} | {'Optimized':<12} | {'Speedup':<10}")
print("-" * 75)
print(f"{'Query Embedding (subqueries)':<35} | {base_query_emb*1000:>9.1f} ms | {opt_stage_timings['retrieval_first_call']*1000:>9.1f} ms | {base_query_emb/max(opt_stage_timings['retrieval_first_call'], 1e-6):>8.1f}x")
print(f"{'DeBERTa NLI Inference':<35} | {base_nli*1000:>9.1f} ms | {opt_stage_timings['deberta_nli_batched']*1000:>9.1f} ms | {base_nli/max(opt_stage_timings['deberta_nli_batched'], 1e-6):>8.1f}x")
print(f"{'LLM Judge Evaluation':<35} | {base_judge:>9.2f} s  | {opt_stage_timings['llm_judge_batched']:>9.2f} s  | {base_judge/max(opt_stage_timings['llm_judge_batched'], 1e-6):>8.1f}x")
print(f"{'Parallel Validation Block':<35} | {base_judge+base_nli:>9.2f} s  | {opt_stage_timings['validation_parallel_total']:>9.2f} s  | {(base_judge+base_nli)/max(opt_stage_timings['validation_parallel_total'], 1e-6):>8.1f}x")
print(f"{'End-to-End Warm Request':<35} | {base_warm_total:>9.2f} s  | {time_warm_run:>9.2f} s  | {base_warm_total/max(time_warm_run, 1e-6):>8.1f}x")
print(f"{'Cached Pipeline Request':<35} | {base_warm_total:>9.2f} s  | {time_cached_run*1000:>9.1f} ms | {base_warm_total/max(time_cached_run, 1e-6):>8.0f}x")
print("=" * 80)

# Save results to JSON
results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "device": device_name,
    "cold_start": {
        "components": cold_timings,
        "total_seconds": cold_start_total,
    },
    "stage_timings": opt_stage_timings,
    "end_to_end": {
        "warm_run_seconds": time_warm_run,
        "cached_run_seconds": time_cached_run,
        "speedup_warm": base_warm_total / max(time_warm_run, 1e-6),
        "speedup_cached": base_warm_total / max(time_cached_run, 1e-6),
    },
    "memory": {
        "ram_mb": ram_final,
        "vram_mb": vram_final,
        "ram_growth_mb": ram_final - ram_init,
    },
    "comparison": {
        "baseline_warm_seconds": base_warm_total,
        "optimized_warm_seconds": time_warm_run,
        "reduction_percentage": round((1.0 - time_warm_run / base_warm_total) * 100, 2),
    }
}

out_path = PROJECT_ROOT / "outputs" / "optimized_profiling_results.json"
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved profiling results to {out_path}")
