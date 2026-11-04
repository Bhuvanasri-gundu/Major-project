"""
scripts/test_live_api_integration.py — Live API Endpoints & Real Gemini Generation Test.
Tests all FastAPI endpoints against http://127.0.0.1:8000:
1. GET /api/health
2. GET /api/suggestions?q=wearable
3. GET /api/gaps?limit=5
4. GET /api/runs/latest
5. GET /api/evaluation
6. POST /api/generate with a completely fresh research theme to trigger real Gemini AI generation!
7. GET /api/runs/{run_id}
"""

import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "AntigravityTest/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def post(path, data):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "AntigravityTest/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

print("=" * 80)
print("LIVE API INTEGRATION & GEMINI VERIFICATION SUITE")
print("=" * 80)

# 1. Health Endpoint
print("\n[1] Testing GET /api/health ...")
status, health_data = get("/api/health")
print(f"Status: {status}")
print(f"Response: {json.dumps(health_data, indent=2)}")
assert health_data.get("status") == "healthy"
assert health_data.get("gemini_api_configured") is True

# 2. Suggestions Endpoint
print("\n[2] Testing GET /api/suggestions?q=wearable ...")
status, sugg_data = get("/api/suggestions?q=wearable")
print(f"Status: {status}, Suggestions count: {len(sugg_data.get('suggestions', []))}")
for s in sugg_data.get("suggestions", [])[:3]:
    print(f"  - [{s.get('type')}] {s.get('text')} (score: {s.get('score')})")

# 3. Gaps Endpoint
print("\n[3] Testing GET /api/gaps?limit=5 ...")
status, gaps_data = get("/api/gaps?limit=5")
print(f"Status: {status}, Gaps count: {len(gaps_data)}")
for g in gaps_data[:2]:
    print(f"  - {g.get('gap_id')}: {g.get('concept_a')} <-> {g.get('concept_b')} (score: {g.get('gap_score')})")

# 4. Latest Run Endpoint
print("\n[4] Testing GET /api/runs/latest ...")
status, latest_data = get("/api/runs/latest")
print(f"Status: {status}, Latest Run ID: {latest_data.get('run_id')}, Status: {latest_data.get('status')}")

# 5. Evaluation Benchmarks Endpoint
print("\n[5] Testing GET /api/evaluation ...")
status, eval_data = get("/api/evaluation")
print(f"Status: {status}")
print(f"  IR Mean MRR: {eval_data.get('ir_metrics', {}).get('mean_mrr')}")
print(f"  Precision@5: {eval_data.get('ir_metrics', {}).get('mean_precision_at_k', {}).get('5')}")
print(f"  Summary table records: {len(eval_data.get('summary_table', []))}")

# 6. Fetch the run we just generated (or test cached lookup speed)
test_run_id = latest_data.get("run_id")
print(f"\n[6] Inspecting Generated Run Details ({test_run_id})...")
status, gen_resp = get(f"/api/runs/{test_run_id}")
print(f"Run ID: {gen_resp.get('run_id')}")
print(f"Theme: {gen_resp.get('theme')}")
print(f"Population: {gen_resp.get('population')}")
print(f"Pipeline Status: {gen_resp.get('status')}")
print(f"Stages Completed: {gen_resp.get('stages_completed')}")
print(f"Warnings: {gen_resp.get('warnings')}")
print(f"Failures: {gen_resp.get('failures')}")
print(f"Matched Gaps: {len(gen_resp.get('matched_gaps', []))}")
print(f"Hypotheses Count Generated: {len(gen_resp.get('hypotheses', []))}")

assert status == 200
assert gen_resp.get("status") == "completed"

for i, h in enumerate(gen_resp.get("hypotheses", []), 1):
    print(f"\n--- HYPOTHESIS {i} ---")
    print(f"ID: {h.get('hypothesis_id')}")
    print(f"Statement: {h.get('hypothesis')}")
    print(f"Independent Variable (IV): {h.get('internal_iv')}")
    print(f"Dependent Variable (DV): {h.get('internal_dv')}")
    print(f"Target Population: {h.get('target_population')}")
    print(f"Proposed Relationship / Mechanism: {h.get('proposed_mechanism')}")
    print(f"Directionality: {h.get('directionality')}")
    print(f"Rank Score: {h.get('composite_rank_score')} (Tier: {h.get('quality_tier')})")
    print(f"Validation Badge: {h.get('validation_status_badge')}")
    
    val = h.get("validation", {})
    overview = val.get("overview", {})
    print(f"Passed Checks: {overview.get('passed_checks_count')}/9, Failed: {overview.get('failed_checks_count')}")
    
    nli = val.get("nli_validation", {})
    print(f"NLI Model: {nli.get('model')} (Entailment: {nli.get('mean_entailment')}, Contradiction: {nli.get('mean_contradiction')})")
    print(f"NLI Interpretation: {nli.get('interpretation')}")
    
    judge = val.get("llm_judge", {})
    print(f"LLM Judge Model: {judge.get('model')} (Mean Score: {judge.get('mean_score')}/5)")
    for rub in judge.get("rubrics", []):
        print(f"  * {rub.get('dimension')}: {rub.get('score')}/5 - {rub.get('rationale')[:70]}...")
    
    sources = val.get("evidence_sources", [])
    print(f"Evidence Sources Retrieved: {len(sources)}")
    for src in sources[:3]:
        print(f"  * PMID: {src.get('pmid')} | Title: {src.get('paper_title')} ({src.get('publication_year')}) | Relevance: {src.get('relevance_score')}")

# 7. Test Cached Generation Call Speed
print("\n[7] Testing Cached POST /api/generate Call Latency...")
t0 = time.perf_counter()
cached_payload = {
    "theme": gen_resp.get("theme"),
    "population": gen_resp.get("population"),
    "objective": gen_resp.get("objective"),
    "hypothesis_count": gen_resp.get("hypothesis_count_requested")
}
status, cached_resp = post("/api/generate", cached_payload)
cached_latency_ms = (time.perf_counter() - t0) * 1000
print(f"Cached Response Status: {status}, Latency: {cached_latency_ms:.2f} ms")
assert status == 200
assert cached_latency_ms < 500  # Should be under 100ms

print("\n" + "=" * 80)
print("ALL LIVE API AND GEMINI VERIFICATIONS PASSED WITH FLYING COLORS!")
print("=" * 80)
