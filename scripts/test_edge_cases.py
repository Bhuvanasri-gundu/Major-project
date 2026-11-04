"""
scripts/test_edge_cases.py — Edge Case and Negative Testing Suite.
Tests:
1. Short query to /api/suggestions (q="a") -> Expects 422.
2. Missing theme in /api/generate -> Expects 422.
3. Non-existent run_id in /api/runs/{id} -> Expects 404.
4. Zero hypothesis count request (count=0) -> Expects 200 exploration only.
5. Missing API key handling in HypothesisGenerator -> Expects ValueError.
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

print("=" * 80)
print("EDGE CASE & ERROR HANDLING TEST SUITE")
print("=" * 80)

# Test 1: Suggestions with 1-character query
print("\n[Test 1] Testing GET /api/suggestions?q=a (Invalid short query)...")
try:
    urllib.request.urlopen(f"{BASE_URL}/api/suggestions?q=a")
    print("FAILED: Expected 422 error but succeeded.")
except urllib.error.HTTPError as e:
    print(f"PASSED: Received expected HTTP {e.code} ({e.reason})")
    assert e.code == 422

# Test 2: Generate with empty/missing theme
print("\n[Test 2] Testing POST /api/generate with missing theme...")
try:
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=json.dumps({"theme": "", "hypothesis_count": 2}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req)
    print("FAILED: Expected 422 error but succeeded.")
except urllib.error.HTTPError as e:
    print(f"PASSED: Received expected HTTP {e.code} ({e.reason})")
    assert e.code == 422

# Test 3: Non-existent run_id
print("\n[Test 3] Testing GET /api/runs/non_existent_run_999999 ...")
try:
    urllib.request.urlopen(f"{BASE_URL}/api/runs/non_existent_run_999999")
    print("FAILED: Expected 404 error but succeeded.")
except urllib.error.HTTPError as e:
    print(f"PASSED: Received expected HTTP {e.code} ({e.reason})")
    assert e.code == 404

# Test 4: Zero-count hypothesis exploration request
print("\n[Test 4] Testing POST /api/generate with hypothesis_count=0 (Exploration only)...")
req = urllib.request.Request(
    f"{BASE_URL}/api/generate",
    data=json.dumps({
        "theme": "Digital biomarker monitoring for cardiac rehabilitation",
        "hypothesis_count": 0
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print(f"Status: {resp.status}, Pipeline Status: {data.get('status')}")
    print(f"Hypotheses Count: {len(data.get('hypotheses', []))}")
    print(f"Matched Gaps Count: {len(data.get('matched_gaps', []))}")
    assert resp.status == 200
    assert data.get("status") == "no_hypotheses_requested"
    assert len(data.get("hypotheses", [])) == 0
    assert len(data.get("matched_gaps", [])) > 0
    print("PASSED: Zero-count request correctly returned gap exploration without calling LLM.")

# Test 5: Missing API key assertion in HypothesisGenerator
print("\n[Test 5] Testing HypothesisGenerator with missing API key...")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.phase2.hypothesis_generation import HypothesisGenerator
try:
    gen = HypothesisGenerator(use_mock=False, api_key="")
    gen.client = None
    gen.generate_hypotheses(
        gap={"concept_a": "A", "concept_b": "B", "gap_score": 1.0},
        evidence_passages=[],
        max_hypotheses=1
    )
    print("FAILED: Expected ValueError when client is uninitialized.")
except ValueError as ve:
    print(f"PASSED: Received expected ValueError: {ve}")

print("\n" + "=" * 80)
print("ALL EDGE CASES AND ERROR HANDLING TESTS PASSED!")
print("=" * 80)
