"""
scripts/benchmark_retrieval_k.py — Empirical benchmark of retrieval candidate sizes (k=10, 20, 30).
Measures: Retrieval latency, Precision@K, MRR, nDCG@10, evidence diversity, and source coverage.
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2.retrieval import EvidenceRetriever
from src.phase3.retrieval_eval import evaluate_retrieval_for_gap

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
gaps_csv = OUTPUTS_DIR / "candidate_research_gaps.csv"
doc_concepts_csv = OUTPUTS_DIR / "document_concepts.csv"

gaps_df = pd.read_csv(gaps_csv)
doc_concepts_df = pd.read_csv(doc_concepts_csv)

retriever = EvidenceRetriever(index_path=OUTPUTS_DIR / "faiss_index")

# Test on top 5 benchmark gaps
test_gaps = gaps_df.head(5).to_dict(orient="records")
k_values_to_test = [10, 20, 30]

results_by_k = {}

for k in k_values_to_test:
    latencies = []
    precisions_5 = []
    precisions_10 = []
    mrrs = []
    ndcgs_10 = []
    distinct_papers_counts = []
    
    for gap in test_gaps:
        t0 = time.perf_counter()
        ev_list = retriever.retrieve_for_gap(gap, top_k=k)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        
        # Diversity: unique papers count
        unique_pmids = set(e.source_pmid for e in ev_list)
        distinct_papers_counts.append(len(unique_pmids))
        
        # IR metrics against MeSH ground truth
        metrics = evaluate_retrieval_for_gap(gap, ev_list, doc_concepts_df, k_values=(5, 10))
        precisions_5.append(metrics.precision_at_k.get(5, 0.0))
        precisions_10.append(metrics.precision_at_k.get(10, 0.0))
        mrrs.append(metrics.mrr)
        ndcgs_10.append(metrics.ndcg_at_k.get(10, 0.0))
        
    results_by_k[k] = {
        "k": k,
        "mean_latency_ms": round(float(np.mean(latencies)) * 1000, 2),
        "mean_precision_at_5": round(float(np.mean(precisions_5)), 4),
        "mean_precision_at_10": round(float(np.mean(precisions_10)), 4),
        "mean_mrr": round(float(np.mean(mrrs)), 4),
        "mean_ndcg_at_10": round(float(np.mean(ndcgs_10)), 4),
        "mean_distinct_papers": round(float(np.mean(distinct_papers_counts)), 1),
        "evidence_diversity_ratio": round(float(np.mean(distinct_papers_counts)) / k, 4),
    }

print("=" * 80)
print("RETRIEVAL CANDIDATE SIZE BENCHMARK RESULTS (k=10, 20, 30)")
print("=" * 80)
print(pd.DataFrame(results_by_k.values()).to_string(index=False))

out_file = OUTPUTS_DIR / "retrieval_k_benchmark.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results_by_k, f, indent=2)
print(f"\nSaved benchmark results to: {out_file}")
