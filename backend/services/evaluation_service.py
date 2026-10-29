"""
backend/services/evaluation_service.py — Loads Phase 3 empirical evaluation benchmarks.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_evaluation_benchmarks() -> Dict[str, Any]:
    """
    Returns empirical evaluation results from Phase 3 benchmarks.
    Never fabricates scores — loads directly from validated output artifacts.
    """
    results: Dict[str, Any] = {
        "ir_metrics": {},
        "grounding_audit": {},
        "baseline_comparisons": [],
        "ablation_study": [],
        "summary_table": [],
    }

    # 1. JSON benchmarks
    json_path = OUTPUTS_DIR / "phase3_evaluation_results.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results["ir_metrics"] = data.get("retrieval_evaluation", {})
                results["grounding_audit"] = data.get("grounding_evaluation", {})
                results["baseline_comparisons"] = data.get("baseline_comparison", [])
                results["ablation_study"] = data.get("ablation_study", [])
        except Exception:
            pass

    # 2. Summary CSV
    csv_path = OUTPUTS_DIR / "phase3_evaluation_summary.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            results["summary_table"] = df.to_dict(orient="records")
        except Exception:
            pass

    return results
