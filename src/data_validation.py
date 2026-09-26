"""
data_validation.py — Dataset validation, quality checks, and cleaning.
Produces validation report, summary JSON, and cleaned candidate dataset.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections import Counter

from .config_loader import load_config, get_project_root, ensure_dirs


# ================================================================
# DATA LOADING
# ================================================================

def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load the mHealth dataset, auto-detecting CSV vs Excel format.

    The dataset may have a .xls extension but contain CSV data.
    This function detects the actual format and loads accordingly.
    """
    cfg = load_config()
    if dataset_path is None:
        dataset_path = get_project_root() / cfg["paths"]["dataset"]
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        fallback = get_project_root() / "data" / "pubmed_mhealth_2015_2025_proposed_final.xls"
        if fallback.exists():
            dataset_path = fallback
        else:
            fallback2 = get_project_root().parent / "pubmed_mhealth_2015_2025_proposed_final.xls"
            if fallback2.exists():
                dataset_path = fallback2

    print(f"[DATA] Loading dataset from: {dataset_path}")
    print(f"[DATA] File size: {dataset_path.stat().st_size / 1e6:.1f} MB")

    # Detect actual format by reading first bytes
    with open(dataset_path, "rb") as f:
        header_bytes = f.read(4096)

    # XLS (BIFF) files start with D0 CF 11 E0; XLSX starts with PK (ZIP)
    is_real_excel = header_bytes[:4] in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")

    if is_real_excel:
        print("[DATA] Detected real Excel format, using read_excel()")
        df = pd.read_excel(dataset_path)
    else:
        print("[DATA] Detected CSV format (despite .xls extension), using read_csv()")
        # Try utf-8-sig first (handles BOM), fall back to utf-8, then latin-1
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(dataset_path, encoding=enc)
                print(f"[DATA] Successfully loaded with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue

    print(f"[DATA] Loaded {len(df):,} records with {len(df.columns)} columns")
    print(f"[DATA] Columns: {df.columns.tolist()}")
    return df


# ================================================================
# VALIDATION CHECKS
# ================================================================

def validate_dataset(df: pd.DataFrame) -> dict:
    """Run comprehensive validation checks on the dataset.

    Returns a dict with all validation statistics.
    """
    cfg = load_config()
    val_cfg = cfg.get("data_validation", {})
    year_range = val_cfg.get("year_range", [2015, 2025])

    stats = {}

    # --- Basic counts ---
    stats["total_records"] = len(df)
    stats["columns"] = df.columns.tolist()

    # --- Missing values ---
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        empty_count = int(
            df[col].fillna("").astype(str).str.strip().eq("").sum()
        ) if df[col].dtype == object else 0
        stats[f"missing_{col}"] = null_count
        stats[f"empty_{col}"] = empty_count

    # --- Duplicates ---
    if "pmid" in df.columns:
        stats["duplicate_pmids"] = int(df["pmid"].duplicated().sum())
        stats["unique_pmids"] = int(df["pmid"].nunique())

    if "doi" in df.columns:
        doi_valid = df["doi"].dropna().replace("", np.nan).dropna()
        stats["duplicate_dois"] = int(doi_valid.duplicated().sum())

    if "document" in df.columns:
        stats["duplicate_documents"] = int(df["document"].duplicated().sum())

    # --- Year distribution ---
    if "year" in df.columns:
        year_counts = (
            df["year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        stats["year_distribution"] = {str(k): v for k, v in year_counts.items()}
        stats["year_min"] = int(df["year"].dropna().min())
        stats["year_max"] = int(df["year"].dropna().max())
        stats["records_outside_year_range"] = int(
            ((df["year"] < year_range[0]) | (df["year"] > year_range[1])).sum()
        )

    # --- Publication type distribution ---
    if "publication_types" in df.columns:
        pt_counts = (
            df["publication_types"]
            .dropna()
            .value_counts()
            .head(30)
            .to_dict()
        )
        stats["publication_type_distribution"] = pt_counts

    # --- Text length stats ---
    for col in ("title", "abstract", "document"):
        if col in df.columns:
            lengths = df[col].fillna("").astype(str).str.len()
            stats[f"{col}_length_mean"] = float(lengths.mean())
            stats[f"{col}_length_median"] = float(lengths.median())
            stats[f"{col}_length_min"] = int(lengths.min())
            stats[f"{col}_length_max"] = int(lengths.max())

    # --- Short text records ---
    if "title" in df.columns:
        stats["short_titles"] = int(
            (df["title"].fillna("").str.len() < val_cfg.get("min_title_length", 10)).sum()
        )
    if "abstract" in df.columns:
        stats["short_abstracts"] = int(
            (df["abstract"].fillna("").str.len() < val_cfg.get("min_abstract_length", 50)).sum()
        )
    if "document" in df.columns:
        stats["short_documents"] = int(
            (df["document"].fillna("").str.len() < val_cfg.get("min_document_length", 80)).sum()
        )

    return stats


def print_validation_report(stats: dict) -> None:
    """Pretty-print the validation report."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal records: {stats['total_records']:,}")
    print(f"Columns: {stats['columns']}")

    print("\n--- Missing Values ---")
    for col in stats["columns"]:
        m = stats.get(f"missing_{col}", 0)
        e = stats.get(f"empty_{col}", 0)
        print(f"  {col}: {m:,} null, {e:,} empty")

    print("\n--- Duplicates ---")
    for key in ("duplicate_pmids", "duplicate_dois", "duplicate_documents"):
        if key in stats:
            print(f"  {key}: {stats[key]:,}")

    if "year_distribution" in stats:
        print("\n--- Year Distribution ---")
        for year, count in sorted(stats["year_distribution"].items()):
            print(f"  {year}: {count:,}")

    if "publication_type_distribution" in stats:
        print("\n--- Publication Types (top 20) ---")
        for pt, count in list(stats["publication_type_distribution"].items())[:20]:
            print(f"  {pt}: {count:,}")

    print("\n--- Text Length Stats ---")
    for col in ("title", "abstract", "document"):
        if f"{col}_length_mean" in stats:
            print(
                f"  {col}: mean={stats[f'{col}_length_mean']:.0f}, "
                f"median={stats[f'{col}_length_median']:.0f}, "
                f"min={stats[f'{col}_length_min']}, "
                f"max={stats[f'{col}_length_max']}"
            )

    print("=" * 70)


# ================================================================
# CLEANING
# ================================================================

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Clean the dataset, returning (cleaned_df, filter_log).

    Every filtering step is logged with before/after counts.
    """
    cfg = load_config()
    val_cfg = cfg.get("data_validation", {})
    year_range = val_cfg.get("year_range", [2015, 2025])

    log = []
    current = df.copy()

    def _log_step(reason: str, before: int, after: int):
        removed = before - after
        entry = {
            "step": reason,
            "before": before,
            "removed": removed,
            "after": after,
        }
        log.append(entry)
        print(
            f"  [{reason}] before={before:,}  removed={removed:,}  "
            f"after={after:,}"
        )

    print("\n[CLEAN] Starting dataset cleaning...")

    # 1. Remove duplicate PMIDs
    before = len(current)
    if "pmid" in current.columns:
        current = current.drop_duplicates(subset=["pmid"], keep="first")
    _log_step("Remove duplicate PMIDs", before, len(current))

    # 2. Remove records with missing/empty titles
    before = len(current)
    if "title" in current.columns:
        current = current[
            current["title"].fillna("").str.strip().str.len()
            >= val_cfg.get("min_title_length", 10)
        ]
    _log_step("Remove short/missing titles", before, len(current))

    # 3. Remove records with missing/empty abstracts
    before = len(current)
    if "abstract" in current.columns:
        current = current[
            current["abstract"].fillna("").str.strip().str.len()
            >= val_cfg.get("min_abstract_length", 50)
        ]
    _log_step("Remove short/missing abstracts", before, len(current))

    # 4. Filter year range
    before = len(current)
    if "year" in current.columns:
        current["year"] = pd.to_numeric(current["year"], errors="coerce")
        current = current[
            (current["year"] >= year_range[0])
            & (current["year"] <= year_range[1])
        ]
    _log_step(
        f"Filter year range {year_range[0]}-{year_range[1]}",
        before,
        len(current),
    )

    # 5. Remove records with very short documents
    before = len(current)
    if "document" in current.columns:
        current = current[
            current["document"].fillna("").str.strip().str.len()
            >= val_cfg.get("min_document_length", 80)
        ]
    _log_step("Remove short documents", before, len(current))

    # 6. Remove exact duplicate documents
    before = len(current)
    if "document" in current.columns:
        current = current.drop_duplicates(subset=["document"], keep="first")
    _log_step("Remove duplicate documents", before, len(current))

    # Reset index
    current = current.reset_index(drop=True)

    print(f"\n[CLEAN] Final cleaned dataset: {len(current):,} records")
    return current, log


# ================================================================
# SAVE OUTPUTS
# ================================================================

def save_validation_outputs(
    df: pd.DataFrame,
    stats: dict,
    filter_log: list[dict],
    cleaned_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save all validation outputs and return paths."""
    dirs = ensure_dirs()
    out = dirs["outputs"]
    saved = {}

    # Validation report CSV
    report_rows = []
    for k, v in stats.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        report_rows.append({"metric": k, "value": v})
    report_df = pd.DataFrame(report_rows)
    p = out / "data_validation_report.csv"
    report_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["validation_report"] = p
    print(f"[SAVE] Validation report -> {p}")

    # Summary JSON
    p = out / "data_validation_summary.json"
    # Convert numpy types for JSON serialisation
    clean_stats = {}
    for k, v in stats.items():
        if isinstance(v, (np.integer,)):
            clean_stats[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_stats[k] = float(v)
        else:
            clean_stats[k] = v
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=2, ensure_ascii=False, default=str)
    saved["validation_summary"] = p
    print(f"[SAVE] Validation summary -> {p}")

    # Filter log
    p = out / "cleaning_filter_log.csv"
    pd.DataFrame(filter_log).to_csv(p, index=False, encoding="utf-8-sig")
    saved["filter_log"] = p
    print(f"[SAVE] Cleaning filter log -> {p}")

    # Cleaned dataset
    p = out / "mhealth_candidate_cleaned.csv"
    cleaned_df.to_csv(p, index=False, encoding="utf-8-sig")
    saved["cleaned_dataset"] = p
    print(f"[SAVE] Cleaned dataset -> {p}  ({len(cleaned_df):,} records)")

    return saved
