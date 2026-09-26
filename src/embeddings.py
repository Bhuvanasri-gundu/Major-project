"""
embeddings.py — Scientific document embeddings using SPECTER2
(or a SentenceTransformer fallback).

Generates, saves, and loads dense vector representations of
scientific documents with proper PMID alignment.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config_loader import load_config, ensure_dirs, set_seed


def _get_device(cfg_device: str = "auto") -> str:
    """Resolve device string."""
    if cfg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg_device


def generate_embeddings(
    df: pd.DataFrame,
    text_column: str = "clean_document",
    resume: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate document embeddings for the corpus.

    Parameters
    ----------
    df : preprocessed dataframe (must have ``text_column`` and ``pmid``)
    text_column : column to embed
    resume : if True and embeddings already exist on disk, load them

    Returns
    -------
    embeddings : (N, D) numpy array
    index_df : dataframe with pmid and embedding_row columns
    """
    cfg = load_config()
    emb_cfg = cfg["embeddings"]
    dirs = ensure_dirs()

    emb_path = dirs["outputs"] / emb_cfg["save_file"]
    idx_path = dirs["outputs"] / emb_cfg["index_file"]

    # --- Resume check ---
    if resume and emb_path.exists() and idx_path.exists():
        print(f"[EMBED] Found existing embeddings at {emb_path}")
        embeddings = np.load(emb_path)
        index_df = pd.read_csv(idx_path)

        # Verify alignment
        if len(embeddings) == len(index_df) == len(df):
            if list(index_df["pmid"]) == list(df["pmid"]):
                print(f"[EMBED] Loaded {embeddings.shape[0]} embeddings, "
                      f"dim={embeddings.shape[1]}")
                return embeddings, index_df
        print("[EMBED] Alignment mismatch — regenerating embeddings")

    # --- Setup ---
    set_seed()
    device = _get_device(emb_cfg.get("device", "auto"))
    batch_size = emb_cfg.get("batch_size", 32)
    model_name = emb_cfg.get("model_name", "allenai/specter2_base")
    fallback = emb_cfg.get("fallback_model", "allenai/specter")

    print(f"\n[EMBED] Device: {device}")
    print(f"[EMBED] Batch size: {batch_size}")
    print(f"[EMBED] Documents: {len(df):,}")

    # --- Load model ---
    from sentence_transformers import SentenceTransformer

    model = None
    used_model = model_name
    for mname in (model_name, fallback, "sentence-transformers/allenai-specter"):
        try:
            print(f"[EMBED] Trying model: {mname}")
            model = SentenceTransformer(mname, device=device)
            used_model = mname
            print(f"[EMBED] Loaded model: {mname}")
            break
        except Exception as e:
            print(f"[EMBED] Could not load {mname}: {e}")
            continue

    if model is None:
        raise RuntimeError(
            "Could not load any embedding model. "
            "Please install sentence-transformers and check your network."
        )

    # --- Prepare texts ---
    texts = df[text_column].fillna("").tolist()

    # Truncate very long texts to max_length tokens (approx)
    max_len = emb_cfg.get("max_length", 512)
    # Rough char estimate: ~4 chars per token
    max_chars = max_len * 4
    texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]

    # --- Encode in batches ---
    print(f"[EMBED] Encoding {len(texts):,} documents...")
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"[EMBED] Embedding shape: {embeddings.shape}")

    # --- Build index ---
    index_df = pd.DataFrame({
        "pmid": df["pmid"].values,
        "embedding_row": range(len(df)),
    })

    # --- Save ---
    np.save(emb_path, embeddings)
    print(f"[SAVE] Embeddings -> {emb_path}")

    index_df.to_csv(idx_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Embedding index -> {idx_path}")

    # Save model metadata
    meta_path = dirs["outputs"] / "embedding_model_info.txt"
    with open(meta_path, "w") as f:
        f.write(f"model: {used_model}\n")
        f.write(f"dimensions: {embeddings.shape[1]}\n")
        f.write(f"num_documents: {embeddings.shape[0]}\n")
        f.write(f"device: {device}\n")
    print(f"[SAVE] Model info -> {meta_path}")

    return embeddings, index_df


def load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """Load previously saved embeddings and index."""
    cfg = load_config()
    emb_cfg = cfg["embeddings"]
    dirs = ensure_dirs()

    emb_path = dirs["outputs"] / emb_cfg["save_file"]
    idx_path = dirs["outputs"] / emb_cfg["index_file"]

    embeddings = np.load(emb_path)
    index_df = pd.read_csv(idx_path)

    assert len(embeddings) == len(index_df), (
        f"Alignment error: {len(embeddings)} embeddings vs "
        f"{len(index_df)} index rows"
    )

    print(f"[EMBED] Loaded {embeddings.shape[0]} embeddings, "
          f"dim={embeddings.shape[1]}")
    return embeddings, index_df
