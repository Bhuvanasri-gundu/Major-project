"""
config_loader.py — Centralised configuration loader.
Loads config.yaml relative to the project root, sets random seeds,
and resolves paths using pathlib.
"""

from pathlib import Path
import yaml
import random
import numpy as np

_CONFIG_CACHE = {}


def load_config(config_path: str | Path | None = None) -> dict:
    """Load the YAML config file and return a dict.

    Parameters
    ----------
    config_path : str or Path, optional
        Absolute or relative path to config.yaml.
        Defaults to ``<proposed>/config.yaml``.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config_path = Path(config_path).resolve()

    cache_key = str(config_path)
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    return (get_project_root() / relative).resolve()


def set_seed(seed: int | None = None) -> None:
    """Set random seed for reproducibility."""
    cfg = load_config()
    seed = seed or cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"[CONFIG] Random seed set to {seed}")


def ensure_dirs() -> dict[str, Path]:
    """Create output directories and return a dict of resolved paths."""
    cfg = load_config()
    root = get_project_root()
    dirs = {}
    for key in ("data", "outputs", "figures", "models", "logs"):
        p = root / cfg["paths"].get(key, key)
        p.mkdir(parents=True, exist_ok=True)
        dirs[key] = p
    return dirs
