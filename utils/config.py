from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml

# Project root is the parent of the utils/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_config_path(arg: str) -> Path:
    """Resolve a config path or name to an absolute path.

    - If `arg` is a path (absolute or relative) and exists, use it.
    - Otherwise, look under `<PROJECT_ROOT>/configs/` and try with .yaml/.yml.
    - Allows passing a bare name without extension.
    """
    p = Path(arg)
    candidates = [p]
    if p.suffix == "":
        candidates = [p.with_suffix(".yaml"), p.with_suffix(".yml"), p]

    # Try as given
    for c in candidates:
        if c.exists():
            return c.resolve()

    # Try under configs/
    cfg_dir = PROJECT_ROOT / "configs"
    for c in candidates:
        candidate = cfg_dir / c.name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find config at '{arg}' or under '{cfg_dir}'.")


def load_config(arg: str) -> Dict[str, Any]:
    cfg_path = resolve_config_path(arg)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("_config_path", str(cfg_path))
    return cfg


def _ensure_keys(d: Dict[str, Any], required: Dict[str, Any], ctx: str) -> None:
    missing = []
    for k, sub in required.items():
        if k not in d or d[k] is None:
            missing.append(k)
        elif isinstance(sub, dict):
            if not isinstance(d[k], dict):
                missing.append(k)
            else:
                _ensure_keys(d[k], sub, f"{ctx}.{k}")
    if missing:
        raise KeyError(f"Missing required config keys in {ctx}: {missing}")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Basic schema validation and path normalization.

    Required sections and keys:
      data: train_csv, val_csv
      training: batch_size, num_epochs, learning_rate
      model: one of {model_name} for text-only or {text_model, vision_model} for multimodal
    """
    required = {
        "data": {"train_csv": None, "val_csv": None},
        "training": {"batch_size": None, "num_epochs": None, "learning_rate": None},
        "model": {},
    }
    _ensure_keys(cfg, required, "root")

    model = cfg.get("model", {})
    has_text_only = "model_name" in model
    has_mm = "text_model" in model and "vision_model" in model
    if not (has_text_only or has_mm):
        raise KeyError(
            "model section must contain either 'model_name' (text) or both 'text_model' and 'vision_model' (multimodal)."
        )

    # Normalize data paths relative to project root
    data = cfg["data"]
    for key in ("train_csv", "val_csv", "test_csv"):
        if key in data and isinstance(data[key], str):
            data[key] = str((PROJECT_ROOT / data[key]).resolve())

    # Coerce common training types (handles quoted YAML values like "2e-5")
    tr = cfg.get("training", {})
    def _coerce(key: str, typ):
        if key in tr and tr[key] is not None:
            try:
                tr[key] = typ(tr[key])
            except Exception:
                raise TypeError(f"training.{key} must be {typ.__name__}, got {tr[key]!r}")

    _coerce("learning_rate", float)
    _coerce("weight_decay", float)
    _coerce("batch_size", int)
    _coerce("num_epochs", int)
    _coerce("seed", int)

    # Coerce model.max_length if present
    model = cfg.get("model", {})
    if "max_length" in model and model["max_length"] is not None:
        try:
            model["max_length"] = int(model["max_length"])
        except Exception:
            raise TypeError(f"model.max_length must be int, got {model['max_length']!r}")


def merge_config_and_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Override config values with CLI arguments if provided."""
    if getattr(args, "learning_rate", None) is not None:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate

    if getattr(args, "batch_size", None) is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    if getattr(args, "num_epochs", None) is not None:
        config.setdefault("training", {})["num_epochs"] = args.num_epochs

    return config
