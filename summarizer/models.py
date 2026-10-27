"""Unified model registry with lazy loading and GPU auto-detection."""

import os
import threading
from functools import lru_cache

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.yaml")

# Loaded (tokenizer, model) pairs, shared by every Streamlit session and API
# request in this process. The lock stops two sessions from loading the same
# multi-GB checkpoint at once.
_LOADED: dict = {}
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load model configuration from YAML (read once per process)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Auto-detect GPU, fallback to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_config(model_key: str) -> dict:
    """Return config dict for a specific model key."""
    cfg = _load_config()
    if model_key not in cfg["models"]:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(cfg['models'].keys())}")
    return cfg["models"][model_key]


def list_models() -> list[str]:
    """Return all available model keys."""
    return list(_load_config()["models"].keys())


def get_model_label(model_key: str) -> str:
    """Return the human-readable label for a model."""
    return get_model_config(model_key).get("label", model_key)


def get_model_color(model_key: str) -> str:
    """Return the color hex for a model."""
    return get_model_config(model_key).get("color", "#6B7280")


def restore_sinusoidal_positions(model) -> None:
    """Make sure fixed sinusoidal position tables (PEGASUS, MarianMT) are intact.

    These tables are not stored in the checkpoint. Some transformers 5.x
    releases left them randomly initialised on load ("embed_positions.weight
    | MISSING" in the load report), which turns PEGASUS output into gibberish.
    A correct table has sin(0) = 0 in its first half and cos(0) = 1 in its
    second half at position 0; only a table failing that check is rebuilt.
    """
    for module in model.modules():
        create = getattr(module, "create_weight", None)
        if not callable(create) or not isinstance(module, torch.nn.Embedding):
            continue
        row0 = module.weight[0].detach().float()
        half = (row0.numel() + 1) // 2
        if torch.allclose(row0[:half], torch.zeros(half), atol=1e-6) and \
                torch.allclose(row0[half:], torch.ones(row0.numel() - half), atol=1e-6):
            continue
        with torch.no_grad():
            module.weight.copy_(create().to(module.weight.dtype))


def load_model(model_key: str):
    """
    Lazily load and cache a (tokenizer, model) tuple.
    Each model is loaded once per process and reused by every caller.
    AutoModelForSeq2SeqLM picks the right class (BART, T5, PEGASUS, mT5)
    from the checkpoint's own config.
    """
    if model_key in _LOADED:
        return _LOADED[model_key]

    with _LOAD_LOCK:
        if model_key not in _LOADED:
            hf_name = get_model_config(model_key)["hf_name"]
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
            restore_sinusoidal_positions(model)
            model = model.to(get_device())
            model.eval()
            _LOADED[model_key] = (tokenizer, model)
    return _LOADED[model_key]


def get_weights_config() -> dict:
    """Return metric weights from config."""
    return _load_config().get("weights", {})
