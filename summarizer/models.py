"""Unified model registry with lazy loading and GPU auto-detection."""

import os
import yaml
import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    PegasusForConditionalGeneration,
)

# Map model key prefixes to their HuggingFace model class
_MODEL_CLASS_MAP = {
    "bart": BartForConditionalGeneration,
    "t5": T5ForConditionalGeneration,
    "pegasus": PegasusForConditionalGeneration,
    "mt5": T5ForConditionalGeneration,  # mT5 shares T5 architecture
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.yaml")


def _load_config() -> dict:
    """Load model configuration from YAML."""
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


@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """
    Lazily load and cache a (tokenizer, model) tuple.
    Uses @st.cache_resource so each model is loaded only once per session.
    Works uniformly for T5, BART, PEGASUS, and mT5 architectures.
    """
    cfg = get_model_config(model_key)
    hf_name = cfg["hf_name"]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # Pick the right model class based on model key
    model_cls = None
    for prefix, cls in _MODEL_CLASS_MAP.items():
        if prefix in model_key:
            model_cls = cls
            break
    if model_cls is None:
        raise ValueError(f"No model class mapped for key: {model_key}")

    model = model_cls.from_pretrained(hf_name)
    model = model.to(device)
    model.eval()

    return tokenizer, model


def get_weights_config() -> dict:
    """Return metric weights from config."""
    return _load_config().get("weights", {})
