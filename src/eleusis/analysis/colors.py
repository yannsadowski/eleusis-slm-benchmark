"""Color scheme utilities for analysis plots."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Fallback color for unknown models
DEFAULT_COLOR = "#888888"

# Providers that indicate open-source models
OPEN_PROVIDERS = {"huggingface"}


def load_model_metadata() -> dict[str, dict]:
    """Load model metadata from models.yaml.

    Returns dict mapping model_key to {color, is_open, provider}.
    """
    models_path = Path(__file__).parent.parent.parent.parent / "models.yaml"
    if not models_path.exists():
        logger.warning(f"models.yaml not found at {models_path}")
        return {}

    with open(models_path) as f:
        models = yaml.safe_load(f)

    metadata = {}
    for model_key, config in models.items():
        if isinstance(config, dict):
            provider = config.get("provider", "")
            metadata[model_key] = {
                "color": config.get("color", DEFAULT_COLOR),
                "is_open": provider in OPEN_PROVIDERS,
                "provider": provider,
                "parameters_b": config.get("parameters_b"),
                "publish_date": config.get("publish_date"),
            }
    return metadata


def load_model_colors() -> dict[str, str]:
    """Load model colors from models.yaml."""
    models_path = Path(__file__).parent.parent.parent.parent / "models.yaml"
    if not models_path.exists():
        logger.warning(f"models.yaml not found at {models_path}")
        return {}

    with open(models_path) as f:
        models = yaml.safe_load(f)

    colors = {}
    for model_key, config in models.items():
        if isinstance(config, dict) and "color" in config:
            colors[model_key] = config["color"]
    return colors


def normalize_model_name(name: str) -> str:
    """Normalize model name for matching (lowercase, strip whitespace, replace spaces)."""
    return name.lower().strip().replace(" ", "-").replace("_", "-")


def get_model_color(model_name: str, model_colors: dict[str, str]) -> str:
    """Get color for a model name with fuzzy matching.

    Tries exact match first, then normalized match, then substring match.
    """
    # Exact match
    if model_name in model_colors:
        return model_colors[model_name]

    # Normalized exact match
    normalized = normalize_model_name(model_name)
    for key, color in model_colors.items():
        if normalize_model_name(key) == normalized:
            return color

    # Substring match (model name contains key or vice versa)
    for key, color in model_colors.items():
        norm_key = normalize_model_name(key)
        if norm_key in normalized or normalized in norm_key:
            return color

    logger.debug(f"No color found for model '{model_name}', using default")
    return DEFAULT_COLOR


def get_color_map(model_names: list[str], model_colors: dict[str, str]) -> dict[str, str]:
    """Build color map for a list of model names."""
    return {name: get_model_color(name, model_colors) for name in model_names}
