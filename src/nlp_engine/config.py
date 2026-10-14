"""
Configuration loader and path constants for nlp_engine.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

# Anchor paths relative to repository root
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent

CONFIGS_DIR = ROOT_DIR / "configs"
CONFIG_PATH = CONFIGS_DIR / "config.yaml"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_configs.yaml"

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TOKENIZERS_DIR = ARTIFACTS_DIR / "tokenizers"
LOGS_DIR = ARTIFACTS_DIR / "logs"


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_project_config() -> Dict[str, Any]:
    """Retrieve global project configuration."""
    return load_yaml(CONFIG_PATH)


def get_model_config() -> Dict[str, Any]:
    """Retrieve model hyperparameters configuration."""
    return load_yaml(MODEL_CONFIG_PATH)
