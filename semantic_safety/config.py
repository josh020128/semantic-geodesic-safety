"""Load config from YAML and env."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML; path defaults to config/default.yaml next to package."""
    if path is None:
        base = Path(__file__).resolve().parent.parent
        path = base / "config" / "default.yaml"
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
