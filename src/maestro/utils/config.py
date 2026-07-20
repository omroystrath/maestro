"""Tiny YAML config loader with environment-variable expansion.

We keep configuration in YAML under ``configs/`` and load it into plain dicts. Deliberately
minimal; if the project grows into needing sweeps and composition, migrate to Hydra without
changing call sites much (they already pass a dict around).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([^}^{]+)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, expanding ${ENV_VAR} references."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return _expand(data)
