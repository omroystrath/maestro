"""Optional-torch pattern.

The MAESTRO package imports cleanly without torch installed (so the schema, simulators, CLI
plumbing, and tests that do not need training all work on a bare environment). Modules that need
torch call ``require_torch(...)`` at the point of use, which gives a clear, actionable error if
torch is missing instead of an ImportError at package import time.
"""

from __future__ import annotations

from types import ModuleType


def require_torch(who: str) -> ModuleType:
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"{who} needs PyTorch, which is not installed. "
            "Install the training extras:  pip install -e '.[train]'"
        ) from e
