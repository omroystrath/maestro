"""Evaluation metrics and baselines for MAESTRO.

Two things matter for v0:
  1. Does the foundation model predict outcomes better than naive baselines
     (atlas-only, fixed-parameter)? See :func:`prediction_scores` and the baselines below.
  2. Does the flow-matching generator produce realistic responses? See :func:`generator_scores`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionScores:
    mse: float
    mae: float
    r2: float
    # skill relative to a baseline: 1 - mse/mse_baseline (higher is better, 0 = no better)
    skill_vs_atlas: float | None = None


def prediction_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray | None = None,
) -> PredictionScores:
    """Standard regression metrics plus skill over an optional baseline."""
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(y_true)) or 1e-12
    r2 = 1.0 - mse / var
    skill = None
    if y_baseline is not None:
        base_mse = float(np.mean((y_baseline - y_true) ** 2)) or 1e-12
        skill = 1.0 - mse / base_mse
    return PredictionScores(mse=mse, mae=mae, r2=r2, skill_vs_atlas=skill)


# --------------------------------------------------------------------- baselines


def atlas_baseline(y_train: np.ndarray) -> np.ndarray:
    """The population-average outcome: what an atlas-only approach effectively predicts.

    Returns the mean training response, to be broadcast over the eval set. Beating this is the
    minimum bar for v0: it shows the model is using the *specific brain*, not just the average.
    """
    return y_train.mean(axis=0, keepdims=True)


def fixed_parameter_baseline(y_train: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    """A fixed-parameter clinician-style baseline: a single canned response profile."""
    return np.quantile(y_train, quantile, axis=0, keepdims=True)


# ------------------------------------------------------------- generator quality


def generator_scores(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """Cheap distribution-match diagnostics between real and generated responses.

    Not a substitute for a proper two-sample test, but enough to catch a generator that has
    collapsed or drifted. Compares per-dimension means/stds and overall energy distance-ish gap.
    """
    m_gap = float(np.mean(np.abs(real.mean(0) - synthetic.mean(0))))
    s_gap = float(np.mean(np.abs(real.std(0) - synthetic.std(0))))
    energy = float(
        np.abs(np.linalg.norm(real, axis=1).mean() - np.linalg.norm(synthetic, axis=1).mean())
    )
    return {"mean_gap": m_gap, "std_gap": s_gap, "energy_gap": energy}
