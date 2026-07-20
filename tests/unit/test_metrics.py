import numpy as np

from maestro.eval.metrics import atlas_baseline, prediction_scores


def test_perfect_prediction_scores():
    y = np.random.RandomState(0).randn(50, 8)
    s = prediction_scores(y, y.copy())
    assert s.mse < 1e-12
    assert s.r2 > 0.999


def test_skill_vs_baseline_positive_when_better():
    y = np.random.RandomState(0).randn(50, 8)
    base = atlas_baseline(y).repeat(50, axis=0)
    pred = y + 0.01 * np.random.RandomState(1).randn(*y.shape)
    s = prediction_scores(y, pred, y_baseline=base)
    assert s.skill_vs_atlas is not None and s.skill_vs_atlas > 0
