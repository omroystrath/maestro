"""A tiny analytic simulator so the whole MAESTRO pipeline is runnable without MATLAB.

This is NOT a scientific model. It produces plausibly-shaped ``(brain, stimulation) -> outcome``
records using a simple linear-response-plus-noise rule over a random connectome. Its only job is
to let the team develop and test the generator, the foundation model, the training loop, and the
CLI end to end on any laptop. Swap it for :class:`VertexSimulator` for real simulated data.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from maestro.data.schema import (
    Brain,
    Modality,
    Outcome,
    Record,
    Source,
    Stimulation,
)
from maestro.simulators.base import SimSpec, Simulator


class AnalyticSimulator(Simulator):
    """Linear-response toy simulator over a random small-world-ish connectome."""

    name = "analytic"

    def simulate(self, spec: SimSpec) -> Iterator[Record]:
        rng = np.random.default_rng(spec.seed)
        n_regions = int(spec.params.get("n_regions", 32))
        noise = float(spec.params.get("noise", 0.05))

        # one fixed "population" connectome; each subject perturbs it slightly
        base_w = _random_connectome(rng, n_regions)

        for i in range(spec.n_samples):
            subj_w = base_w + rng.normal(0, 0.02, size=base_w.shape)
            subj_w = np.clip(subj_w, 0, None)

            target = rng.integers(0, n_regions)
            intensity = float(rng.uniform(0.5, 1.5))

            # stimulation as a one-hot drive at the target region
            drive = np.zeros(n_regions)
            drive[target] = intensity

            # linear spread through the connectome + saturating nonlinearity + noise
            spread = subj_w @ drive
            response = np.tanh(spread) + rng.normal(0, noise, size=n_regions)

            brain = Brain(
                subject_id=f"analytic-{i:06d}",
                connectome=subj_w,
                region_labels=[f"R{j}" for j in range(n_regions)],
                meta={"engine": "analytic"},
            )
            stim = Stimulation(
                modality=Modality.EFIELD,
                target_region=f"R{target}",
                intensity=intensity,
                duration_ms=500.0,
                on_times_ms=[1500.0],
                off_times_ms=[1500.5],
            )
            outcome = Outcome(
                response=response,
                markers={"peak": float(response.max()), "l2": float(np.linalg.norm(response))},
                provenance={"seed": spec.seed, "index": i},
            )
            rec = Record(brain=brain, stimulation=stim, outcome=outcome, source=Source.SIMULATED)
            rec.validate()
            yield rec


def _random_connectome(rng: np.random.Generator, n: int) -> np.ndarray:
    w = rng.gamma(shape=1.5, scale=0.15, size=(n, n))
    np.fill_diagonal(w, 0.0)
    # symmetrise a bit to look connectome-like
    return 0.5 * (w + w.T)
