"""The MAESTRO data contract.

Every component in MAESTRO reads and writes the same record: a single labelled example of
``(brain, stimulation) -> outcome``. Simulators produce it, the generator learns to sample it,
the foundation model learns to predict the ``outcome`` half from the ``(brain, stimulation)``
half, and evaluation compares predictions against it.

Keep this file small and stable. If the contract changes, it changes here first, with an ADR in
``docs/adr/`` explaining why. Everything downstream depends on this shape.

The dataclasses below are deliberately framework-agnostic (plain Python + numpy). Conversion to
tensors happens in ``maestro.data.tensors`` so the schema stays free of any ML dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

SCHEMA_VERSION = "0.1.0"


class Modality(StrEnum):
    """Non-invasive stimulation modalities MAESTRO targets."""

    FUS = "focused_ultrasound"
    TMS = "transcranial_magnetic"
    TDCS = "transcranial_direct_current"
    TIS = "temporal_interference"
    # simulation-only modality used while bootstrapping v0 on VERTEX-style field stimulation
    EFIELD = "electric_field"


class Source(StrEnum):
    """Where a record came from. Drives train/eval splits and governance."""

    SIMULATED = "simulated"  # from VERTEX / CANDO / analytic fields
    SYNTHETIC = "synthetic"  # sampled from the flow-matching generator
    REAL = "real"  # a consented human/animal session


@dataclass
class Brain:
    """The subject side of a record: what we start with.

    All arrays are plain numpy so this stays serialisable and ML-framework-free.
    Fields may be ``None`` when a given data source does not provide them; loaders are
    responsible for imputation / masking policy, not the schema.
    """

    subject_id: str
    # structural: region x region connectivity (e.g. from dMRI / a parcellation)
    connectome: np.ndarray | None = None  # shape (R, R)
    region_labels: list[str] | None = None  # length R
    # functional baseline: channels x time
    baseline_eeg: np.ndarray | None = None  # shape (C, T)
    # geometry needed for field propagation (skull thickness map, tissue segmentation id, ...)
    geometry: dict[str, np.ndarray] = field(default_factory=dict)
    # coarse, non-identifying context
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stimulation:
    """The action side of a record: what we do to the brain."""

    modality: Modality
    # target in the same coordinate frame as the brain geometry / parcellation
    target_xyz: np.ndarray | None = None  # shape (3,)
    target_region: str | None = None
    montage: dict[str, Any] = field(default_factory=dict)
    # dose parameters; keep flexible across modalities
    intensity: float | None = None
    frequency_hz: float | None = None
    duration_ms: float | None = None
    # on/off schedule for multi-pulse protocols, milliseconds since t0
    on_times_ms: list[float] = field(default_factory=list)
    off_times_ms: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    """The label side of a record: what comes back."""

    # deep-brain / network change estimate, e.g. per-region activity delta
    response: np.ndarray | None = None  # shape (R,) or (R, T)
    # pre/post markers of interest (band power, connectivity metrics, ...)
    markers: dict[str, float] = field(default_factory=dict)
    # consented clinical/behavioural readout, if any
    clinical: dict[str, Any] = field(default_factory=dict)
    adverse_events: list[str] = field(default_factory=list)
    # provenance for the label (simulator settings, session id hash, ...)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    """A single (brain, stimulation) -> outcome example, the atom of MAESTRO."""

    brain: Brain
    stimulation: Stimulation
    outcome: Outcome
    source: Source
    schema_version: str = SCHEMA_VERSION

    def is_labelled(self) -> bool:
        """True if there is any outcome signal to learn from."""
        return self.outcome.response is not None or bool(self.outcome.markers)

    def validate(self) -> None:
        """Cheap structural checks. Raises ValueError on contract violations."""
        b, s = self.brain, self.stimulation
        if b.connectome is not None:
            r, c = b.connectome.shape
            if r != c:
                raise ValueError(f"connectome must be square, got {b.connectome.shape}")
            if b.region_labels is not None and len(b.region_labels) != r:
                raise ValueError("region_labels length must match connectome size")
        if s.on_times_ms and s.off_times_ms and len(s.on_times_ms) != len(s.off_times_ms):
            raise ValueError("on_times_ms and off_times_ms must be the same length")
        if self.outcome.response is not None and b.connectome is not None:
            r = b.connectome.shape[0]
            if self.outcome.response.shape[0] != r:
                raise ValueError(
                    "outcome.response first dim must match number of regions "
                    f"({self.outcome.response.shape[0]} != {r})"
                )


__all__ = [
    "SCHEMA_VERSION",
    "Modality",
    "Source",
    "Brain",
    "Stimulation",
    "Outcome",
    "Record",
]
