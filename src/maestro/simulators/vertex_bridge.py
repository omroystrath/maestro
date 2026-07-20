"""Bridge to the vendored VERTEX 2.0 MATLAB simulator.

VERTEX ("Virtual Electrode Recording Tool for EXtracellular Potentials") simulates a patch of
brain tissue and the effect of electric-field stimulation on it, including short-term plasticity
and STDP. We use it as one engine that produces biologically plausible ``(brain, stimulation) ->
outcome`` records to seed the flow-matching generator.

Reference: Thornton, Hutchings & Kaiser, *Wellcome Open Research* 2019 (VERTEX 2.0).
The MATLAB source lives under ``simulators/vertex/`` under its own non-commercial licence.

This bridge is intentionally thin. It:
  1. writes a spec to a temp workspace,
  2. invokes MATLAB (or Octave) to run a VERTEX driver script that writes ``.mat`` outputs,
  3. loads those outputs and maps them onto MAESTRO :class:`Record` objects.

If MATLAB is not available, :meth:`available` returns False and the CLI falls back to the
analytic engine, so the pipeline is runnable on any laptop for development.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

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
from maestro.utils.logging import get_logger

log = get_logger(__name__)

# location of the vendored engine relative to the repo root
VERTEX_ROOT = Path(__file__).resolve().parents[3] / "simulators" / "vertex"
# the MATLAB entry point we call; see simulators/vertex/deepbrain/run_vertex_export.m
VERTEX_DRIVER = "run_vertex_export"


def _find_matlab() -> str | None:
    """Return a runnable MATLAB/Octave command, or None if neither is installed."""
    for exe in ("matlab", "octave-cli", "octave"):
        if shutil.which(exe):
            return exe
    return os.environ.get("MAESTRO_MATLAB")  # explicit override


class VertexSimulator(Simulator):
    """Runs VERTEX and returns MAESTRO records."""

    name = "vertex"

    def __init__(self, vertex_root: Path | None = None, matlab: str | None = None) -> None:
        self.vertex_root = Path(vertex_root) if vertex_root else VERTEX_ROOT
        self.matlab = matlab or _find_matlab()

    def available(self) -> bool:
        ok = self.matlab is not None and self.vertex_root.exists()
        if not ok:
            log.info(
                "VERTEX unavailable (matlab=%s, root_exists=%s); "
                "the analytic engine will be used instead.",
                self.matlab,
                self.vertex_root.exists(),
            )
        return ok

    def simulate(self, spec: SimSpec) -> Iterator[Record]:
        if not self.available():
            raise RuntimeError(
                "VERTEX is not available. Install MATLAB (or set MAESTRO_MATLAB), or use the "
                "analytic simulator for development."
            )

        with tempfile.TemporaryDirectory(prefix="maestro_vertex_") as tmp:
            workdir = Path(tmp)
            self._write_spec(workdir, spec)
            self._run_matlab(workdir, spec)
            yield from self._load_outputs(workdir, spec)

    # ------------------------------------------------------------------ internals

    def _write_spec(self, workdir: Path, spec: SimSpec) -> None:
        """Serialise the spec to a .mat-friendly JSON the driver reads."""
        import json

        payload = {
            "n_samples": spec.n_samples,
            "seed": spec.seed if spec.seed is not None else 0,
            # e.g. amplitude_mV, pulse_width_ms, inter_pulse_interval_ms, protocol, density_scale
            "params": spec.params,
        }
        (workdir / "spec.json").write_text(json.dumps(payload, indent=2))

    def _run_matlab(self, workdir: Path, spec: SimSpec) -> None:
        """Invoke the VERTEX driver. It reads spec.json and writes records_*.mat."""
        assert self.matlab is not None
        driver_dir = self.vertex_root / "deepbrain"
        # build a small MATLAB command that adds paths and runs the driver
        mcode = (
            f"addpath(genpath('{self.vertex_root}'));"
            f"addpath('{driver_dir}');"
            f"{VERTEX_DRIVER}('{workdir}');"
            f"exit;"
        )
        if "octave" in self.matlab:
            cmd = [self.matlab, "--eval", mcode]
        else:
            cmd = [self.matlab, "-batch", mcode]
        log.info("Running VERTEX: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(workdir))

    def _load_outputs(self, workdir: Path, spec: SimSpec) -> Iterator[Record]:
        """Load records_*.mat produced by the driver into MAESTRO Records."""
        try:
            from scipy.io import loadmat
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("scipy is required to read VERTEX .mat outputs") from e

        out_files = sorted(workdir.glob("records_*.mat"))
        if not out_files:
            raise RuntimeError(f"VERTEX produced no outputs in {workdir}")

        for f in out_files:
            m = loadmat(f, squeeze_me=True, struct_as_record=False)
            rec = m["record"]
            brain = Brain(
                subject_id=str(getattr(rec, "subject_id", f.stem)),
                connectome=np.asarray(getattr(rec, "connectome", np.zeros((0, 0)))),
                region_labels=list(getattr(rec, "region_labels", []) or []),
                baseline_eeg=_opt_arr(getattr(rec, "baseline_lfp", None)),
                geometry={},
                meta={"engine": "vertex", "vertex_version": "2.0"},
            )
            stim = Stimulation(
                modality=Modality.EFIELD,
                intensity=float(getattr(rec, "amplitude_mV", np.nan)),
                duration_ms=float(getattr(rec, "duration_ms", np.nan)),
                on_times_ms=list(np.atleast_1d(getattr(rec, "on_times_ms", []))),
                off_times_ms=list(np.atleast_1d(getattr(rec, "off_times_ms", []))),
                params={"protocol": str(getattr(rec, "protocol", "single_pulse"))},
            )
            outcome = Outcome(
                response=_opt_arr(getattr(rec, "response", None)),
                markers=_to_marker_dict(getattr(rec, "markers", None)),
                provenance={"file": f.name, "seed": spec.seed},
            )
            record = Record(
                brain=brain,
                stimulation=stim,
                outcome=outcome,
                source=Source.SIMULATED,
            )
            record.validate()
            yield record


def _opt_arr(x) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    return arr if arr.size else None


def _to_marker_dict(x) -> dict[str, float]:
    if x is None:
        return {}
    # MATLAB structs come back as objects; be liberal in what we accept
    try:
        return {k: float(v) for k, v in vars(x).items()}
    except TypeError:
        return {}
