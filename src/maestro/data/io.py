"""Persist and load MAESTRO Records.

We store records as compressed ``.npz`` shards plus a small JSON sidecar for the non-array
metadata. This keeps arrays efficient while staying dependency-light and inspectable. For larger
scale, migrate to a columnar format (parquet/arrow) behind these same two functions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
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


def save_records(records: Iterable[Record], out_dir: str | Path, shard_size: int = 1000) -> int:
    """Write records to ``out_dir`` as npz shards. Returns the number written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    count, shard, buffer = 0, 0, []
    for rec in records:
        buffer.append(rec)
        count += 1
        if len(buffer) >= shard_size:
            _write_shard(out, shard, buffer)
            shard += 1
            buffer = []
    if buffer:
        _write_shard(out, shard, buffer)
    (out / "manifest.json").write_text(json.dumps({"n_records": count, "shard_size": shard_size}))
    return count


def load_records(in_dir: str | Path) -> Iterator[Record]:
    """Yield records from all shards in ``in_dir``."""
    src = Path(in_dir)
    for meta_file in sorted(src.glob("shard_*.json")):
        arrs = np.load(meta_file.with_suffix(".npz"), allow_pickle=False)
        metas = json.loads(meta_file.read_text())
        for i, m in enumerate(metas):
            yield _rebuild(arrs, i, m)


def _write_shard(out: Path, shard: int, recs: list[Record]) -> None:
    arrays: dict[str, np.ndarray] = {}
    metas: list[dict] = []
    for i, r in enumerate(recs):
        if r.brain.connectome is not None:
            arrays[f"conn_{i}"] = r.brain.connectome
        if r.outcome.response is not None:
            arrays[f"resp_{i}"] = r.outcome.response
        metas.append(
            {
                "subject_id": r.brain.subject_id,
                "region_labels": r.brain.region_labels,
                "modality": r.stimulation.modality.value,
                "target_region": r.stimulation.target_region,
                "intensity": r.stimulation.intensity,
                "duration_ms": r.stimulation.duration_ms,
                "on_times_ms": r.stimulation.on_times_ms,
                "off_times_ms": r.stimulation.off_times_ms,
                "markers": r.outcome.markers,
                "source": r.source.value,
                "has_conn": r.brain.connectome is not None,
                "has_resp": r.outcome.response is not None,
            }
        )
    np.savez_compressed(out / f"shard_{shard:05d}.npz", **arrays)
    (out / f"shard_{shard:05d}.json").write_text(json.dumps(metas))


def _rebuild(arrs, i: int, m: dict) -> Record:
    brain = Brain(
        subject_id=m["subject_id"],
        connectome=arrs[f"conn_{i}"] if m.get("has_conn") else None,
        region_labels=m.get("region_labels"),
    )
    stim = Stimulation(
        modality=Modality(m["modality"]),
        target_region=m.get("target_region"),
        intensity=m.get("intensity"),
        duration_ms=m.get("duration_ms"),
        on_times_ms=m.get("on_times_ms", []),
        off_times_ms=m.get("off_times_ms", []),
    )
    outcome = Outcome(
        response=arrs[f"resp_{i}"] if m.get("has_resp") else None,
        markers=m.get("markers", {}),
    )
    return Record(brain=brain, stimulation=stim, outcome=outcome, source=Source(m["source"]))
