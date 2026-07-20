"""Turning Records into the flat tensors the models train on.

The models work on two things:
  - a **condition** vector encoding ``(brain, stimulation)``
  - a **response** vector (the outcome to predict or generate)

This module defines a simple, explicit encoder for both and an in-memory dataset that yields
batches. It is intentionally basic; richer encoders (graph encoders over the connectome,
sequence encoders over EEG) can implement the same ``encode`` surface and be swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maestro.data.schema import Record


@dataclass
class EncoderConfig:
    n_regions: int
    # how many connectome features to keep (row-sums by default, cheap and permutation-stable)
    use_connectome_rowsums: bool = True
    include_intensity: bool = True


class SimpleConditionEncoder:
    """Encodes (brain, stimulation) into a fixed-length vector.

    Layout (concatenated):
      [ connectome_rowsums (n_regions) | target_onehot (n_regions) | intensity (1) ]
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg

    @property
    def cond_dim(self) -> int:
        d = 0
        if self.cfg.use_connectome_rowsums:
            d += self.cfg.n_regions
        d += self.cfg.n_regions  # target one-hot
        if self.cfg.include_intensity:
            d += 1
        return d

    def encode(self, batch: dict) -> np.ndarray:
        recs: list[Record] = batch["record"]
        rows = [self._encode_one(r) for r in recs]
        return np.stack(rows).astype(np.float32)

    def _encode_one(self, r: Record) -> np.ndarray:
        n = self.cfg.n_regions
        parts: list[np.ndarray] = []
        if self.cfg.use_connectome_rowsums:
            if r.brain.connectome is not None:
                parts.append(r.brain.connectome.sum(axis=1))
            else:
                parts.append(np.zeros(n))
        onehot = np.zeros(n)
        if r.stimulation.target_region and r.stimulation.target_region.startswith("R"):
            idx = int(r.stimulation.target_region[1:])
            if 0 <= idx < n:
                onehot[idx] = 1.0
        parts.append(onehot)
        if self.cfg.include_intensity:
            parts.append(np.array([r.stimulation.intensity or 0.0]))
        return np.concatenate(parts)


class InMemoryDataset:
    """Holds Records in memory and yields random batches as flat dicts.

    Batch dict shape used across the codebase::

        {"record": [Record, ...],
         "response": np.ndarray (B, response_dim),
         "source": [str, ...]}
    """

    def __init__(self, records: list[Record], seed: int = 0) -> None:
        self.records = [r for r in records if r.outcome.response is not None]
        self._rng = np.random.default_rng(seed)
        if not self.records:
            raise ValueError("InMemoryDataset needs at least one labelled record")
        self.response_dim = int(self.records[0].outcome.response.shape[0])

    def __len__(self) -> int:
        return len(self.records)

    def sample(self, batch_size: int) -> dict:
        idx = self._rng.integers(0, len(self.records), size=batch_size)
        recs = [self.records[i] for i in idx]
        resp = np.stack([r.outcome.response for r in recs]).astype(np.float32)
        srcs = [r.source.value for r in recs]
        return {"record": recs, "response": resp, "source": srcs}

    def all_responses(self) -> np.ndarray:
        return np.stack([r.outcome.response for r in self.records]).astype(np.float32)
