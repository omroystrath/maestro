"""The neurostimulation foundation model (component two): the predictor.

Given ``(brain, stimulation)``, predict the ``outcome``. Trained on generated data from the
flow-matching generator *and* on real sessions as they arrive. This is the model that has to
solve "every brain is different": it conditions on the specific brain, so its prediction is
personalised rather than a population average.

Design goals baked into the interface:
  - **interpretable by design**: exposes hidden representations via ``features()`` so the
    ``maestro.interp`` tools (the "lens") can read what drives a prediction.
  - **source-aware training**: can weight real vs synthetic vs simulated data differently.
  - **swappable backbone**: the backbone is a detail; the contract is (brain, stim) -> outcome.

Like the generator, this is a runnable scaffold with a small MLP backbone. Replace the backbone
with the real architecture without changing the public surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maestro.utils.logging import get_logger
from maestro.utils.torch_opt import require_torch

log = get_logger(__name__)


@dataclass
class FoundationConfig:
    cond_dim: int  # size of the (brain, stimulation) encoding
    response_dim: int  # size of the predicted outcome
    hidden: int = 512
    depth: int = 6
    lr: float = 3e-4
    steps: int = 20_000
    batch_size: int = 256
    # loss weights by data source; real data counts more once we have it
    weight_real: float = 3.0
    weight_synthetic: float = 1.0
    weight_simulated: float = 1.0
    seed: int = 0


class NeurostimFoundationModel:
    """Predicts stimulation outcomes for a specific brain."""

    def __init__(self, cfg: FoundationConfig) -> None:
        self.cfg = cfg
        self._net = None

    # ------------------------------------------------------------------ training

    def fit(self, dataset, cond_encoder) -> NeurostimFoundationModel:
        torch = require_torch("NeurostimFoundationModel.fit")
        self._build(torch)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.cfg.lr)

        log.info("Training foundation model for %d steps", self.cfg.steps)
        for step in range(self.cfg.steps):
            batch = dataset.sample(self.cfg.batch_size)
            cond = torch.as_tensor(cond_encoder.encode(batch), dtype=torch.float32)
            y = torch.as_tensor(np.asarray(batch["response"]), dtype=torch.float32)
            w = torch.as_tensor(self._weights(batch), dtype=torch.float32).unsqueeze(-1)

            pred, _ = self._net(cond)
            loss = (w * (pred - y) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, self.cfg.steps // 20) == 0:
                log.info("step %6d  loss %.5f", step, float(loss))
        return self

    # ------------------------------------------------------------------ inference

    def predict(self, cond: np.ndarray) -> np.ndarray:
        """Predict the outcome for encoded (brain, stimulation) conditions."""
        torch = require_torch("NeurostimFoundationModel.predict")
        self._build(torch)
        c = torch.as_tensor(cond, dtype=torch.float32)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        with torch.no_grad():
            pred, _ = self._net(c)
        return pred.cpu().numpy()

    def features(self, cond: np.ndarray) -> np.ndarray:
        """Return the penultimate hidden representation for interpretability."""
        torch = require_torch("NeurostimFoundationModel.features")
        self._build(torch)
        c = torch.as_tensor(cond, dtype=torch.float32)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        with torch.no_grad():
            _, feats = self._net(c)
        return feats.cpu().numpy()

    # ------------------------------------------------------------------ io

    def save(self, path: str) -> None:
        torch = require_torch("NeurostimFoundationModel.save")
        self._build(torch)
        torch.save({"cfg": self.cfg.__dict__, "state": self._net.state_dict()}, path)

    def load(self, path: str) -> NeurostimFoundationModel:
        torch = require_torch("NeurostimFoundationModel.load")
        self._build(torch)
        ckpt = torch.load(path, map_location="cpu")
        self._net.load_state_dict(ckpt["state"])
        return self

    # ------------------------------------------------------------------ internals

    def _build(self, torch) -> None:
        if self._net is not None:
            return
        torch.manual_seed(self.cfg.seed)
        self._net = _Backbone(self.cfg)

    def _weights(self, batch) -> np.ndarray:
        srcs = batch.get("source", None)
        if srcs is None:
            return np.ones(len(batch["response"]))
        table = {
            "real": self.cfg.weight_real,
            "synthetic": self.cfg.weight_synthetic,
            "simulated": self.cfg.weight_simulated,
        }
        return np.array([table.get(str(s), 1.0) for s in srcs], dtype=np.float32)


def _Backbone(cfg: FoundationConfig):  # noqa: N802
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(cfg.cond_dim, cfg.hidden), nn.SiLU()]
            for _ in range(cfg.depth - 2):
                layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU()]
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(cfg.hidden, cfg.response_dim)

        def forward(self, cond):
            feats = self.trunk(cond)
            return self.head(feats), feats

    return Net()
