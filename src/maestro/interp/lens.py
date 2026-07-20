"""The lens: interpretability tooling for the foundation model.

The thesis behind MAESTRO is that a model we can read is worth more than a black box that happens
to work. This module holds the tools that let us look inside the neurostimulation foundation model
and ask *which brain signals drive a prediction* and *what structure the model has learned*.

We mirror the approach Goodfire used on Prima Mente's Pleiades model: train a sparse autoencoder
(SAE) on the model's internal activations to pull out interpretable features, then trace
predictions back to those features. This file scaffolds:

  - ``SparseAutoencoder``  : learn an overcomplete, sparse feature basis over hidden activations
  - ``attribute``          : simple gradient x input attribution from output back to conditions

Roadmap: a formal collaboration with Goodfire to run their Ember tooling on MAESTRO. Until then
these give us a working in-house lens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maestro.utils.logging import get_logger
from maestro.utils.torch_opt import require_torch

log = get_logger(__name__)


@dataclass
class SAEConfig:
    d_hidden: int  # width of the activations we are explaining
    n_features: int  # overcomplete dictionary size (e.g. 8x d_hidden)
    l1: float = 1e-3  # sparsity penalty
    lr: float = 1e-3
    steps: int = 5_000
    seed: int = 0


class SparseAutoencoder:
    """Learns a sparse, overcomplete feature basis over a set of activations."""

    def __init__(self, cfg: SAEConfig) -> None:
        self.cfg = cfg
        self._net = None

    def fit(self, activations: np.ndarray) -> SparseAutoencoder:
        torch = require_torch("SparseAutoencoder.fit")
        self._build(torch)
        x = torch.as_tensor(activations, dtype=torch.float32)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg.lr)
        log.info("Training SAE: %d features over %d dims", self.cfg.n_features, self.cfg.d_hidden)
        for step in range(self.cfg.steps):
            idx = torch.randint(0, x.shape[0], (min(1024, x.shape[0]),))
            xb = x[idx]
            recon, feats = self._net(xb)
            loss = ((recon - xb) ** 2).mean() + self.cfg.l1 * feats.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % max(1, self.cfg.steps // 10) == 0:
                log.info("sae step %5d  loss %.5f", step, float(loss))
        return self

    def features(self, activations: np.ndarray) -> np.ndarray:
        """Return the sparse feature activations for a batch."""
        torch = require_torch("SparseAutoencoder.features")
        self._build(torch)
        x = torch.as_tensor(activations, dtype=torch.float32)
        with torch.no_grad():
            _, feats = self._net(x)
        return feats.cpu().numpy()

    def _build(self, torch) -> None:
        if self._net is not None:
            return
        torch.manual_seed(self.cfg.seed)
        import torch.nn as nn

        class SAE(nn.Module):
            def __init__(self, d, k):
                super().__init__()
                self.enc = nn.Linear(d, k)
                self.dec = nn.Linear(k, d, bias=False)
                self.act = nn.ReLU()

            def forward(self, x):
                f = self.act(self.enc(x))
                return self.dec(f), f

        self._net = SAE(self.cfg.d_hidden, self.cfg.n_features)


def attribute(model, cond: np.ndarray, target_index: int | None = None) -> np.ndarray:
    """Gradient x input attribution of one output over the conditioning inputs.

    Answers "which parts of (brain, stimulation) most drive this prediction". Returns an array the
    same shape as ``cond`` where larger magnitude means more influence on the chosen output.
    """
    torch = require_torch("interp.attribute")
    c = torch.as_tensor(cond, dtype=torch.float32, requires_grad=True)
    if c.ndim == 1:
        c = c.unsqueeze(0)
    pred, _ = model._net(c)  # noqa: SLF001  (interp is allowed to reach in)
    target = pred.sum() if target_index is None else pred[:, target_index].sum()
    target.backward()
    return (c.grad * c).detach().cpu().numpy()
