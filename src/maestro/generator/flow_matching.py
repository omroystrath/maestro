"""The flow-matching data generator (component one).

This is the improving data engine. It learns the distribution of brain responses conditioned on
``(brain, stimulation)`` and can sample new, plausible responses on demand. Seeded on simulated
data, it is sharpened as real data arrives.

Flow matching learns a time-dependent velocity field ``v_theta(x, t, cond)`` that transports a
simple base distribution (Gaussian noise) to the data distribution along straight-ish paths. At
sample time we integrate an ODE from noise to a sample. This is the same family of methods behind
modern image and video generation; here the "pixels" are brain responses.

This module is a **scaffold**: the interfaces and training/sampling skeleton are here so the team
can plug in the real network and conditioning. It uses a light optional-torch pattern so importing
the package never hard-fails when torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from maestro.utils.logging import get_logger
from maestro.utils.torch_opt import require_torch

log = get_logger(__name__)


@dataclass
class GeneratorConfig:
    response_dim: int  # size of the response vector to generate
    cond_dim: int  # size of the (brain, stimulation) conditioning vector
    hidden: int = 256
    depth: int = 4
    sigma_min: float = 1e-4  # flow-matching noise floor
    lr: float = 2e-4
    steps: int = 10_000
    batch_size: int = 256
    sample_ode_steps: int = 50  # integration steps at sampling time
    seed: int = 0


class ConditionEncoder(Protocol):
    """Turns a batch of Records into a conditioning matrix (B, cond_dim)."""

    def encode(self, batch) -> np.ndarray: ...


class FlowMatchingGenerator:
    """Conditional flow-matching generator over brain responses.

    Public surface the rest of MAESTRO relies on:
      - ``fit(dataset, cond_encoder)`` : train the velocity field
      - ``sample(cond, n)``            : draw synthetic responses for given conditions
      - ``save(path)`` / ``load(path)``: checkpoint io
    """

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self._net = None  # built lazily so importing this module never needs torch

    # ------------------------------------------------------------------ training

    def fit(self, dataset, cond_encoder: ConditionEncoder) -> FlowMatchingGenerator:
        torch = require_torch("FlowMatchingGenerator.fit")
        self._build(torch)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.cfg.lr)

        log.info("Training flow-matching generator for %d steps", self.cfg.steps)
        for step in range(self.cfg.steps):
            x1, cond = self._next_batch(torch, dataset, cond_encoder)  # target sample + condition
            x0 = torch.randn_like(x1)  # base noise
            t = torch.rand(x1.shape[0], 1, device=x1.device)  # time in [0, 1]
            # straight-line interpolation path and its constant target velocity
            xt = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            pred_v = self._net(xt, t, cond)
            loss = ((pred_v - target_v) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % max(1, self.cfg.steps // 20) == 0:
                log.info("step %6d  loss %.5f", step, float(loss))
        return self

    # ------------------------------------------------------------------ sampling

    def sample(self, cond: np.ndarray, n: int | None = None) -> np.ndarray:
        """Integrate the learned ODE from noise to samples for the given conditions."""
        torch = require_torch("FlowMatchingGenerator.sample")
        self._build(torch)
        c = torch.as_tensor(cond, dtype=torch.float32)
        if c.ndim == 1:
            c = c.unsqueeze(0)
        b = n or c.shape[0]
        if c.shape[0] == 1 and b > 1:
            c = c.repeat(b, 1)

        x = torch.randn(b, self.cfg.response_dim)
        dt = 1.0 / self.cfg.sample_ode_steps
        with torch.no_grad():
            for k in range(self.cfg.sample_ode_steps):
                t = torch.full((b, 1), k * dt)
                x = x + self._net(x, t, c) * dt  # explicit Euler; swap for RK if needed
        return x.cpu().numpy()

    # ------------------------------------------------------------------ io

    def save(self, path: str) -> None:
        torch = require_torch("FlowMatchingGenerator.save")
        self._build(torch)
        torch.save({"cfg": self.cfg.__dict__, "state": self._net.state_dict()}, path)

    def load(self, path: str) -> FlowMatchingGenerator:
        torch = require_torch("FlowMatchingGenerator.load")
        self._build(torch)
        ckpt = torch.load(path, map_location="cpu")
        self._net.load_state_dict(ckpt["state"])
        return self

    # ------------------------------------------------------------------ internals

    def _build(self, torch) -> None:
        if self._net is not None:
            return
        torch.manual_seed(self.cfg.seed)
        self._net = _VelocityField(self.cfg).to("cpu")

    def _next_batch(self, torch, dataset, cond_encoder):
        """Return (x1, cond) tensors for one training step.

        Expects ``dataset`` to yield/return batches of Records or a torch Dataset that already
        provides (response, condition) pairs. This scaffold assumes the simple dict batch shape
        produced by ``maestro.data.tensors.to_flow_batch``.
        """
        batch = dataset.sample(self.cfg.batch_size) if hasattr(dataset, "sample") else dataset
        x1 = torch.as_tensor(np.asarray(batch["response"]), dtype=torch.float32)
        cond = torch.as_tensor(cond_encoder.encode(batch), dtype=torch.float32)
        return x1, cond


def _VelocityField(cfg: GeneratorConfig):  # noqa: N802  (factory reads like a class)
    """Build the MLP velocity field. Separated so torch import stays lazy."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_dim = cfg.response_dim + 1 + cfg.cond_dim  # x, t, cond
            layers: list[nn.Module] = [nn.Linear(in_dim, cfg.hidden), nn.SiLU()]
            for _ in range(cfg.depth - 1):
                layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU()]
            layers += [nn.Linear(cfg.hidden, cfg.response_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x, t, cond):
            return self.net(torch.cat([x, t, cond], dim=-1))

    return Net()
