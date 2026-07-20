"""Simulator interface.

A simulator turns a specification into one or more :class:`~maestro.data.schema.Record`
objects. Concrete simulators (VERTEX, CANDO, analytic fields) implement :meth:`simulate`.
Keeping this interface tiny means the generator and training code never need to know which
engine produced a record.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from maestro.data.schema import Record


@dataclass
class SimSpec:
    """A request to a simulator.

    ``params`` is engine-specific and validated by the concrete simulator, not here. The common
    fields cover what every engine needs: how many samples, a seed for reproducibility, and a
    free-form parameter bag.
    """

    n_samples: int = 1
    seed: int | None = None
    params: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


class Simulator(abc.ABC):
    """Base class for all MAESTRO simulation engines."""

    name: str = "base"

    @abc.abstractmethod
    def simulate(self, spec: SimSpec) -> Iterator[Record]:
        """Yield records for the given specification.

        Implementations should stream records (yield as produced) so large runs do not have to
        fit in memory, and should stamp ``Source.SIMULATED`` on every record they emit.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """Return True if the engine can actually run in this environment.

        Used by the CLI and tests to skip engines whose backends (e.g. MATLAB) are absent.
        """
        return True
