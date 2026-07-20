import pytest

from maestro.simulators.analytic import AnalyticSimulator
from maestro.simulators.base import SimSpec


@pytest.fixture
def toy_records():
    sim = AnalyticSimulator()
    spec = SimSpec(n_samples=64, seed=0, params={"n_regions": 16})
    return list(sim.simulate(spec))
