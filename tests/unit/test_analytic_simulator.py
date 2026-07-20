from maestro.data.schema import Source
from maestro.simulators.analytic import AnalyticSimulator
from maestro.simulators.base import SimSpec


def test_analytic_produces_valid_records():
    sim = AnalyticSimulator()
    recs = list(sim.simulate(SimSpec(n_samples=10, seed=1, params={"n_regions": 8})))
    assert len(recs) == 10
    for r in recs:
        r.validate()
        assert r.source == Source.SIMULATED
        assert r.outcome.response.shape[0] == 8


def test_analytic_is_reproducible():
    a = list(AnalyticSimulator().simulate(SimSpec(n_samples=5, seed=42, params={"n_regions": 8})))
    b = list(AnalyticSimulator().simulate(SimSpec(n_samples=5, seed=42, params={"n_regions": 8})))
    for ra, rb in zip(a, b, strict=True):
        assert (ra.outcome.response == rb.outcome.response).all()
