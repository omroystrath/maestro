from maestro.data.io import load_records, save_records
from maestro.simulators.analytic import AnalyticSimulator
from maestro.simulators.base import SimSpec


def test_save_load_roundtrip(tmp_path):
    recs = list(
        AnalyticSimulator().simulate(SimSpec(n_samples=20, seed=0, params={"n_regions": 8}))
    )
    n = save_records(recs, tmp_path, shard_size=7)
    assert n == 20
    loaded = list(load_records(tmp_path))
    assert len(loaded) == 20
    assert loaded[0].outcome.response.shape == recs[0].outcome.response.shape
