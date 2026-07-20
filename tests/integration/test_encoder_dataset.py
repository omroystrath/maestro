from maestro.data.tensors import EncoderConfig, InMemoryDataset, SimpleConditionEncoder


def test_encoder_dimensions(toy_records):
    ds = InMemoryDataset(toy_records)
    enc = SimpleConditionEncoder(EncoderConfig(n_regions=16))
    batch = ds.sample(8)
    cond = enc.encode(batch)
    assert cond.shape == (8, enc.cond_dim)
    assert batch["response"].shape[0] == 8
