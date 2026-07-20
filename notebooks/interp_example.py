# %% [markdown]
# # Interpretability example: reading the predictor
# Pull activations from the foundation model, train a sparse autoencoder on them, and attribute a
# prediction back to the (brain, stimulation) inputs. Run after training a foundation model.

# %%
import numpy as np

from maestro.data.io import load_records
from maestro.data.tensors import EncoderConfig, InMemoryDataset, SimpleConditionEncoder
from maestro.foundation.model import FoundationConfig, NeurostimFoundationModel
from maestro.interp.lens import SAEConfig, SparseAutoencoder, attribute

# %%
records = list(load_records("data/interim/toy"))
ds = InMemoryDataset(records)
enc = SimpleConditionEncoder(EncoderConfig(n_regions=32))
model = NeurostimFoundationModel(FoundationConfig(cond_dim=enc.cond_dim, response_dim=ds.response_dim))
model.load("experiments/foundation_small.pt")

# %%
batch = ds.sample(512)
cond = enc.encode(batch)
acts = model.features(cond)          # hidden activations to explain
print("activations:", acts.shape)

# %%
sae = SparseAutoencoder(SAEConfig(d_hidden=acts.shape[1], n_features=acts.shape[1] * 8))
sae.fit(acts)
feats = sae.features(acts)
print("mean active features per example:", (feats > 0).sum(1).mean())

# %%
attr = attribute(model, cond[0], target_index=None)   # which inputs drove this prediction
print("top input dims by |attribution|:", np.argsort(-np.abs(attr).ravel())[:10])
