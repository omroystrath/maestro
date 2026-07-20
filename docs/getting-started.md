# Getting started

This walks you from a fresh clone to a trained v0 predictor scored against a baseline, all on
synthetic data so you need nothing but Python.

## 1. Install

```bash
make setup
source .venv/bin/activate
make install-train      # installs torch, needed for training
```

Check what's available:

```bash
maestro info
```

You'll see the analytic simulator is available and VERTEX is not (unless you have MATLAB). That's
fine, the analytic engine is enough to exercise the whole pipeline.

## 2. Generate data

```bash
maestro simulate --config configs/data/toy.yaml --out data/interim/toy
```

This writes ~2000 `(brain, stimulation) -> outcome` records as `.npz` shards. Open
`configs/data/toy.yaml` to see the knobs. To use real simulated data instead, point at
`configs/data/vertex_singlepulse.yaml` (needs MATLAB; it falls back to analytic otherwise).

## 3. Train the two components

The flow-matching **generator** (component one):

```bash
maestro gen train --config configs/model/generator_small.yaml
```

The neurostimulation **foundation model** / predictor (component two):

```bash
maestro fm train --config configs/model/foundation_small.yaml
```

## 4. Evaluate against the baseline

```bash
maestro eval --config configs/train/eval_smoke.yaml
```

You'll get MSE, MAE, R², and **skill vs atlas**. The atlas baseline is just the population-average
outcome, so positive skill means the model is using the *specific brain*, which is the whole point.
On the toy data a correctly wired model should show clearly positive skill.

## 5. Look inside (interpretability)

See `notebooks/` for a short example of pulling activations out of the predictor
(`model.features(...)`), training a sparse autoencoder on them (`maestro.interp.lens`), and running
gradient attribution back to the conditioning inputs. This is the in-house version of the lens we
plan to extend with Goodfire.

## Next

- Swap the analytic engine for VERTEX to get biologically plausible data.
- Replace the toy encoders in `data/tensors.py` with graph/sequence encoders over the connectome
  and EEG.
- Grow the configs from `_small` to real sizes as compute allows.

Read `docs/architecture/overview.md` for how it all fits, and `docs/adr/` for the decisions behind
the structure.
