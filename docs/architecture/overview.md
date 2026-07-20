# Architecture overview

MAESTRO is a small number of components with one shared data contract between them. The contract is
what lets the team work on the generator, the predictor, the simulators, and interpretability in
parallel without stepping on each other.

## The contract

Everything is a `Record`: `(brain, stimulation) -> outcome` (`src/maestro/data/schema.py`).

- **brain**: connectome, baseline EEG, geometry, coarse meta. What we start with.
- **stimulation**: modality, target, dose, on/off schedule. What we do.
- **outcome**: response over regions, markers, consented clinical readout. What comes back.
- **source**: `simulated` (VERTEX/CANDO), `synthetic` (from our generator), or `real`.

Producers write Records; consumers read them. That's the whole coupling.

## The two components

```
                    ┌─────────────────────────────────────────────┐
   simulators/      │              flow-matching generator         │
   (VERTEX, CANDO,  │   learns p(outcome | brain, stimulation)     │
    analytic)  ────▶│   and samples synthetic Records on demand    │
        │           └───────────────┬─────────────────────────────┘
        │  simulated Records         │ synthetic Records
        │                            ▼
        │           ┌─────────────────────────────────────────────┐
        └──────────▶│        neurostimulation foundation model      │
   real Records     │   predicts outcome from (brain, stimulation)  │
   (later)          │   interpretable by design (features())        │
                    └───────────────┬─────────────────────────────┘
                                    │ predictions, hidden features
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │   eval (baselines, skill)   interp (lens)     │
                    └─────────────────────────────────────────────┘
```

**Generator** (`src/maestro/generator/flow_matching.py`). Conditional flow matching: learn a
velocity field that transports noise to responses along straight paths, conditioned on
`(brain, stimulation)`. Same method family as image/video generation, re-aimed at brain responses.
It is the improving data engine.

**Predictor** (`src/maestro/foundation/model.py`). Given `(brain, stimulation)`, predict the
outcome. Trained on simulated + synthetic + real data, with per-source loss weights so real data
counts more once we have it. Exposes `features()` so it can be read.

Both climb their own scaling law as data grows, and each improves the other: better generator ->
more/better training data -> better predictor -> better sense of which real data to collect ->
better generator. That loop is the thesis.

## Simulators

`src/maestro/simulators/` holds thin bridges behind one `Simulator` interface:

- `analytic.py`: dependency-free toy engine so the pipeline runs anywhere (development only).
- `vertex_bridge.py`: runs the vendored VERTEX MATLAB engine and maps outputs to Records.
- (future) `cando.py`, analytic-field engines, etc.

Swapping engines never touches the generator or predictor because both only see Records.

## Interpretability (the lens)

`src/maestro/interp/lens.py`: a sparse autoencoder over the predictor's activations plus gradient
attribution, mirroring the Goodfire approach used on Prima Mente's Pleiades. The predictor is built
to be read (`features()`), not just to be right.

## Why this shape

- **Parallel work**: the contract decouples teams.
- **Runnable anywhere**: analytic engine + optional-torch means CI and laptops work with no GPU or
  MATLAB.
- **Swap, don't rewrite**: engines, encoders, and backbones are replaceable behind stable surfaces.
- **Governance-friendly**: `source` on every record and a hard rule that nothing sensitive enters
  git (see `docs/data-governance.md`).
