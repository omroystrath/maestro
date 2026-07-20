# MAESTRO

**M**odelling **A**nticipated **E**ffects of **S**timulating **T**argeted **R**egions **O**ptimally

A foundation model that learns how a brain responds to stimulation, so we can predict
neurostimulation outcomes for a specific brain instead of guessing. This repository is where
the DeepBrain research team develops MAESTRO.

> The read side of neuroscience is being claimed. MAESTRO is our work on the write side:
> the control layer of the brain.

---

## The idea in one paragraph

MAESTRO has **two components that improve each other**:

1. A **flow-matching generator** that learns the distribution of brain responses to
   stimulation and produces training data on demand. It is seeded on biologically plausible
   simulated data and sharpened by real data over time.
2. A **neurostimulation foundation model** (the predictor). Trained on generated *and* real
   data, its job is to predict the outcome of stimulating any brain.

As data grows, the generator gets better, which makes the predictor better, which tells us
which data to gather next. Both climb their own scaling law at the same time. The whole point
is to solve the "every brain is different" problem and close the loop for scalable neuromodulation.

```
        physics + simulation (VERTEX/CANDO)
                    │
                    ▼
        ┌───────────────────────┐        feeds data        ┌──────────────────────────┐
        │  flow-matching         │ ───────────────────────▶ │  neurostimulation        │
        │  generator             │                          │  foundation model        │
        │  (improving sampler)   │ ◀─────────────────────── │  (predicts outcomes)     │
        └───────────────────────┘      sharpens targets     └──────────────────────────┘
                    ▲                                                     │
                    │                    real, consented data            │
                    └─────────────────────────────────────────── close the loop
```

## What lives where

```
maestro/
├── src/maestro/            # the MAESTRO Python package (our code)
│   ├── simulators/         # thin bridges to simulation engines (VERTEX, CANDO, analytic)
│   ├── data/               # data contracts, loaders, the (brain, stim, outcome) schema
│   ├── generator/          # the flow-matching data generator
│   ├── foundation/         # the neurostimulation foundation model (the predictor)
│   ├── interp/             # interpretability tooling (the "lens")
│   ├── eval/               # metrics, baselines, benchmarks
│   ├── cli/                # `maestro` command line entry points
│   └── utils/              # config, logging, seeding, io
├── simulators/vertex/      # VERTEX 2.0 (vendored MATLAB engine, see NOTICE + its own licence)
├── configs/                # Hydra/YAML configs for data, model, training
├── scripts/                # end-to-end pipeline scripts (simulate → train → eval)
├── experiments/            # experiment definitions and run outputs (gitignored heavy files)
├── notebooks/              # exploration, kept light
├── tests/                  # unit + integration tests
├── docs/                   # architecture, ADRs, onboarding
└── data/                   # raw / interim / processed / synthetic (gitignored)
```

The data contract that ties everything together is a single record:
**(brain, stimulation) → outcome**. It is defined once in
[`src/maestro/data/schema.py`](src/maestro/data/schema.py) and every component reads and writes
that shape. Get that contract right and the generator, the predictor, and the simulators can all
be developed independently.

## Quickstart

```bash
# 1. clone
git clone <your-remote-url> maestro && cd maestro

# 2. environment (Python 3.11+)
make setup            # creates a venv and installs the package in editable mode
source .venv/bin/activate

# 3. smoke test the pipeline on tiny synthetic data (no MATLAB needed)
maestro simulate --config configs/data/toy.yaml --out data/interim/toy
maestro gen train  --config configs/model/generator_small.yaml
maestro fm  train  --config configs/model/foundation_small.yaml
maestro eval        --config configs/train/eval_smoke.yaml
```

See [`docs/getting-started.md`](docs/getting-started.md) for the full walkthrough, and
[`docs/architecture/overview.md`](docs/architecture/overview.md) for how the pieces fit.

## The roadmap (gated by proof)

| Version | What it is | Gate |
| ------- | ---------- | ---- |
| **v0**  | Generator trained on simulated data; predictor beats atlas-only baselines | now, small compute |
| **v1**  | Population models; first real sessions flow in | compute grows |
| **v2**  | Per-patient targeting; consented outcomes refine protocols | data advantage |
| **v3**  | Closed-loop controller: predict, stimulate, sense, update | the general model |

We scale after the evidence, never before it. Each version ships something and earns the next.

## Working here

- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **How we make decisions:** architecture decision records in [`docs/adr/`](docs/adr/)
- **Code style:** `ruff` + `black` + `mypy`, enforced in CI. Run `make lint` before pushing.
- **Data governance:** nothing sensitive in git. See [`docs/data-governance.md`](docs/data-governance.md).

## A note on VERTEX

VERTEX 2.0 is a brain-tissue simulator from the University of Newcastle (Thornton, Hutchings,
Kaiser). We use it as one of our simulation engines to generate biologically plausible data. It is
**vendored under its own non-commercial licence** in [`simulators/vertex/`](simulators/vertex/)
and kept separate from MAESTRO's own code. See [`NOTICE`](NOTICE) before using it in any
product context. If you extend it, keep changes inside `simulators/vertex/` and document them in
`simulators/vertex/CHANGES-DEEPBRAIN.md`.

## Licence

MAESTRO's own code (everything under `src/`, `configs/`, `scripts/`, `tests/`) is released under
the licence in [`LICENSE`](LICENSE). Vendored components keep their original licences; see
[`NOTICE`](NOTICE).
