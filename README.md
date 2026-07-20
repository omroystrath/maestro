# MAESTRO

**M**odelling **A**nticipated **E**ffects of **S**timulating **T**argeted **R**egions **O**ptimally

A foundation model that learns how a brain responds to stimulation, so we can predict
neurostimulation outcomes for a specific brain instead of guessing. This repository is where
the DeepBrain research team develops MAESTRO.

> The read side of neuroscience is being claimed. MAESTRO is our work on the write side:
> the control layer of the brain.


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
