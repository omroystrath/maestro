# Changelog

All notable changes to MAESTRO are recorded here. Format loosely follows Keep a Changelog.

## [0.1.0] - 2026-07-20
### Added
- Initial repository scaffold for the DeepBrain MAESTRO programme.
- Core `(brain, stimulation) -> outcome` data contract (`maestro.data.schema`).
- Simulator interface with analytic (dev) and VERTEX (MATLAB) engines.
- Flow-matching generator scaffold (component one).
- Neurostimulation foundation model scaffold (component two, the predictor).
- Interpretability lens (SAE + attribution) scaffold.
- Evaluation metrics and atlas / fixed-parameter baselines.
- `maestro` CLI: simulate, gen train, fm train, eval, info.
- Configs, tests, CI, docs, ADRs, data-governance policy.
- Vendored VERTEX 2.0 under its non-commercial licence with a clean adapter boundary.
