# Contributing to MAESTRO

This is the DeepBrain research team's working repository. The aim is to move fast without the code
turning to mud. A few conventions keep it that way.

## Setup

```bash
make setup            # venv + editable install with dev tools
source .venv/bin/activate
make install-train    # only if you need to train (installs torch)
```

## The one rule that matters most

**The data contract in `src/maestro/data/schema.py` is sacred.** Everything reads and writes the
`(brain, stimulation) -> outcome` `Record`. If you need to change it, open an ADR in `docs/adr/`
first (copy `0000-template.md`), get a second pair of eyes, then change the schema and migrate the
loaders in the same PR. Don't route around it with side channels.

## Branches and PRs

- Branch off `develop`. `main` is always releasable.
- Name branches `type/short-description`, e.g. `feat/graph-cond-encoder`, `fix/vertex-mat-load`.
- Keep PRs focused. A reviewer should be able to hold the whole change in their head.
- Every PR must pass CI: `ruff`, `black --check`, `mypy`, and `pytest`.

Run everything locally before pushing:

```bash
make fmt      # auto-format
make lint     # ruff + black check
make type     # mypy
make test     # pytest
```

## Where code goes

| You are adding... | Put it in... |
| ----------------- | ------------ |
| a new simulation engine | `src/maestro/simulators/` (subclass `Simulator`) |
| a change to the generator | `src/maestro/generator/` |
| a change to the predictor | `src/maestro/foundation/` |
| interpretability tooling | `src/maestro/interp/` |
| a metric or baseline | `src/maestro/eval/` |
| a new pipeline step for the CLI | `src/maestro/cli/main.py` (thin) + library code |
| a full experiment | `experiments/<name>/` with its own config and README |

Keep the CLI thin: it parses args and calls the library. No science in `cli/`.

## Tests

- Unit tests next to the thing they test under `tests/unit/`.
- Anything that spans components goes in `tests/integration/`.
- Tests must run without a GPU and without MATLAB (use the analytic simulator). Gate anything that
  needs torch behind a skip if torch is absent.

## Style

- Type hints on public functions. `mypy` is advisory-strict (missing imports ignored).
- Docstrings explain *why*, not *what the code obviously does*.
- Prefer plain numpy in the schema and data layers; keep torch inside `generator/`,
  `foundation/`, and `interp/` behind the `require_torch` helper.

## VERTEX

VERTEX is vendored under a non-commercial licence. Don't edit files under `simulators/vertex/`
except inside `simulators/vertex/deepbrain/` (our bridge/driver code). Record any change in
`simulators/vertex/CHANGES-DEEPBRAIN.md`. See `NOTICE`.
