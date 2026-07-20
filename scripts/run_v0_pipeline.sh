#!/usr/bin/env bash
# End-to-end v0 pipeline on synthetic data. Requires the [train] extra (torch).
set -euo pipefail

echo ">> environment"
maestro info

echo ">> simulate"
maestro simulate --config configs/data/toy.yaml --out data/interim/toy

echo ">> train generator (component one)"
maestro gen train --config configs/model/generator_small.yaml

echo ">> train foundation model (component two)"
maestro fm train --config configs/model/foundation_small.yaml

echo ">> evaluate vs atlas baseline"
maestro eval --config configs/train/eval_smoke.yaml

echo ">> done. Positive 'skill vs atlas' means the model uses the specific brain."
