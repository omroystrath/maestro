"""The ``maestro`` command line interface.

Subcommands mirror the pipeline:

    maestro simulate   # run a simulator -> write Records to disk
    maestro gen train  # train the flow-matching generator
    maestro gen sample # sample synthetic Records from the generator
    maestro fm train   # train the neurostimulation foundation model (the predictor)
    maestro eval       # score the predictor against baselines
    maestro info       # print environment / component availability

Everything is config-driven (YAML under configs/). This file stays thin: it parses args, loads
config, and calls into the library. No science lives here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maestro import __version__
from maestro.utils.config import load_config
from maestro.utils.logging import get_logger
from maestro.utils.seed import seed_everything

log = get_logger("maestro.cli")


def _cmd_info(_: argparse.Namespace) -> int:
    from maestro.simulators.analytic import AnalyticSimulator
    from maestro.simulators.vertex_bridge import VertexSimulator

    print(f"MAESTRO {__version__}")
    try:
        import torch

        print(f"torch: {torch.__version__} (cuda={torch.cuda.is_available()})")
    except ImportError:
        print("torch: not installed (install '.[train]' to train)")
    print(f"analytic simulator: available={AnalyticSimulator().available()}")
    print(f"vertex simulator:   available={VertexSimulator().available()}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from maestro.data.io import save_records
    from maestro.simulators.analytic import AnalyticSimulator
    from maestro.simulators.base import SimSpec
    from maestro.simulators.vertex_bridge import VertexSimulator

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 0))

    engine = cfg.get("engine", "analytic")
    sim = VertexSimulator() if engine == "vertex" else AnalyticSimulator()
    if engine == "vertex" and not sim.available():
        log.warning("VERTEX unavailable; falling back to analytic simulator.")
        sim = AnalyticSimulator()

    spec = SimSpec(
        n_samples=cfg.get("n_samples", 100),
        seed=cfg.get("seed", 0),
        params=cfg.get("params", {}),
    )
    out = args.out or cfg.get("out", "data/interim/sim")
    n = save_records(sim.simulate(spec), out)
    log.info("Wrote %d records to %s", n, out)
    return 0


def _cmd_gen_train(args: argparse.Namespace) -> int:
    from maestro.data.io import load_records
    from maestro.data.tensors import (
        EncoderConfig,
        InMemoryDataset,
        SimpleConditionEncoder,
    )
    from maestro.generator.flow_matching import FlowMatchingGenerator, GeneratorConfig

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 0))
    records = list(load_records(cfg["data"]))
    ds = InMemoryDataset(records, seed=cfg.get("seed", 0))
    enc = SimpleConditionEncoder(EncoderConfig(n_regions=cfg["n_regions"]))

    gen = FlowMatchingGenerator(
        GeneratorConfig(
            response_dim=ds.response_dim,
            cond_dim=enc.cond_dim,
            **cfg.get("generator", {}),
        )
    )
    gen.fit(ds, enc)
    out = cfg.get("out", "experiments/generator.pt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    gen.save(out)
    log.info("Saved generator to %s", out)
    return 0


def _cmd_fm_train(args: argparse.Namespace) -> int:
    from maestro.data.io import load_records
    from maestro.data.tensors import (
        EncoderConfig,
        InMemoryDataset,
        SimpleConditionEncoder,
    )
    from maestro.foundation.model import FoundationConfig, NeurostimFoundationModel

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 0))
    records = list(load_records(cfg["data"]))
    ds = InMemoryDataset(records, seed=cfg.get("seed", 0))
    enc = SimpleConditionEncoder(EncoderConfig(n_regions=cfg["n_regions"]))

    model = NeurostimFoundationModel(
        FoundationConfig(
            cond_dim=enc.cond_dim,
            response_dim=ds.response_dim,
            **cfg.get("foundation", {}),
        )
    )
    model.fit(ds, enc)
    out = cfg.get("out", "experiments/foundation.pt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    log.info("Saved foundation model to %s", out)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from maestro.data.io import load_records
    from maestro.data.tensors import (
        EncoderConfig,
        InMemoryDataset,
        SimpleConditionEncoder,
    )
    from maestro.eval.metrics import atlas_baseline, prediction_scores
    from maestro.foundation.model import FoundationConfig, NeurostimFoundationModel

    cfg = load_config(args.config)
    records = list(load_records(cfg["data"]))
    ds = InMemoryDataset(records, seed=cfg.get("seed", 0))
    enc = SimpleConditionEncoder(EncoderConfig(n_regions=cfg["n_regions"]))

    model = NeurostimFoundationModel(
        FoundationConfig(cond_dim=enc.cond_dim, response_dim=ds.response_dim)
    )
    model.load(cfg["checkpoint"])

    batch = ds.sample(min(len(ds), cfg.get("eval_n", 512)))
    y_true = batch["response"]
    y_pred = model.predict(enc.encode(batch))
    y_base = atlas_baseline(ds.all_responses()).repeat(len(y_true), axis=0)

    scores = prediction_scores(y_true, y_pred, y_baseline=y_base)
    print("MSE          :", round(scores.mse, 6))
    print("MAE          :", round(scores.mae, 6))
    print("R^2          :", round(scores.r2, 4))
    print(
        "skill vs atlas:",
        None if scores.skill_vs_atlas is None else round(scores.skill_vs_atlas, 4),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maestro", description="MAESTRO research CLI")
    p.add_argument("--version", action="version", version=f"maestro {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("info", help="print environment and component availability")
    sp.set_defaults(func=_cmd_info)

    sp = sub.add_parser("simulate", help="run a simulator and write records")
    sp.add_argument("--config", required=True)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=_cmd_simulate)

    gen = sub.add_parser("gen", help="flow-matching generator")
    gensub = gen.add_subparsers(dest="gen_command", required=True)
    g = gensub.add_parser("train", help="train the generator")
    g.add_argument("--config", required=True)
    g.set_defaults(func=_cmd_gen_train)

    fm = sub.add_parser("fm", help="neurostimulation foundation model")
    fmsub = fm.add_subparsers(dest="fm_command", required=True)
    f = fmsub.add_parser("train", help="train the predictor")
    f.add_argument("--config", required=True)
    f.set_defaults(func=_cmd_fm_train)

    sp = sub.add_parser("eval", help="score the predictor against baselines")
    sp.add_argument("--config", required=True)
    sp.set_defaults(func=_cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
