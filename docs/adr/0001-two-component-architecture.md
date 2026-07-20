# ADR 0001: Two-component architecture (generator + predictor) around one Record contract

- Status: accepted
- Date: 2026-07-20
- Deciders: DeepBrain Research

## Context

MAESTRO must predict neurostimulation outcomes for a specific brain, but real labelled data is
scarce, slow, and sensitive. We need to bootstrap from simulation, improve as real data arrives,
and let several people work in parallel.

## Decision

Model MAESTRO as two components that co-evolve:
1. a flow-matching **generator** that samples plausible outcomes and acts as an improving data
   engine, and
2. a neurostimulation **foundation model** (predictor) trained on generated + real data.

Couple them only through a single data contract, the `(brain, stimulation) -> outcome` `Record`
defined in `src/maestro/data/schema.py`. Simulators produce Records; the generator learns and
samples them; the predictor consumes them; eval and interp read predictions and activations.

## Consequences

- Teams can work independently behind the contract.
- The pipeline runs anywhere via a dependency-free analytic simulator and optional torch.
- We must guard the contract carefully; schema changes require an ADR and a coordinated migration.
- VERTEX and other engines are swappable without touching model code.
