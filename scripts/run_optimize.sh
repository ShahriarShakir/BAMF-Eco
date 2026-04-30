#!/usr/bin/env bash
# Phase 1: Low-fidelity exploration with the BAMF-Eco optimizer.
# Supports checkpoint/resume; rerun the same command to continue.

set -euo pipefail

DATASET="${BAMF_ECO_DATASET:-./datasets/coco_person/coco_person.yaml}"
OUTDIR="${BAMF_ECO_OUTPUTS_ROOT:-./outputs}/bamf_eco_main"
mkdir -p "${OUTDIR}"

python -u -c "
from bamf_eco.optimizer import BAMFEcoOptimizer
from bamf_eco.optimizer.acquisition import EcoConstraints
from bamf_eco.training import TrainingRunner
from bamf_eco.sustainability import SustainabilityAccountant

accountant = SustainabilityAccountant(carbon_region='default')
evaluator = TrainingRunner(
    sustainability_accountant=accountant,
    power_sample_interval_ms=100,
)

optimizer = BAMFEcoOptimizer(
    objectives=['mAP', 'energy_kwh', 'latency_ms'],
    directions=['maximize', 'minimize', 'minimize'],
    reference_point={'mAP': 0.0, 'energy_kwh': 0.1, 'latency_ms': 500.0},
    constraints=EcoConstraints(
        max_energy_kwh=0.05,
        max_co2e_kg=0.5,
        max_latency_ms=100.0,
    ),
    fidelity_levels=[0.05, 0.1, 0.2, 0.5, 1.0],
    eta=3,
    max_evaluations=200,
    early_termination=True,
    patience=15,
    min_evaluations=30,
    n_initial_random=10,
    seed=42,
    output_dir='${OUTDIR}',
)

state = optimizer.run(
    evaluator,
    dataset_path='${DATASET}',
    resume=True,
)
print(f'Iterations: {state.iteration}, Pareto size: {state.pareto_size}, '
      f'Energy: {state.total_energy_kwh:.4f} kWh, Best mAP: {state.best_map:.4f}')
"
