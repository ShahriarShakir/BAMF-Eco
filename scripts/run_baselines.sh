#!/usr/bin/env bash
# Run baseline comparison: random, hyperband, single-objective BO (acc/eff), manual expert.

set -euo pipefail

DATASET="${BAMF_ECO_DATASET:-./datasets/coco_person/coco_person.yaml}"
OUTDIR="${BAMF_ECO_OUTPUTS_ROOT:-./outputs}/baselines"
SEED="${1:-42}"
mkdir -p "${OUTDIR}"

python -u -c "
import json, os
from bamf_eco.baselines import get_baseline
from bamf_eco.training import TrainingRunner
from bamf_eco.sustainability import SustainabilityAccountant

accountant = SustainabilityAccountant(carbon_region='default')
evaluator = TrainingRunner(sustainability_accountant=accountant)

baseline_names = [
    'random',
    'hyperband',
    'single_obj_accuracy',
    'single_obj_efficiency',
    'manual_expert',
]

results = {}
for name in baseline_names:
    print(f'Running {name} (seed=${SEED})')
    baseline = get_baseline(name, max_evaluations=50, seed=${SEED})
    res = baseline.run(
        evaluator,
        dataset_path='${DATASET}',
        checkpoint_path=f'${OUTDIR}/{name}_seed${SEED}.json',
    )
    results[name] = {
        'n_evaluations': res.n_evaluations,
        'total_energy_kwh': res.total_energy_kwh,
        'total_co2e_kg': res.total_co2e_kg,
        'best_map': res.best_map,
        'best_energy': res.best_energy,
    }

with open('${OUTDIR}/baselines_seed${SEED}.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved baseline results')
"
