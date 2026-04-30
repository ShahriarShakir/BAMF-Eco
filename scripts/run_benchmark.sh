#!/usr/bin/env bash
# Run the EcoDetBench inference benchmark across all configurations.

set -euo pipefail

OUTDIR="${BAMF_ECO_OUTPUTS_ROOT:-./outputs}/benchmark"
mkdir -p "${OUTDIR}"

python -u -c "
from bamf_eco.benchmark import InferenceBenchmarkRunner
from bamf_eco.utils import save_results
from bamf_eco.sustainability import SustainabilityAccountant

accountant = SustainabilityAccountant(carbon_region='default')
runner = InferenceBenchmarkRunner(
    sustainability_accountant=accountant,
    verbose=True,
)

results = runner.run_sweep(
    model_names=None,
    precisions=['fp32', 'fp16'],
    image_sizes=[320, 416, 512, 640],
    warmup_iters=50,
    benchmark_iters=200,
    device='cuda:0',
    seed=42,
    checkpoint_path='${OUTDIR}/sweep_checkpoint.json',
)
save_results(results, '${OUTDIR}/ecodetbench_results.json')
print(f'Benchmark complete: {len(results)} configurations evaluated')
"
