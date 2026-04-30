# BAMF-Eco: Budget-Aware Multi-Fidelity Eco-Optimizer

This repository provides code and instructions to reproduce the main experiments.

## Overview

BAMF-Eco is a constrained multi-objective Bayesian optimizer for eco-aware
object detection model selection. It combines multi-fidelity evaluation with
an acquisition function that integrates expected hypervolume improvement,
constraint-satisfaction probability, and evaluation energy cost. The
companion benchmark, EcoDetBench, evaluates 184 detector configurations
across accuracy, latency, energy, carbon, and water on a single GPU.

## Repository Structure

```
bamf_eco/                              Core library
  optimizer/                           BAMF-Eco optimizer, EFA acquisition,
                                       fidelity correction
  training/                            Training runner with energy monitoring
  benchmark/                           EcoDetBench inference benchmark
  measurement/                         GPU power monitoring (pynvml)
  sustainability/                      Carbon and water footprint accounting
  analysis/                            Result aggregation and figure generation
  baselines/                           Random Search, Hyperband, Single-Obj BO,
                                       Manual Expert
  utils/                               Configuration and helpers
configs/                               YAML experiment configurations
scripts/
  prepare_datasets.py                  COCO 2017 person-detection preparation
  run_promotion.py                     Phase 2 full-fidelity promotion driver
  run_analysis.py                      Aggregation, tables, and figures
  fill_paper_results.py                Update manuscript result fields
  export_correction_metrics_summary.py Export correction-quality JSON
  run_optimize.sh                      Phase 1 (low-fidelity exploration)
  run_promote.sh                       Phase 2 wrapper
  run_baselines.sh                     Baseline comparison wrapper
  run_benchmark.sh                     EcoDetBench inference sweep wrapper
tests/
  test_core.py                         Unit tests
requirements.txt                       Python dependencies
README.md                              This file
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify the install
pytest tests/test_core.py -v
```

A Python 3.11 environment is recommended. PyTorch with a matching CUDA
build is required for GPU energy measurement.

## Hardware Assumptions

- A single NVIDIA GPU with at least 16 GB VRAM (the paper uses V100-SXM2-32GB).
- 64 GB system RAM is sufficient for the configurations in this repository.
- Energy measurement requires NVIDIA Management Library access (`pynvml`).
- A CPU fallback monitor is used on systems without an NVIDIA GPU; in that
  case energy values are wall-clock estimates rather than measured power.

## Dataset Preparation

Raw and generated dataset files are not included in this anonymous
repository. Reviewers can prepare the COCO 2017 person-detection split
using the scripts in this repository.

1. Download the COCO 2017 validation set and annotations from
   https://cocodataset.org and place them under
   `./datasets/coco2017/`.
2. Convert to the YOLO person-detection format used in the paper:

   ```bash
   python scripts/prepare_datasets.py
   ```

After preparation the expected layout is:

```
datasets/
  coco2017/                Original COCO files (val2017/, annotations/)
  coco_person/
    coco_person.yaml       YOLO data config consumed by the runs
    images/                Filtered person images
    labels/                YOLO-format labels
```

Set `BAMF_ECO_DATASET` to override the dataset path used by the run
scripts:

```bash
export BAMF_ECO_DATASET=./datasets/coco_person/coco_person.yaml
```

## Quick Sanity Check

```bash
pytest tests/test_core.py -v
python -m bamf_eco.benchmark --config configs/smoke_test.yaml
```

The smoke test runs a small inference sweep and finishes in a few minutes
on a single GPU. It writes results to `./outputs/benchmark/`.

## Reproducing Paper Results

The full pipeline mirrors the experiments in the paper.

```bash
# 1. Phase 1: low-fidelity exploration (~2 GPU-hours)
bash scripts/run_optimize.sh

# 2. Phase 2: full-fidelity promotion of top candidates (~59 GPU-hours)
bash scripts/run_promote.sh

# 3. Baseline comparison (~90 GPU-hours total)
bash scripts/run_baselines.sh 42

# 4. EcoDetBench inference benchmark (~4 GPU-hours)
bash scripts/run_benchmark.sh

# 5. Aggregation, tables, and figures
python scripts/run_analysis.py --output-dir outputs/figures/

# 6. Optional: export the correction-metrics summary JSON
python scripts/export_correction_metrics_summary.py
```

Approximate total compute: 155 GPU-hours on a single V100. Each of the
phase scripts supports checkpoint and resume; rerun the same script to
continue an interrupted run.

## Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Fidelity levels | {0.05, 0.1, 0.2, 0.5, 1.0} | Fraction of full training epochs |
| GP kernel | Matern 5/2 | Surrogate model kernel |
| Initial random evals | 10 | Random exploration before BO |
| Promotion schedule | Every 5th, 10th, 20th iter | Successive halving |
| Eco-constraints | E ≤ 0.05 kWh, τ ≤ 100 ms | Energy and latency budgets |
| Stagnation patience | 10 | Consecutive non-improving promotions |

## Headline Results (from the paper)

| Optimizer | Evals | Best mAP | Pareto size | HV | Energy (kWh) |
|-----------|-------|----------|-------------|----|--------------|
| BAMF-Eco | 228 | 0.891 | 6 | 1.28 | 13.40 |
| Random Search | 50 | 0.734 | 8 | 1.09 | 6.61 |
| Hyperband | 50 | 0.651 | 6 | 0.97 | 0.44 |
| Single-Obj (Acc.) | 50 | 0.854 | – | – | 8.13 |
| Single-Obj (Eff.) | 50 | 0.724 | – | – | 1.42 |
| Manual Expert | 6 | 0.774 | 2 | 1.09 | 1.14 |

Fidelity-correction quality on 30 matched configurations:
Spearman ρ = 0.92, Kendall τ = 0.82, R² = 0.95 on mAP.

## Energy Measurement

GPU power is sampled via `pynvml` at 100 ms intervals. Total energy is
computed by numerical integration: `E = sum(P_i * dt)`. The reported
deviation against `nvidia-smi` logs is below 2%. CO₂e and water are
derived from energy using regional grid factors (defaults: 0.79 kg
CO₂e / kWh and 3.5 L / kWh). See `bamf_eco/sustainability/__init__.py`
for available regions and their factor ranges.

## Troubleshooting

- `pynvml` import fails: ensure NVIDIA drivers and `nvidia-ml-py` are
  installed; the package falls back to CPU monitoring otherwise.
- Out-of-memory during full-fidelity training: reduce batch size or
  image size in the relevant YAML config.
- Dataset path not found: confirm `BAMF_ECO_DATASET` points at the
  generated `coco_person.yaml` (or that the default
  `./datasets/coco_person/coco_person.yaml` exists).
- Outputs directory is writable: by default outputs go to `./outputs/`.
  Override with `BAMF_ECO_OUTPUTS_ROOT`.

## Licence

This code will be released under the MIT licence upon acceptance.
Model weights are sourced from the Ultralytics model hub (AGPL-3.0).
The COCO 2017 dataset is used under CC-BY-4.0.
