#!/usr/bin/env bash
# Phase 2: Promote top-k low-fidelity candidates to full fidelity.

set -euo pipefail

DATASET="${BAMF_ECO_DATASET:-./datasets/coco_person/coco_person.yaml}"

python -u scripts/run_promotion.py \
    --top-k 30 \
    --dataset "${DATASET}"
