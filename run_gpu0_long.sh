#!/usr/bin/env bash
# run_gpu0_long.sh — n_assets=1  (GPU 0)  — full training
# 300 iters, 256 paths, 64 steps per arch (~50x more than default).
set -euo pipefail
mkdir -p logs results/gpu0_long

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "============================================================"
echo "  GPU 0 (LONG)  |  n_assets=1  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    1 \
    --seeds       1 \
    --results-dir results/gpu0_long \
    --device      cpu \
    --long \
    --resume \
    2>&1 | tee logs/gpu0_long.log

echo ""
echo "GPU 0 (long) finished at $(date)"
echo "Results in: results/gpu0_long/"
