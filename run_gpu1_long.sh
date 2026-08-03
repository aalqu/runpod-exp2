#!/usr/bin/env bash
# run_gpu1_long.sh — n_assets=5  (GPU 1)  — full training
# 300 iters, 256 paths, 64 steps per arch (~50x more than default).
set -euo pipefail
mkdir -p logs results/gpu1_long

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "============================================================"
echo "  GPU 1 (LONG)  |  n_assets=5  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    5 \
    --seeds       1,2,3 \
    --results-dir results/gpu1_long \
    --device      cuda \
    --long \
    --resume \
    2>&1 | tee logs/gpu1_long.log

echo ""
echo "GPU 1 (long) finished at $(date)"
echo "Results in: results/gpu1_long/"
