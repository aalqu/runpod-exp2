#!/usr/bin/env bash
# run_gpu1.sh — n_assets=5  (GPU 1)
# 5-asset universe: IVV, QQQ, TLT, GLD, VNQ.
# Medium job (~2-3h). FD HJB is solved on a 5D numerical grid.
set -euo pipefail
mkdir -p logs results/gpu1

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "============================================================"
echo "  GPU 1  |  n_assets=5  |  $(date)"
echo "============================================================"

python3 run_experiment.py \
    --n-assets    5 \
    --seeds       1,2,3 \
    --results-dir results/gpu1 \
    --device      cuda \
    --resume \
    2>&1 | tee logs/gpu1.log

echo ""
echo "GPU 1 finished at $(date)"
echo "Results in: results/gpu1/"
