#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Starting Figure 3B: $(date)"
echo "Host: $(hostname)"
echo "========================================"

source /nethome/phussein/miniconda3/etc/profile.d/conda.sh
conda activate llama-brains

cd /nethome/phussein/brains/bert-brains

OUTPUT="/nethome/phussein/brains/bert-brains/results/figure3b_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT}"

python analysis/reproduce_figure3b.py \
    --backend torch_cuda \
    --n-iter 100 \
    --n-boot 10000 \
    --output "${OUTPUT}"

echo
echo "========================================"
echo "Finished successfully: $(date)"
echo "Output: ${OUTPUT}"
echo "========================================"