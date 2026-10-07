#!/bin/bash
set -eo pipefail

echo "========================================"
echo "Black Llama feature extraction"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "========================================"

source /nethome/phussein/miniconda3/etc/profile.d/conda.sh
conda activate llama-brains

cd /nethome/phussein/brains/bert-brains

python extract_black_llama_transformations.py

echo "========================================"
echo "Finished: $(date)"
echo "========================================"