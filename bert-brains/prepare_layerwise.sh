#!/bin/bash
set -euo pipefail

source /nethome/phussein/miniconda3/etc/profile.d/conda.sh
conda activate llama-brains

cd /nethome/phussein/brains/bert-brains

python prepare_black_layerwise_features.py