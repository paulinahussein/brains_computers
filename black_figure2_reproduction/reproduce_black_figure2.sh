#!/bin/bash
set -eo pipefail

BASE="/nethome/phussein/brains/black_figure2_reproduction"
OUTPUT_ROOT="/scratch/phussein/logs/brains/black_figure2_reproduction"
OUTPUT_DIR="${OUTPUT_ROOT}/rerun_results"

echo "=================================================="
echo "Black Figure 2 reproduction"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Base: ${BASE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "=================================================="

cd "${BASE}"

# --------------------------------------------------
# Initialize Conda
# --------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"

# --------------------------------------------------
# Create environment if necessary
# --------------------------------------------------

if conda env list | grep -q "^black-figure2 "; then
    echo "Environment black-figure2 already exists."
else
    echo "Creating black-figure2 environment..."
    conda create -n black-figure2 python=3.12 -y
fi

conda activate black-figure2

echo
echo "Python:"
which python
python --version

# --------------------------------------------------
# Install dependencies
# --------------------------------------------------


echo "Installing requirements..."
python -m pip install -r requirements.txt
python -m pip install --upgrade numpy

python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
echo
echo "Installing CUDA PyTorch..."

python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# --------------------------------------------------
# Verify environment
# --------------------------------------------------

echo
echo "Checking environment..."

python - <<'PY'
import himalaya
import numpy
import scipy
import sklearn
import matplotlib
import nibabel
import nilearn
import torch

print("All imports OK")
print("himalaya:", himalaya.__version__)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this node")

print("GPU:", torch.cuda.get_device_name(0))
PY

# --------------------------------------------------
# Prepare output directory
# --------------------------------------------------

mkdir -p "${OUTPUT_ROOT}"

if [ -e "${OUTPUT_DIR}" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OUTPUT_ROOT}/rerun_results_${TIMESTAMP}"
fi

echo
echo "Results will be written to:"
echo "${OUTPUT_DIR}"

# --------------------------------------------------
# Thread settings
# --------------------------------------------------

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

# --------------------------------------------------
# Audit
# --------------------------------------------------

echo
echo "=================================================="
echo "Running input audit"
echo "=================================================="

python analysis/reproduce_figure2.py audit

# --------------------------------------------------
# Full reproduction
# --------------------------------------------------

echo
echo "=================================================="
echo "Starting full Figure 2 reproduction"
echo "=================================================="

python analysis/reproduce_figure2.py run \
    --backend torch_cuda \
    --output "${OUTPUT_DIR}"

# --------------------------------------------------
# Render Figure 2
# --------------------------------------------------

echo
echo "=================================================="
echo "Rendering Figure 2"
echo "=================================================="

python analysis/render_figure2.py \
    --results "${OUTPUT_DIR}"

# --------------------------------------------------
# Done
# --------------------------------------------------

echo
echo "=================================================="
echo "Reproduction finished successfully"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "=================================================="