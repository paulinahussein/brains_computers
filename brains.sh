#!/bin/bash
set -eo pipefail

SCRIPT="/nethome/phussein/brains/black_figure2_reproduction/analysis/reproduce_figure2_llama3.py"
OUTPUT="/nethome/phussein/brains/black_figure2_reproduction/rerun_results_llama4_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Figure 2: BERT + Llama 3 + Llama 4"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Output: ${OUTPUT}"
echo "========================================"

source /nethome/phussein/miniconda3/etc/profile.d/conda.sh
conda activate llama-brains

mkdir -p "${OUTPUT}"

python "${SCRIPT}" run \
    --backend torch_cuda \
    --solver kernel \
    --output "${OUTPUT}"

echo
echo "========================================"
echo "Finished successfully: $(date)"
echo "Output: ${OUTPUT}"
echo "========================================"

# #!/bin/bash
# set -eo pipefail

# #SCRIPT="/nethome/phussein/brains/bert-brains/extract_black_llama4_embeddings.py"
# SCRIPT="/nethome/phussein/brains/bert-brains/extract_black_llama4_transformations.py"
# echo "========================================"
# echo "Llama 4 Scout - Black transformations"
# echo "Host: $(hostname)"
# echo "Started: $(date)"
# echo "Script: ${SCRIPT}"
# echo "========================================"

# source /nethome/phussein/miniconda3/etc/profile.d/conda.sh
# conda activate llama-brains
# export HF_TOKEN="hf_


# # IMPORTANT: do not use home directory for the huge Llama 4 cache
# export HF_HOME="/scratch/phussein/huggingface"
# export HF_HUB_CACHE="${HF_HOME}/hub"

# mkdir -p "${HF_HUB_CACHE}"

# echo
# echo "Python:"
# which python
# python --version

# echo
# echo "CUDA:"
# python - <<'PY'
# import torch
# import transformers

# print("torch:", torch.__version__)
# print("transformers:", transformers.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Visible GPUs:", torch.cuda.device_count())

# for i in range(torch.cuda.device_count()):
#     p = torch.cuda.get_device_properties(i)
#     print(
#         f"GPU {i}: {torch.cuda.get_device_name(i)} | "
#         f"{p.total_memory / 1024**3:.1f} GB"
#     )
# PY

# echo
# echo "HF cache:"
# echo "${HF_HOME}"

# echo
# echo "Starting Llama 4 embedding extraction..."

# python "${SCRIPT}"

# echo
# echo "========================================"
# echo "Finished successfully: $(date)"
# echo "========================================"
