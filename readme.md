# Transformer Representations and Neural Language Processing

This repository contains the code used to reproduce and extend selected analyses from
Kumar et al. (2024), *Shared functional specialization in transformer-based language
models and the human brain*.

The project focuses on the **"I Knew You Were Black"** dataset and compares neural
predictivity of internal representations from:

- BERT-base
- Llama 3.2 3B
- Llama 4 Scout

The main analyses reproduce and extend **Figure 2** and **Figure 3B** from Kumar et al.
(2024).

---

## Project structure

```text
bert-brains/
├── analysis/
│   ├── banded_ridge_regression.py
│   └── reproduce_figure3b.py
│
├── text_processing_and_transformers/
│   ├── ...
│   ├── llama_black/
│   └── llama4_black/
│
├── prepare_black_layerwise_features.py
│
├── run_figure3b.sh
├── figure3b.sub
│
└── results/
    └── figure3b_*/


black_figure2_reproduction/
├── analysis/
│   ├── encoding_backend.py
│   ├── reproduce_figure2.py
│   ├── reproduce_figure2_llama3.py
│   └── reproduce_figure2_llama4.py
│
├── save_data/
│   ├── X/
│   └── Y/
│
└── Parcel2ROI_new.npy

## Workflow on LST clusters

### Figure 2 
1. create model activations (BERT, Llama 3, Llama 4)
e.g. for Llama 4: embeddings and transformations
brains.sh
# SCRIPT="/nethome/phussein/brains/bert-brains/extract_black_llama4_embeddings.py"
# SCRIPT="/nethome/phussein/brains/bert-brains/extract_black_llama4_transformations.py"

2. Regression model & plot Figure 2
brains.sh
INPUT="/nethome/phussein/brains/black_figure2_reproduction/analysis/reproduce_figure2_llama3.py"
OUTPUT="/nethome/phussein/brains/black_figure2_reproduction/rerun_results_llama4_$(date +%Y%m%d_%H%M%S)"


### Figure 3b 
3. extract model activation
extract_black_llama.sh
#embeddings & transformations
extract_black_llama.py // extract_black_llama_transformations.py 

4.reproduce figure 3b
run_figure3b.sh
input: reproduce_figure3b.py
OUTPUT="/nethome/phussein/brains/bert-brains/results/figure3b_$(date +%Y%m%d_%H%M%S)"


This readme was generated using AI.