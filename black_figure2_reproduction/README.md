# Figure 2 reproduction — I Knew You Were Black

This package reproduces the Figure 2 analysis for **45 participants** listening
to *I Knew You Were Black*, using the supplied TR-aligned parcel data. It contains
recomputed model results, figures, and an executable GPU/CPU analysis pipeline.
This is a single-dataset reproduction.

## View the results

- **`results/figure2_with_rois.pdf`** — figure with anatomical ROI insets.
- `results/figure2_with_rois.png` — image preview.
- `results/figure2_with_rois.svg` — editable vector figure.
- `results/figure2_summary.csv` — 50 model/ROI medians and 95% confidence intervals.
- `results/figure2_subject_scores.csv` — 2,250 participant/model/ROI values.
- `results/black_*.npz` — foldwise encoding correlations, ISC, fitted band weights,
  and explicit participant/parcel identifiers.

The five models are linguistic features, GloVe, BERT transformation magnitudes,
BERT embeddings, and BERT transformations. Ten left-hemisphere ROIs comprise
102 Schaefer parcels. Figure points are medians across participants.

## Archives

The **full** archive includes `save_data/`: all feature and response arrays
needed to refit the models. The **results-and-code** archive includes the same
code, fitted outputs, figure, and atlas, but omits the large input arrays. It can
regenerate the figure immediately; refitting requires the full archive's data.
Neither archive includes a Python environment. Extract the archive into a new
folder, then run commands from `black_figure2_reproduction/`.

## Environment

Use Python **3.12** (the tested version). Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU fitting, also install a CUDA-enabled PyTorch build suitable for the
machine. The tested build was PyTorch 2.5.1+cu118:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

The packaged results were computed with Himalaya's `torch_cuda` backend on an
RTX 2070 SUPER (8 GB). Exact library versions are in `results/environment.json`.
The CPU path uses NumPy and does not require PyTorch.

## Recreate the figure from the supplied results

```bash
python analysis/render_figure2.py
```

This uses the supplied parcellation and mapping, without downloading an atlas.

## Refit from the input arrays

```bash
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
python analysis/reproduce_figure2.py audit
python analysis/reproduce_figure2.py run --backend torch_cuda --output rerun_results
python analysis/render_figure2.py --results rerun_results
```

Use `--backend numpy` for CPU fitting. `--backend auto` selects CUDA if visible,
otherwise NumPy. An explicit GPU request fails if GPU access is unavailable.
Use a fresh output folder when changing the configuration. Each run records
input hashes, code hashes, and parameters and rejects incompatible cached fits.
The supplied `results/` folder preserves the verified run.

## Methods

- **Sample:** 45 analyzed participants. The input directory contains 46 response
  arrays; `sub-296` is excluded from model summaries according to the source
  analysis's comprehension exclusion. The ISC reference average includes all
  46 available response arrays, following the original ISC procedure.
- **Inputs:** responses have 514 TRs × 1,000 parcels. Features are already
  temporally aligned and include four consecutive lag blocks; no additional
  trimming or lagging is applied.
- **Encoding:** separate feature bands for the main model, silence, word/phoneme
  counts, and phoneme indicators. Three contiguous outer folds and three inner
  folds; training-partition standardization; 100 random-search candidates,
  seed 0, global alpha 1.0; correlation-based inner scoring. The implementation
  uses linear multiple-kernel ridge, equivalent to group ridge. Each
  participant/parcel retains its own fitted coefficients and band weights.
- **Evaluation:** Pearson correlation using only the main band's predictions.
  Nuisance-band predictions are excluded.
- **ISC:** correlate each participant with the self-inclusive group mean on the
  same outer test folds used for encoding.
- **ROI normalization:** average correlations across folds, Fisher-z transform,
  then average parcels within each ROI separately for encoding and ISC.
  Report `100 × mean_ROI(z_encoding) / mean_ROI(z_ISC)`. Normalize within 
  participants, then take the median across them.
- **Intervals:** 95% percentile bootstrap confidence intervals from 10,000
  participant resamples, seed 0.
- **Embeddings:** omit the first 768 columns of each of the four saved delay
  blocks, interpreting them as BERT's static layer 0. This retains layers 1–12.

## Validation

```bash
python -m unittest discover -s analysis -p test_figure2.py -v
```

Nine tests cover input integrity, feature scaling, layer selection, mapping
structure, ISC/folds, ROI aggregation, kernel/group-ridge equivalence,
independent-target batching, nuisance exclusion, and plotting. A standalone
GPU run was compared with the previously computed Black results. Its
verification is recorded in `results/package_validation.json`.

