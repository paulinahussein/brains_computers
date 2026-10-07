#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------
# configuration
# --------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-3B"

BASE = Path("/nethome/phussein/brains/bert-brains")

TR_FILE = (
    BASE
    / "data"
    / "stimuli"
    / "black"
    / "tr_tokens.csv"
)

RAW_OUTPUT_DIR = (
    BASE
    / "text_processing_and_transformers"
    / "llama_black"
)

FINAL_OUTPUT = Path(
    "/nethome/phussein/brains/"
    "black_figure2_reproduction/"
    "save_data/X/"
    "black_llama3_full.npy"
)

NUM_CONTEXT_TRS = 20

EXPECTED_RAW_TRS = 534
EXPECTED_FINAL_TRS = 514

# Reconstructed from the existing BERT raw/final arrays.
# Each block has exactly 514 rows.
DELAY_SLICES = [
    (8, 522),
    (7, 521),
    (6, 520),
    (5, 519),
]


# --------------------------------------------------
# setup
# --------------------------------------------------

RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Model:", MODEL_NAME)
print("TR file:", TR_FILE)
print("Raw output:", RAW_OUTPUT_DIR)
print("Final output:", FINAL_OUTPUT)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# load stimulus
# --------------------------------------------------

df = pd.read_csv(TR_FILE)

if "tokens" not in df.columns:
    raise ValueError("Expected a 'tokens' column in tr_tokens.csv")

df["tokens"] = df["tokens"].fillna("")

trs = df["tokens"].tolist()

print("Number of TRs:", len(trs))

if len(trs) != EXPECTED_RAW_TRS:
    raise ValueError(
        f"Expected {EXPECTED_RAW_TRS} Black TRs, got {len(trs)}"
    )


# --------------------------------------------------
# load model
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    output_hidden_states=True,
)

model.to(device)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

print("Layers:", n_layers)
print("Hidden size:", hidden_size)

if n_layers != 28:
    raise ValueError(f"Expected 28 layers, got {n_layers}")

if hidden_size != 3072:
    raise ValueError(
        f"Expected hidden size 3072, got {hidden_size}"
    )


# --------------------------------------------------
# storage
# --------------------------------------------------

# One list per transformer layer.
#
# IMPORTANT:
# outputs.hidden_states has 29 entries:
#
# hidden_states[0]  = input embedding state
# hidden_states[1]  = output after layer 0
# ...
# hidden_states[28] = output after layer 27
#
# We deliberately exclude hidden_states[0], analogous
# to omitting BERT's static layer 0.

layer_features = [
    [] for _ in range(n_layers)
]

previous_features = None


# --------------------------------------------------
# extract representations
# --------------------------------------------------

for i, current_tr in enumerate(trs):

    if i % 25 == 0:
        print(f"Processing TR {i}/{len(trs)}")

    # Match the original repository:
    # current TR + up to 20 preceding TRs.
    start = max(0, i - NUM_CONTEXT_TRS)

    window_text = " ".join(
        trs[start:i + 1]
    )

    # Empty TR:
    # the repository extraction code initially sets None
    # and later forward-fills representations.
    if current_tr.strip() == "":

        if previous_features is None:
            # Only relevant if the very first TR were empty.
            current_features = [
                np.zeros(hidden_size, dtype=np.float32)
                for _ in range(n_layers)
            ]
        else:
            current_features = [
                x.copy() for x in previous_features
            ]

        for layer_idx in range(n_layers):
            layer_features[layer_idx].append(
                current_features[layer_idx]
            )

        previous_features = current_features
        continue

    # Tokenize the complete context window.
    window_ids = tokenizer(
        window_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)

    # Tokenize current TR separately so that we know
    # how many final tokens belong to the current TR.
    current_ids = tokenizer(
        current_tr,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    n_current_tokens = current_ids.shape[1]

    if n_current_tokens == 0:
        raise RuntimeError(
            f"Non-empty TR {i} produced zero tokens."
        )

    with torch.inference_mode():

        outputs = model(
            input_ids=window_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states

    if len(hidden_states) != n_layers + 1:
        raise RuntimeError(
            f"Expected {n_layers + 1} hidden states, "
            f"got {len(hidden_states)}"
        )

    current_features = []

    # Skip hidden_states[0].
    for layer_idx in range(n_layers):

        state = hidden_states[layer_idx + 1]

        # Shape:
        # batch x tokens x hidden_size
        #
        # Keep only tokens assigned to current TR.
        current_state = state[
            0,
            -n_current_tokens:,
            :
        ]

        # Match original analysis:
        # mean over tokens occurring in current TR.
        mean_state = (
            current_state
            .mean(dim=0)
            .float()
            .cpu()
            .numpy()
        )

        current_features.append(mean_state)

        layer_features[layer_idx].append(
            mean_state
        )

    previous_features = current_features


# --------------------------------------------------
# save raw per-layer arrays
# --------------------------------------------------

raw_layers = []

for layer_idx in range(n_layers):

    x = np.stack(
        layer_features[layer_idx],
        axis=0,
    ).astype(np.float32)

    expected = (
        EXPECTED_RAW_TRS,
        hidden_size,
    )

    if x.shape != expected:
        raise ValueError(
            f"Layer {layer_idx}: "
            f"got {x.shape}, expected {expected}"
        )

    out = (
        RAW_OUTPUT_DIR
        / f"black_Llama-3.2-3B_layer_{layer_idx}_activations.npy"
    )

    np.save(out, x)

    raw_layers.append(x)

    print(
        f"Layer {layer_idx:02d}: "
        f"saved {out.name} {x.shape}"
    )


# --------------------------------------------------
# concatenate all transformer layers
# --------------------------------------------------

raw = np.concatenate(
    raw_layers,
    axis=1,
)

expected_raw_shape = (
    EXPECTED_RAW_TRS,
    n_layers * hidden_size,
)

if raw.shape != expected_raw_shape:
    raise ValueError(
        f"Raw concatenated shape {raw.shape}, "
        f"expected {expected_raw_shape}"
    )

print()
print("Raw concatenated:", raw.shape)


# --------------------------------------------------
# reproduce Black alignment + four delay blocks
# --------------------------------------------------

delay_blocks = []

for block_idx, (start, stop) in enumerate(DELAY_SLICES):

    block = raw[start:stop]

    if block.shape[0] != EXPECTED_FINAL_TRS:
        raise ValueError(
            f"Delay block {block_idx}: "
            f"{block.shape}"
        )

    delay_blocks.append(block)

    print(
        f"Delay {block_idx}: "
        f"raw[{start}:{stop}] -> {block.shape}"
    )


llama_full = np.concatenate(
    delay_blocks,
    axis=1,
).astype(np.float32)


expected_final_shape = (
    EXPECTED_FINAL_TRS,
    4 * n_layers * hidden_size,
)

if llama_full.shape != expected_final_shape:
    raise ValueError(
        f"Final shape {llama_full.shape}, "
        f"expected {expected_final_shape}"
    )


# --------------------------------------------------
# validation
# --------------------------------------------------

if not np.isfinite(llama_full).all():
    raise ValueError(
        "Final Llama matrix contains NaN or Inf."
    )


# --------------------------------------------------
# save final encoding input
# --------------------------------------------------

np.save(
    FINAL_OUTPUT,
    llama_full,
)

print()
print("========================================")
print("Finished")
print("========================================")
print("Raw:", raw.shape)
print("Final:", llama_full.shape)
print("Saved:", FINAL_OUTPUT)
print("NaNs:", np.isnan(llama_full).sum())
print("Infs:", np.isinf(llama_full).sum())