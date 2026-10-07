#!/usr/bin/env python3
"""
Extract Llama 3.2 3B attention transformations and transformation magnitudes
for the Black stimulus, then build Figure-2-ready four-delay feature matrices.

Methodological analogue to Kumar et al. (2024):
- "transformations": attention output after heads are merged, immediately BEFORE o_proj
- "transformation magnitudes": L2 norm of each head slice of the TR-mean transformation
- current-TR tokens are averaged, matching the original transformer-representations pipeline
- empty TRs are forward-filled, matching the repository's representation extraction behavior
- four final delay blocks use the already reconstructed Black alignment:
    raw[8:522], raw[7:521], raw[6:520], raw[5:519]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-3B"
MODEL_SHORT = "Llama-3.2-3B"

BRAINS = Path("/nethome/phussein/brains")
BERT_BRAINS = BRAINS / "bert-brains"
TR_FILE = BERT_BRAINS / "data" / "stimuli" / "black" / "tr_tokens.csv"

RAW_OUT = BERT_BRAINS / "text_processing_and_transformers" / "llama_black"
FIG2_X = BRAINS / "black_figure2_reproduction" / "save_data" / "X"

N_CONTEXT_TRS = 20
EXPECTED_RAW_TRS = 534
EXPECTED_FINAL_TRS = 514

DELAY_SLICES = [
    (8, 522),
    (7, 521),
    (6, 520),
    (5, 519),
]


def get_current_token_positions(tokenizer, prefix_text, current_text):
    if prefix_text:
        full_text = prefix_text + " " + current_text
        current_char_start = len(prefix_text) + 1
    else:
        full_text = current_text
        current_char_start = 0

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    current_positions = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > current_char_start and end > start
    ]

    if not current_positions:
        raise RuntimeError(
            f"Could not identify current-TR tokens for text: {current_text!r}"
        )

    return encoded, current_positions


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this extraction job.")

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    FIG2_X.mkdir(parents=True, exist_ok=True)

    print("Model:", MODEL_NAME)
    print("GPU:", torch.cuda.get_device_name(0))
    print("TR file:", TR_FILE)

    df = pd.read_csv(TR_FILE)

    if "tokens" not in df.columns:
        raise KeyError(f"'tokens' column missing. Columns: {list(df.columns)}")

    if len(df) != EXPECTED_RAW_TRS:
        raise ValueError(
            f"Expected {EXPECTED_RAW_TRS} Black TRs, found {len(df)}"
        )

    tr_texts = [
        "" if pd.isna(x) else str(x).strip()
        for x in df["tokens"].tolist()
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required because this script uses offset mappings."
        )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    config = model.config
    n_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // n_heads))

    if n_heads * head_dim != hidden_size:
        raise ValueError(
            f"Unexpected attention geometry: "
            f"{n_heads} heads x {head_dim} != hidden_size {hidden_size}"
        )

    print("Layers:", n_layers)
    print("Hidden size:", hidden_size)
    print("Attention heads:", n_heads)
    print("Head dim:", head_dim)
    print("KV heads:", getattr(config, "num_key_value_heads", "unknown"))

    layer_rows = [[] for _ in range(n_layers)]
    magnitude_rows = []

    captured = [None] * n_layers
    handles = []

    def make_hook(layer_idx):
        def hook(module, args):
            if len(args) != 1:
                raise RuntimeError(
                    f"Unexpected o_proj input tuple length at layer {layer_idx}: "
                    f"{len(args)}"
                )
            x = args[0]
            if x.ndim != 3 or x.shape[-1] != hidden_size:
                raise RuntimeError(
                    f"Unexpected o_proj input shape at layer {layer_idx}: "
                    f"{tuple(x.shape)}"
                )
            captured[layer_idx] = x.detach()
        return hook

    for layer_idx, layer in enumerate(model.layers):
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer_idx)
            )
        )

    previous_layer_means = None

    try:
        for tr_idx, current_text in enumerate(tr_texts):
            if current_text == "":
                if previous_layer_means is None:
                    current_layer_means = [
                        np.zeros(hidden_size, dtype=np.float32)
                        for _ in range(n_layers)
                    ]
                else:
                    current_layer_means = [
                        x.copy() for x in previous_layer_means
                    ]

                mags = np.stack(current_layer_means, axis=0).reshape(
                    n_layers, n_heads, head_dim
                )
                mags = np.linalg.norm(
                    mags, axis=-1
                ).reshape(-1).astype(np.float32)

            else:
                start = max(0, tr_idx - N_CONTEXT_TRS)
                previous_nonempty_texts = [
                    t for t in tr_texts[start:tr_idx] if t
                ]
                prefix_text = " ".join(previous_nonempty_texts)

                encoded, current_positions = get_current_token_positions(
                    tokenizer,
                    prefix_text,
                    current_text,
                )

                encoded = {
                    k: v.to("cuda")
                    for k, v in encoded.items()
                }

                captured[:] = [None] * n_layers

                with torch.inference_mode():
                    _ = model(**encoded, use_cache=False)

                if any(x is None for x in captured):
                    missing = [
                        i for i, x in enumerate(captured) if x is None
                    ]
                    raise RuntimeError(
                        f"Hooks did not fire for layers: {missing}"
                    )

                current_layer_means = []

                for layer_idx, layer_tensor in enumerate(captured):
                    token_tensor = layer_tensor[
                        0, current_positions, :
                    ]
                    tr_mean = (
                        token_tensor
                        .float()
                        .mean(dim=0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    if tr_mean.shape != (hidden_size,):
                        raise RuntimeError(
                            f"Layer {layer_idx}, TR {tr_idx}: "
                            f"bad mean shape {tr_mean.shape}"
                        )

                    current_layer_means.append(tr_mean)

                mags = np.stack(current_layer_means, axis=0).reshape(
                    n_layers, n_heads, head_dim
                )
                mags = np.linalg.norm(
                    mags, axis=-1
                ).reshape(-1).astype(np.float32)

                previous_layer_means = [
                    x.copy() for x in current_layer_means
                ]

            for layer_idx, row in enumerate(current_layer_means):
                layer_rows[layer_idx].append(row)

            magnitude_rows.append(mags)

            if (
                tr_idx < 5
                or (tr_idx + 1) % 25 == 0
                or tr_idx + 1 == EXPECTED_RAW_TRS
            ):
                print(
                    f"TR {tr_idx + 1:3d}/{EXPECTED_RAW_TRS} | "
                    f"empty={current_text == ''}"
                )

    finally:
        for handle in handles:
            handle.remove()

    layer_arrays = []

    for layer_idx in range(n_layers):
        arr = np.stack(layer_rows[layer_idx], axis=0).astype(np.float32)

        expected = (EXPECTED_RAW_TRS, hidden_size)
        if arr.shape != expected:
            raise ValueError(
                f"Layer {layer_idx}: {arr.shape}, expected {expected}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(
                f"Layer {layer_idx} contains NaN/Inf"
            )

        out = RAW_OUT / (
            f"black_{MODEL_SHORT}_layer_{layer_idx}"
            f"_z_representations.npy"
        )
        np.save(out, arr)
        layer_arrays.append(arr)
        print("Saved:", out, arr.shape)

    raw_transformations = np.concatenate(layer_arrays, axis=1)

    expected_raw_width = n_layers * hidden_size
    if raw_transformations.shape != (
        EXPECTED_RAW_TRS,
        expected_raw_width,
    ):
        raise ValueError(
            f"Raw transformations have shape "
            f"{raw_transformations.shape}"
        )

    raw_magnitudes = np.stack(
        magnitude_rows, axis=0
    ).astype(np.float32)

    expected_mag_width = n_layers * n_heads
    if raw_magnitudes.shape != (
        EXPECTED_RAW_TRS,
        expected_mag_width,
    ):
        raise ValueError(
            f"Raw magnitudes have shape {raw_magnitudes.shape}"
        )

    mag_raw_out = (
        RAW_OUT
        / f"black_{MODEL_SHORT}_zrep_magnitudes.npy"
    )
    np.save(mag_raw_out, raw_magnitudes)

    transformation_delays = []
    magnitude_delays = []

    for delay_idx, (start, end) in enumerate(DELAY_SLICES):
        z = raw_transformations[start:end]
        m = raw_magnitudes[start:end]

        if len(z) != EXPECTED_FINAL_TRS:
            raise ValueError(
                f"Delay {delay_idx}: transformations have {len(z)} rows"
            )
        if len(m) != EXPECTED_FINAL_TRS:
            raise ValueError(
                f"Delay {delay_idx}: magnitudes have {len(m)} rows"
            )

        transformation_delays.append(z)
        magnitude_delays.append(m)

        print(
            f"Delay {delay_idx}: raw[{start}:{end}] "
            f"-> transformations {z.shape}, magnitudes {m.shape}"
        )

    final_transformations = np.concatenate(
        transformation_delays, axis=1
    ).astype(np.float32)

    final_magnitudes = np.concatenate(
        magnitude_delays, axis=1
    ).astype(np.float32)

    transform_out = FIG2_X / "black_llama3_full_z.npy"
    magnitude_out = FIG2_X / "black_llama3_zrep_magnitudes.npy"

    np.save(transform_out, final_transformations)
    np.save(magnitude_out, final_magnitudes)

    expected_transform_shape = (
        EXPECTED_FINAL_TRS,
        4 * n_layers * hidden_size,
    )
    expected_magnitude_shape = (
        EXPECTED_FINAL_TRS,
        4 * n_layers * n_heads,
    )

    assert final_transformations.shape == expected_transform_shape
    assert final_magnitudes.shape == expected_magnitude_shape
    assert np.isfinite(final_transformations).all()
    assert np.isfinite(final_magnitudes).all()

    print()
    print("========================================")
    print("Finished")
    print("========================================")
    print("Raw transformations:", raw_transformations.shape)
    print("Raw magnitudes:", raw_magnitudes.shape)
    print("Figure-2 transformations:", final_transformations.shape)
    print("Figure-2 magnitudes:", final_magnitudes.shape)
    print("Transformations output:", transform_out)
    print("Magnitudes output:", magnitude_out)
    print("NaNs transformations:", int(np.isnan(final_transformations).sum()))
    print("NaNs magnitudes:", int(np.isnan(final_magnitudes).sum()))


if __name__ == "__main__":
    main()
