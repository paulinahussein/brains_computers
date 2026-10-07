#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E"
MODEL_SHORT = "Llama-4-Scout-17B-16E"

BRAINS = Path("/nethome/phussein/brains")
BERT_BRAINS = BRAINS / "bert-brains"
TR_FILE = BERT_BRAINS / "data" / "stimuli" / "black" / "tr_tokens.csv"
RAW_OUT = BERT_BRAINS / "text_processing_and_transformers" / "llama4_black"
FIG2_X = BRAINS / "black_figure2_reproduction" / "save_data" / "X"

N_CONTEXT_TRS = 20
EXPECTED_RAW_TRS = 534
EXPECTED_FINAL_TRS = 514
DELAY_SLICES = [(8, 522), (7, 521), (6, 520), (5, 519)]


def auth_kwargs():
    token = os.environ.get("HF_TOKEN")
    return {"token": token} if token else {}


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
        i for i, (start, end) in enumerate(offsets)
        if end > current_char_start and end > start
    ]
    if not current_positions:
        raise RuntimeError(
            f"Could not identify current-TR tokens for text: {current_text!r}"
        )
    return encoded, current_positions


def main():
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    FIG2_X.mkdir(parents=True, exist_ok=True)

    print("Model:", MODEL_NAME)
    print("Visible CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA GPU is visible.")

    df = pd.read_csv(TR_FILE)
    if "tokens" not in df.columns:
        raise KeyError(f"'tokens' column missing. Columns: {list(df.columns)}")
    if len(df) != EXPECTED_RAW_TRS:
        raise ValueError(f"Expected {EXPECTED_RAW_TRS} TRs, found {len(df)}")

    tr_texts = ["" if pd.isna(x) else str(x).strip() for x in df["tokens"]]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, **auth_kwargs()
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required for offset mappings.")

    print("Loading Llama 4 Scout in BF16 with device_map='auto' ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        **auth_kwargs(),
    )
    model.eval()

    text_model = model.model
    config = text_model.config

    n_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", hidden_size // n_heads))

    if n_heads * head_dim != hidden_size:
        raise ValueError(
            f"Unexpected attention geometry: {n_heads} x {head_dim} != {hidden_size}"
        )

    print("Layers:", n_layers)
    print("Hidden size:", hidden_size)
    print("Query heads:", n_heads)
    print("KV heads:", getattr(config, "num_key_value_heads", "unknown"))
    print("Head dim:", head_dim)
    print("Local experts:", getattr(config, "num_local_experts", "unknown"))
    print("HF device map:", getattr(model, "hf_device_map", None))

    input_device = text_model.embed_tokens.weight.device
    print("Input embedding device:", input_device)

    layer_rows = [[] for _ in range(n_layers)]
    magnitude_rows = []
    captured = [None] * n_layers
    handles = []

    def make_hook(layer_idx):
        def hook(module, args):
            if len(args) != 1:
                raise RuntimeError(
                    f"Layer {layer_idx}: unexpected o_proj input tuple length {len(args)}"
                )
            x = args[0]
            expected_last_dim = n_heads * head_dim
            if x.ndim != 3 or x.shape[-1] != expected_last_dim:
                raise RuntimeError(
                    f"Layer {layer_idx}: pre-o_proj shape {tuple(x.shape)}, "
                    f"expected last dim {expected_last_dim}"
                )
            captured[layer_idx] = x.detach()
        return hook

    for layer_idx, layer in enumerate(text_model.layers):
        handles.append(
            layer.self_attn.o_proj.register_forward_pre_hook(make_hook(layer_idx))
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
                    current_layer_means = [x.copy() for x in previous_layer_means]
            else:
                start = max(0, tr_idx - N_CONTEXT_TRS)
                prefix_text = " ".join(t for t in tr_texts[start:tr_idx] if t)

                encoded, current_positions = get_current_token_positions(
                    tokenizer, prefix_text, current_text
                )
                encoded = {k: v.to(input_device) for k, v in encoded.items()}

                captured[:] = [None] * n_layers

                with torch.inference_mode():
                    _ = model(
                        **encoded,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                missing = [i for i, x in enumerate(captured) if x is None]
                if missing:
                    raise RuntimeError(
                        f"o_proj hooks did not fire for layers: {missing}"
                    )

                current_layer_means = []
                for layer_idx, layer_tensor in enumerate(captured):
                    tr_mean = (
                        layer_tensor[0, current_positions, :]
                        .float()
                        .mean(dim=0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    if tr_mean.shape != (hidden_size,):
                        raise RuntimeError(
                            f"Layer {layer_idx}, TR {tr_idx}: {tr_mean.shape}"
                        )
                    current_layer_means.append(tr_mean)

                previous_layer_means = [x.copy() for x in current_layer_means]
                captured[:] = [None] * n_layers

            headwise = np.stack(
                current_layer_means, axis=0
            ).reshape(n_layers, n_heads, head_dim)

            mags = np.linalg.norm(
                headwise, axis=-1
            ).reshape(-1).astype(np.float32)

            for layer_idx, row in enumerate(current_layer_means):
                layer_rows[layer_idx].append(row)
            magnitude_rows.append(mags)

            if tr_idx < 5 or (tr_idx + 1) % 25 == 0 or tr_idx + 1 == EXPECTED_RAW_TRS:
                print(
                    f"TR {tr_idx + 1:3d}/{EXPECTED_RAW_TRS} | "
                    f"empty={current_text == ''}",
                    flush=True,
                )

    finally:
        for handle in handles:
            handle.remove()

    layer_arrays = []
    for layer_idx in range(n_layers):
        arr = np.stack(layer_rows[layer_idx], axis=0).astype(np.float32)
        expected = (EXPECTED_RAW_TRS, hidden_size)
        if arr.shape != expected:
            raise ValueError(f"Layer {layer_idx}: {arr.shape}, expected {expected}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Layer {layer_idx} contains NaN/Inf")

        out = RAW_OUT / (
            f"black_{MODEL_SHORT}_layer_{layer_idx}_z_representations.npy"
        )
        np.save(out, arr)
        layer_arrays.append(arr)
        print("Saved:", out, arr.shape)

    raw_transformations = np.concatenate(layer_arrays, axis=1)
    raw_magnitudes = np.stack(magnitude_rows, axis=0).astype(np.float32)

    raw_mag_out = RAW_OUT / f"black_{MODEL_SHORT}_zrep_magnitudes.npy"
    np.save(raw_mag_out, raw_magnitudes)

    z_delays = []
    mag_delays = []

    for delay_idx, (start, end) in enumerate(DELAY_SLICES):
        z = raw_transformations[start:end]
        m = raw_magnitudes[start:end]

        if len(z) != EXPECTED_FINAL_TRS or len(m) != EXPECTED_FINAL_TRS:
            raise ValueError(
                f"Delay {delay_idx}: bad row count z={len(z)}, magnitudes={len(m)}"
            )

        z_delays.append(z)
        mag_delays.append(m)

        print(
            f"Delay {delay_idx}: raw[{start}:{end}] -> "
            f"transformations {z.shape}, magnitudes {m.shape}"
        )

    final_z = np.concatenate(z_delays, axis=1).astype(np.float32)
    final_mag = np.concatenate(mag_delays, axis=1).astype(np.float32)

    expected_final_z = (
        EXPECTED_FINAL_TRS,
        4 * n_layers * hidden_size,
    )
    expected_final_mag = (
        EXPECTED_FINAL_TRS,
        4 * n_layers * n_heads,
    )

    if final_z.shape != expected_final_z:
        raise ValueError(
            f"Final transformations: {final_z.shape}, expected {expected_final_z}"
        )
    if final_mag.shape != expected_final_mag:
        raise ValueError(
            f"Final magnitudes: {final_mag.shape}, expected {expected_final_mag}"
        )

    if not np.isfinite(final_z).all():
        raise ValueError("Final transformations contain NaN/Inf")
    if not np.isfinite(final_mag).all():
        raise ValueError("Final magnitudes contain NaN/Inf")

    z_out = FIG2_X / "black_llama4_full_z.npy"
    mag_out = FIG2_X / "black_llama4_zrep_magnitudes.npy"

    np.save(z_out, final_z)
    np.save(mag_out, final_mag)

    print("\n========================================")
    print("Finished Llama 4 transformations")
    print("========================================")
    print("Raw transformations:", raw_transformations.shape)
    print("Raw magnitudes:", raw_magnitudes.shape)
    print("Figure-2 transformations:", final_z.shape)
    print("Figure-2 magnitudes:", final_mag.shape)
    print("Transformations output:", z_out)
    print("Magnitudes output:", mag_out)
    print("NaNs transformations:", int(np.isnan(final_z).sum()))
    print("NaNs magnitudes:", int(np.isnan(final_mag).sum()))


if __name__ == "__main__":
    main()
