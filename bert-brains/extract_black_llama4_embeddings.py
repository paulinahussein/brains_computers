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

    print("Layers:", n_layers)
    print("Hidden size:", hidden_size)
    print("HF device map:", getattr(model, "hf_device_map", None))

    input_device = text_model.embed_tokens.weight.device
    print("Input embedding device:", input_device)

    layer_rows = [[] for _ in range(n_layers)]
    previous_layer_means = None

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

            with torch.inference_mode():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")
            if len(hidden_states) != n_layers + 1:
                raise RuntimeError(
                    f"Expected {n_layers + 1} hidden-state tensors, "
                    f"got {len(hidden_states)}"
                )

            current_layer_means = []
            for layer_idx, hs in enumerate(hidden_states[1:]):
                tr_mean = (
                    hs[0, current_positions, :]
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
            del outputs, hidden_states

        for layer_idx, row in enumerate(current_layer_means):
            layer_rows[layer_idx].append(row)

        if tr_idx < 5 or (tr_idx + 1) % 25 == 0 or tr_idx + 1 == EXPECTED_RAW_TRS:
            print(
                f"TR {tr_idx + 1:3d}/{EXPECTED_RAW_TRS} | "
                f"empty={current_text == ''}",
                flush=True,
            )

    layer_arrays = []
    for layer_idx in range(n_layers):
        arr = np.stack(layer_rows[layer_idx], axis=0).astype(np.float32)
        expected = (EXPECTED_RAW_TRS, hidden_size)
        if arr.shape != expected:
            raise ValueError(f"Layer {layer_idx}: {arr.shape}, expected {expected}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Layer {layer_idx} contains NaN/Inf")

        out = RAW_OUT / f"black_{MODEL_SHORT}_layer_{layer_idx}_activations.npy"
        np.save(out, arr)
        layer_arrays.append(arr)
        print("Saved:", out, arr.shape)

    raw = np.concatenate(layer_arrays, axis=1)
    delayed = []

    for delay_idx, (start, end) in enumerate(DELAY_SLICES):
        part = raw[start:end]
        if len(part) != EXPECTED_FINAL_TRS:
            raise ValueError(f"Delay {delay_idx}: got {len(part)} rows")
        delayed.append(part)
        print(f"Delay {delay_idx}: raw[{start}:{end}] -> {part.shape}")

    final = np.concatenate(delayed, axis=1).astype(np.float32)
    expected_final = (
        EXPECTED_FINAL_TRS,
        4 * n_layers * hidden_size,
    )
    if final.shape != expected_final:
        raise ValueError(f"Final embeddings: {final.shape}, expected {expected_final}")
    if not np.isfinite(final).all():
        raise ValueError("Final embeddings contain NaN/Inf")

    output = FIG2_X / "black_llama4_full.npy"
    np.save(output, final)

    print("\n========================================")
    print("Finished Llama 4 embeddings")
    print("========================================")
    print("Raw:", raw.shape)
    print("Figure-2:", final.shape)
    print("Output:", output)
    print("NaNs:", int(np.isnan(final).sum()))
    print("Infs:", int(np.isinf(final).sum()))


if __name__ == "__main__":
    main()
