#!/usr/bin/env python3
"""Create the Pieman BERT transformation-magnitude features for Figure 2.

This implements the relevant computation from
transformer-transformation-magnitudes.ipynb: each 768-dimensional layer-wise
Z-representation is split into twelve 64-dimensional attention-head vectors,
and the L2 norm of every head vector is computed. Across twelve layers, this
produces 144 features per TR.
"""

from pathlib import Path

import numpy as np


STIMULUS = "pieman"
MODEL = "bert-base-uncased"
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
EXPECTED_TRS = 277


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    # Works both when the script is placed in the repository root and when it
    # is placed directly in text_processing_and_transformers/.
    feature_dir = script_dir / "text_processing_and_transformers"
    if not feature_dir.is_dir():
        feature_dir = script_dir

    layer_arrays = []
    expected_width = N_HEADS * HEAD_DIM

    for layer in range(N_LAYERS):
        input_path = feature_dir / (
            f"{STIMULUS}_{MODEL}_layer_{layer}_z_representations.npy"
        )
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input file: {input_path}")

        z_representations = np.load(input_path)
        expected_shape = (EXPECTED_TRS, expected_width)
        if z_representations.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {input_path.name}: "
                f"{z_representations.shape}; expected {expected_shape}"
            )

        layer_arrays.append(z_representations)
        print(f"Layer {layer:2d}: {z_representations.shape}")

    # Shape before norm: TR x layer x head x head dimension.
    z_headwise = np.stack(layer_arrays, axis=1).reshape(
        EXPECTED_TRS, N_LAYERS, N_HEADS, HEAD_DIM
    )
    magnitudes = np.linalg.norm(z_headwise, axis=-1).reshape(
        EXPECTED_TRS, N_LAYERS * N_HEADS
    )

    if not np.isfinite(magnitudes).all():
        raise ValueError("Output contains NaN or infinite values")

    output_path = feature_dir / f"{STIMULUS}_{MODEL}_zrep_magnitudes.npy"
    np.save(output_path, magnitudes)

    print("\nSaved:", output_path)
    print("Shape:", magnitudes.shape)
    print("Dtype:", magnitudes.dtype)
    print("NaNs:", int(np.isnan(magnitudes).sum()))
    print("Infs:", int(np.isinf(magnitudes).sum()))
    print("Min:", float(magnitudes.min()))
    print("Max:", float(magnitudes.max()))


if __name__ == "__main__":
    main()
