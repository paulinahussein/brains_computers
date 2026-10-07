import os
import numpy as np


BASE = "/nethome/phussein/brains/bert-brains"

OUT = os.path.join(
    "/nethome/phussein/brains/black_figure2_reproduction",
    "save_data",
    "X",
    "layerwise",
)

os.makedirs(OUT, exist_ok=True)


MODELS = {
    "bert": {
        "dir": os.path.join(BASE, "text_processing_and_transformers"),
        "prefix": "black_bert-base-uncased",
        "layers": 12,
    },
    "llama3": {
        "dir": os.path.join(
            BASE,
            "text_processing_and_transformers",
            "llama_black",
        ),
        "prefix": "black_Llama-3.2-3B",
        "layers": 28,
    },
    "llama4": {
        "dir": os.path.join(
            BASE,
            "text_processing_and_transformers",
            "llama4_black",
        ),
        "prefix": "black_Llama-4-Scout-17B-16E",
        "layers": 48,
    },
}


def make_delayed_features(raw):
    """
    Reproduce the same four temporal blocks used for the Black
    Figure 2 features.

    Input:
        raw: (534, features)

    Output:
        (514, 4 * features)
    """

    if raw.shape[0] != 534:
        raise ValueError(
            f"Expected 534 raw TRs, got {raw.shape}"
        )

    blocks = [
        raw[8:522],
        raw[7:521],
        raw[6:520],
        raw[5:519],
    ]

    for block in blocks:
        assert block.shape[0] == 514

    return np.concatenate(blocks, axis=1)


for model_name, cfg in MODELS.items():

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    for layer in range(cfg["layers"]):

        files = {
            "embeddings": os.path.join(
                cfg["dir"],
                f"{cfg['prefix']}_layer_{layer}_activations.npy",
            ),
            "transformations": os.path.join(
                cfg["dir"],
                f"{cfg['prefix']}_layer_{layer}_z_representations.npy",
            ),
        }

        for feature_type, infile in files.items():

            if not os.path.exists(infile):
                raise FileNotFoundError(infile)

            raw = np.load(infile)

            print(
                f"{model_name:7s} "
                f"layer {layer:02d} "
                f"{feature_type:15s} "
                f"raw={raw.shape}"
            )

            delayed = make_delayed_features(raw)

            outfile = os.path.join(
                OUT,
                f"black_{model_name}_layer_{layer:02d}_{feature_type}.npy",
            )

            np.save(outfile, delayed.astype(np.float32))

            print(
                f"  -> {delayed.shape} "
                f"NaNs={np.isnan(delayed).sum()} "
                f"Infs={np.isinf(delayed).sum()}"
            )


print()
print("Finished.")
print("Output directory:")
print(OUT)