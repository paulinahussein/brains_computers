"""Shared backend selection and training-only feature scaling."""
import os
import numpy as np


def configure_backend(name=None):
    from himalaya.backend import set_backend
    name = name or os.environ.get("HIMALAYA_BACKEND", "auto")
    if name == "auto":
        try:
            import torch
            name = "torch_cuda" if torch.cuda.is_available() else "numpy"
        except ImportError:
            name = "numpy"
    # An explicit GPU request must fail rather than silently running on CPU.
    backend = set_backend(name, on_error="raise")
    print(f"Himalaya backend: {backend.name}", flush=True)
    return backend


def standardize_train_test(train, test):
    """Apply the original training statistics to both partitions."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return ((train - mean) / scale).astype("float32"), ((test - mean) / scale).astype("float32")
