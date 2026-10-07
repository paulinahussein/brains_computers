#!/usr/bin/env python3
"""Compute the parcelwise ISC noise ceiling for the 14 Pieman subjects.

The default reproduces the paper/repository convention: the group-average
time series includes the target subject. ISC is computed separately in three
contiguous temporal folds and then averaged across folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold


SUBJECTS = [
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-005",
    "sub-006", "sub-008", "sub-010", "sub-011", "sub-012",
    "sub-013", "sub-014", "sub-015", "sub-016",
]

N_PARCELS = 1000
N_FEATURE_TRS = 277
MAX_HRF_DELAY = 5
BOUNDARY_TRIM = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/nethome/phussein/brains/bert-brains/pieman_data"),
        help="Directory containing sub-XXX.nii.gz and Schaefer1000_3mm.nii.gz.",
    )
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help=(
            "Exclude the target subject from the group mean. This is a useful "
            "control, but differs from the paper's reported convention."
        ),
    )
    parser.add_argument(
        "--overwrite-parcel-data",
        action="store_true",
        help="Recreate cached sub-XXX_parcelwise_data.npy files.",
    )
    return parser.parse_args()


def zscore_rows(values: np.ndarray) -> np.ndarray:
    """Z-score each voxel across time, matching scipy zscore(ddof=1)."""
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, ddof=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = (values - means) / stds
    standardized[~np.isfinite(standardized)] = 0.0
    return standardized.astype(np.float32, copy=False)


def make_parcel_data(
    fmri_path: Path,
    atlas_labels: np.ndarray,
) -> np.ndarray:
    nii = nib.load(fmri_path)
    if nii.shape[:3] != atlas_labels.shape:
        raise ValueError(
            f"Spatial shape mismatch for {fmri_path}: "
            f"fMRI={nii.shape[:3]}, atlas={atlas_labels.shape}"
        )

    # Mirrors the Pieman branch in banded_ridge_regression.py. The delayed
    # feature matrix has 277 + 5 = 282 rows before the boundary trim.
    n_tr = min(nii.shape[3], N_FEATURE_TRS + MAX_HRF_DELAY)
    if n_tr <= 2 * BOUNDARY_TRIM:
        raise ValueError(f"Too few TRs in {fmri_path}: {n_tr}")

    mask = atlas_labels > 0
    labels = atlas_labels[mask]
    voxel_data = np.asanyarray(nii.dataobj)[mask, :n_tr].astype(np.float32)
    voxel_data = zscore_rows(voxel_data)
    voxel_data = voxel_data[:, BOUNDARY_TRIM:-BOUNDARY_TRIM]

    parcel_data = np.zeros((N_PARCELS, voxel_data.shape[1]), dtype=np.float32)
    for parcel_id in range(1, N_PARCELS + 1):
        selected = labels == parcel_id
        if not np.any(selected):
            raise ValueError(f"Atlas contains no voxels for parcel {parcel_id}")
        parcel_data[parcel_id - 1] = voxel_data[selected].mean(axis=0)

    if not np.isfinite(parcel_data).all():
        raise ValueError(f"Non-finite parcel values produced for {fmri_path}")
    return parcel_data


def rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation for corresponding rows of two 2-D arrays."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = numerator / denominator
    correlations[~np.isfinite(correlations)] = 0.0
    return correlations


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    atlas_path = data_dir / "Schaefer1000_3mm.nii.gz"
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Missing atlas: {atlas_path}")

    atlas_labels = np.rint(nib.load(atlas_path).get_fdata()).astype(np.int16)
    unique_labels = np.unique(atlas_labels)
    missing_labels = sorted(set(range(1, N_PARCELS + 1)) - set(unique_labels))
    if missing_labels:
        raise ValueError(
            f"Atlas is missing {len(missing_labels)} parcel labels; "
            f"first missing labels: {missing_labels[:10]}"
        )

    parcel_arrays: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for subject in SUBJECTS:
        fmri_path = data_dir / f"{subject}.nii.gz"
        cache_path = data_dir / f"{subject}_parcelwise_data.npy"
        if not fmri_path.is_file():
            raise FileNotFoundError(f"Missing fMRI file: {fmri_path}")

        if cache_path.is_file() and not args.overwrite_parcel_data:
            parcel_data = np.load(cache_path)
            print(f"Loaded cached parcel data: {subject} {parcel_data.shape}")
        else:
            print(f"Parcellating: {subject}", flush=True)
            parcel_data = make_parcel_data(fmri_path, atlas_labels)
            np.save(cache_path, parcel_data)
            print(f"Saved: {cache_path} {parcel_data.shape}", flush=True)

        if parcel_data.ndim != 2 or parcel_data.shape[0] != N_PARCELS:
            raise ValueError(f"Unexpected parcel data shape for {subject}: {parcel_data.shape}")
        if expected_shape is None:
            expected_shape = parcel_data.shape
        elif parcel_data.shape != expected_shape:
            raise ValueError(
                f"Subjects have different parcel data shapes: "
                f"{subject}={parcel_data.shape}, expected={expected_shape}"
            )
        if not np.isfinite(parcel_data).all():
            raise ValueError(f"Cached parcel data contains non-finite values: {subject}")
        parcel_arrays.append(parcel_data.astype(np.float32, copy=False))

    all_data = np.stack(parcel_arrays, axis=0)  # subjects x parcels x time
    isc_dir = data_dir / ("isc_leave_one_out" if args.leave_one_out else "isc")
    isc_dir.mkdir(parents=True, exist_ok=True)
    folds = KFold(n_splits=3, shuffle=False)

    for subject_index, subject in enumerate(SUBJECTS):
        subject_data = all_data[subject_index]
        if args.leave_one_out:
            group_data = np.mean(
                np.delete(all_data, subject_index, axis=0), axis=0
            )
        else:
            # This intentional self-inclusion matches compute_isc.py and the
            # Methods statement that including the subject biases the ceiling up.
            group_data = np.mean(all_data, axis=0)

        fold_scores = []
        for _, test_indices in folds.split(np.arange(subject_data.shape[1])):
            fold_scores.append(
                rowwise_correlation(
                    subject_data[:, test_indices], group_data[:, test_indices]
                )
            )
        isc = np.mean(np.stack(fold_scores, axis=1), axis=1).astype(np.float32)
        output_path = isc_dir / f"{subject}.npy"
        np.save(output_path, isc)
        print(
            f"Saved ISC: {subject} {isc.shape} "
            f"min={isc.min():.6f} mean={isc.mean():.6f} max={isc.max():.6f}",
            flush=True,
        )

    print(f"Finished: {len(SUBJECTS)} subjects")
    print(f"ISC directory: {isc_dir}")
    print("Mode:", "leave-one-subject-out" if args.leave_one_out else "paper-compatible inclusive mean")


if __name__ == "__main__":
    main()
