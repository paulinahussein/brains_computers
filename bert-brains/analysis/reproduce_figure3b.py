import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# paths
# ---------------------------------------------------------

BLACK_ROOT = Path(
    "/nethome/phussein/brains/black_figure2_reproduction"
)

DATA_DIR = BLACK_ROOT / "save_data"

LAYER_DIR = (
    DATA_DIR
    / "X"
    / "layerwise"
)

MAPPING_FILE = (
    BLACK_ROOT
    / "Parcel2ROI_new.npy"
)

# reuse exactly the Figure-2 encoding backend
BLACK_ANALYSIS = BLACK_ROOT / "analysis"

if str(BLACK_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(BLACK_ANALYSIS))

from encoding_backend import (  # noqa: E402
    configure_backend,
    standardize_train_test,
)


# ---------------------------------------------------------
# constants
# ---------------------------------------------------------

MODELS = {
    "bert": 12,
    "llama3": 28,
    "llama4": 48,
}

MODEL_NAMES = {
    "bert": "BERT-base",
    "llama3": "Llama 3.2 3B",
    "llama4": "Llama 4 Scout",
}

NUISANCE = [
    "nuisance_silent_",
    "nuisance_number_",
    "nuisance_phonemes_",
]

# Parcel2ROI_new.npy:
# 1..10  = left-hemisphere language ROIs
# 11     = left V1
# 12..21 = right-hemisphere language ROIs
# 22     = right V1
#
# For Fig. 3B we use the cortical language network,
# i.e. language parcels from both hemispheres.
LANGUAGE_LABELS = list(range(1, 11)) + list(range(12, 22))


# ---------------------------------------------------------
# basic helpers
# ---------------------------------------------------------

def load_matrix(path):
    x = np.load(
        path,
        allow_pickle=False,
    )

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2-D array: {path}, got {x.shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"Non-finite values in {path}"
        )

    return x.astype(
        np.float32,
        copy=False,
    )


def column_correlation(x, y):
    """
    Pearson correlation independently for every target column.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = x - x.mean(
        axis=0,
        keepdims=True,
    )

    y = y - y.mean(
        axis=0,
        keepdims=True,
    )

    numerator = np.sum(
        x * y,
        axis=0,
    )

    denominator = np.sqrt(
        np.sum(x * x, axis=0)
        *
        np.sum(y * y, axis=0)
    )

    return np.divide(
        numerator,
        denominator,
        out=np.full(
            x.shape[1],
            np.nan,
            dtype=np.float64,
        ),
        where=denominator > 0,
    )


def partial_correlation(
    y,
    x,
    control,
):
    """
    Correlation between y and x while controlling for control.

    Arrays are:
        time x targets
    """

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    control = np.asarray(
        control,
        dtype=np.float64,
    )

    def residualize(a, c):

        a0 = a - a.mean(
            axis=0,
            keepdims=True,
        )

        c0 = c - c.mean(
            axis=0,
            keepdims=True,
        )

        denom = np.sum(
            c0 * c0,
            axis=0,
        )

        beta = np.divide(
            np.sum(
                c0 * a0,
                axis=0,
            ),
            denom,
            out=np.zeros(
                a.shape[1],
                dtype=np.float64,
            ),
            where=denom > 0,
        )

        return (
            a0
            -
            c0 * beta
        )

    y_res = residualize(
        y,
        control,
    )

    x_res = residualize(
        x,
        control,
    )

    return column_correlation(
        y_res,
        x_res,
    )


# ---------------------------------------------------------
# encoding
# ---------------------------------------------------------

def encode_with_predictions(
    bands,
    y,
    backend,
    n_iter=100,
    seed=0,
    alphas=(1.0,),
):
    """
    Same nested-CV / multiple-kernel encoding model used
    in the reconstructed Figure-2 pipeline.

    The difference is that we retain the out-of-sample
    predictions from all three outer folds.
    """

    from sklearn.model_selection import KFold
    from himalaya.scoring import correlation_score
    from himalaya.kernel_ridge import (
        MultipleKernelRidgeCV,
    )

    outer_cv = KFold(
        3,
        shuffle=False,
    )

    predictions = np.full(
        y.shape,
        np.nan,
        dtype=np.float32,
    )

    for fold, (train, test) in enumerate(
        outer_cv.split(y),
        start=1,
    ):

        start = time.monotonic()

        pairs = [
            standardize_train_test(
                x[train],
                x[test],
            )
            for x in bands
        ]

        train_bands, test_bands = map(
            list,
            zip(*pairs),
        )

        train_gpu = [
            backend.asarray(x)
            for x in train_bands
        ]

        test_gpu = [
            backend.asarray(x)
            for x in test_bands
        ]

        xtrain = backend.stack([
            x @ x.T
            for x in train_gpu
        ])

        xtest = backend.stack([
            x_test @ x_train.T
            for x_test, x_train
            in zip(
                test_gpu,
                train_gpu,
            )
        ])

        params = dict(
            n_iter=n_iter,
            score_func=correlation_score,
            progress_bar=False,
            alphas=np.asarray(
                alphas,
                dtype=np.float32,
            ),
            n_targets_batch=512,
        )

        model = MultipleKernelRidgeCV(
            kernels="precomputed",
            cv=KFold(
                3,
                shuffle=False,
            ),
            fit_intercept=False,
            random_state=seed,
            solver_params=params,
        )

        model.fit(
            xtrain,
            y[train],
        )

        # prediction from main feature band only;
        # nuisance-band predictions are excluded
        prediction = backend.to_numpy(
            model.predict(
                xtest,
                split=True,
            )[-1]
        )

        predictions[test] = prediction

        print(
            f"    fold {fold}/3: "
            f"{time.monotonic() - start:.1f}s",
            flush=True,
        )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            "Missing out-of-sample predictions"
        )

    return predictions


# ---------------------------------------------------------
# bootstrap
# ---------------------------------------------------------

def bootstrap_ci(
    values,
    n_boot=10000,
    seed=0,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    samples = rng.choice(
        values,
        size=(n_boot, n),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(values.mean()),
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


# ---------------------------------------------------------
# main analysis
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "numpy",
            "torch",
            "torch_cuda",
        ],
        default="torch_cuda",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BLACK_ROOT
            / "figure3b_results"
        ),
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    backend = configure_backend(
        args.backend
    )

    print(
        f"Backend: {backend.name}",
        flush=True,
    )

    # -----------------------------------------------------
    # parcel selection
    # -----------------------------------------------------

    mapping = np.load(
        MAPPING_FILE,
        allow_pickle=False,
    )

    parcel_ids = np.flatnonzero(
        np.isin(
            mapping,
            LANGUAGE_LABELS,
        )
    )

    print(
        f"Language-network parcels: "
        f"{len(parcel_ids)}",
        flush=True,
    )

    # -----------------------------------------------------
    # subjects
    # -----------------------------------------------------

    response_paths = sorted(
        (DATA_DIR / "Y").glob(
            "black_sub-*.npy"
        )
    )

    # same exclusion as Figure 2 reproduction
    response_paths = [
        p
        for p in response_paths
        if p.stem != "black_sub-296"
    ]

    subjects = [
        p.stem.replace(
            "black_",
            "",
        )
        for p in response_paths
    ]

    responses = [
        load_matrix(p)[
            :,
            parcel_ids,
        ]
        for p in response_paths
    ]

    n_subjects = len(
        responses
    )

    n_parcels = len(
        parcel_ids
    )

    print(
        f"Subjects: {n_subjects}",
        flush=True,
    )

    print(
        f"Parcels: {n_parcels}",
        flush=True,
    )

    # Each target is one subject x parcel.
    #
    # This is the same computational shortcut used
    # in the Figure-2 reconstruction: subjects remain
    # statistically independent targets, but the kernel
    # decomposition is reused.
    joint_y = np.concatenate(
        responses,
        axis=1,
    )

    # -----------------------------------------------------
    # nuisance bands
    # -----------------------------------------------------

    nuisance = [
        load_matrix(
            DATA_DIR
            / "X"
            / f"black_{name}.npy"
        )
        for name in NUISANCE
    ]

    subject_rows = []
    summary_rows = []

    # -----------------------------------------------------
    # model / layer loop
    # -----------------------------------------------------

    for model_name in args.models:

        n_layers = MODELS[
            model_name
        ]

        print()
        print("=" * 70)
        print(
            MODEL_NAMES[
                model_name
            ]
        )
        print("=" * 70)

        for layer in range(
            n_layers
        ):

            print()
            print(
                f"Layer "
                f"{layer + 1}/{n_layers}",
                flush=True,
            )

            emb_file = (
                LAYER_DIR
                / (
                    f"black_{model_name}"
                    f"_layer_{layer:02d}"
                    f"_embeddings.npy"
                )
            )

            trans_file = (
                LAYER_DIR
                / (
                    f"black_{model_name}"
                    f"_layer_{layer:02d}"
                    f"_transformations.npy"
                )
            )

            if not emb_file.exists():
                raise FileNotFoundError(
                    emb_file
                )

            if not trans_file.exists():
                raise FileNotFoundError(
                    trans_file
                )

            embeddings = load_matrix(
                emb_file
            )

            transformations = load_matrix(
                trans_file
            )

            # -------------------------------------------------
            # Embedding encoding model
            # -------------------------------------------------

            print(
                "  embeddings",
                flush=True,
            )

            embedding_predictions = (
                encode_with_predictions(
                    nuisance
                    + [embeddings],
                    joint_y,
                    backend,
                    n_iter=args.n_iter,
                    seed=args.seed,
                )
            )

            # -------------------------------------------------
            # Transformation encoding model
            # -------------------------------------------------

            print(
                "  transformations",
                flush=True,
            )

            transformation_predictions = (
                encode_with_predictions(
                    nuisance
                    + [transformations],
                    joint_y,
                    backend,
                    n_iter=args.n_iter,
                    seed=args.seed,
                )
            )

            # -------------------------------------------------
            # partial correlations
            # -------------------------------------------------

            embedding_partial = (
                partial_correlation(
                    joint_y,
                    embedding_predictions,
                    transformation_predictions,
                )
            )

            transformation_partial = (
                partial_correlation(
                    joint_y,
                    transformation_predictions,
                    embedding_predictions,
                )
            )

            # reshape:
            # subjects x parcels
            embedding_partial = (
                embedding_partial.reshape(
                    n_subjects,
                    n_parcels,
                )
            )

            transformation_partial = (
                transformation_partial.reshape(
                    n_subjects,
                    n_parcels,
                )
            )

            # Figure 3B:
            # average partial correlations across
            # language-network parcels for each subject
            subject_embedding = np.nanmean(
                embedding_partial,
                axis=1,
            )

            subject_transformation = np.nanmean(
                transformation_partial,
                axis=1,
            )

            # -------------------------------------------------
            # save subject-level values
            # -------------------------------------------------

            relative_depth = (
                layer
                / (n_layers - 1)
                if n_layers > 1
                else 0.0
            )

            for s, subject in enumerate(
                subjects
            ):

                subject_rows.append({
                    "model": model_name,
                    "model_name": (
                        MODEL_NAMES[
                            model_name
                        ]
                    ),
                    "layer_zero_based": layer,
                    "layer_one_based": layer + 1,
                    "relative_depth": (
                        relative_depth
                    ),
                    "subject": subject,
                    "embedding_partial_r": float(
                        subject_embedding[s]
                    ),
                    "transformation_partial_r": float(
                        subject_transformation[s]
                    ),
                })

            # -------------------------------------------------
            # bootstrap CI
            # -------------------------------------------------

            emb_mean, emb_low, emb_high = (
                bootstrap_ci(
                    subject_embedding,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
            )

            trans_mean, trans_low, trans_high = (
                bootstrap_ci(
                    subject_transformation,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
            )

            summary_rows.append({
                "model": model_name,
                "model_name": (
                    MODEL_NAMES[
                        model_name
                    ]
                ),
                "layer_zero_based": layer,
                "layer_one_based": layer + 1,
                "relative_depth": (
                    relative_depth
                ),
                "representation": (
                    "embeddings"
                ),
                "mean_partial_r": (
                    emb_mean
                ),
                "ci_low": emb_low,
                "ci_high": emb_high,
                "n_subjects": (
                    n_subjects
                ),
            })

            summary_rows.append({
                "model": model_name,
                "model_name": (
                    MODEL_NAMES[
                        model_name
                    ]
                ),
                "layer_zero_based": layer,
                "layer_one_based": layer + 1,
                "relative_depth": (
                    relative_depth
                ),
                "representation": (
                    "transformations"
                ),
                "mean_partial_r": (
                    trans_mean
                ),
                "ci_low": trans_low,
                "ci_high": trans_high,
                "n_subjects": (
                    n_subjects
                ),
            })

            print(
                f"  embedding partial r: "
                f"{emb_mean:.4f}",
                flush=True,
            )

            print(
                f"  transformation partial r: "
                f"{trans_mean:.4f}",
                flush=True,
            )

            # free large arrays before next layer
            del embeddings
            del transformations
            del embedding_predictions
            del transformation_predictions

    # -----------------------------------------------------
    # write CSVs
    # -----------------------------------------------------

    subject_csv = (
        args.output
        / "figure3b_subject_scores.csv"
    )

    with subject_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                subject_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            subject_rows
        )

    summary_csv = (
        args.output
        / "figure3b_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # -----------------------------------------------------
    # plotting
    # -----------------------------------------------------

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    for model_name in args.models:

        selected = [
            r
            for r in summary_rows
            if r["model"] == model_name
        ]

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        for representation in [
            "embeddings",
            "transformations",
        ]:

            rows = [
                r
                for r in selected
                if r["representation"]
                == representation
            ]

            x = np.array([
                r["layer_one_based"]
                for r in rows
            ])

            y = np.array([
                r["mean_partial_r"]
                for r in rows
            ])

            low = np.array([
                r["ci_low"]
                for r in rows
            ])

            high = np.array([
                r["ci_high"]
                for r in rows
            ])

            label = (
                "Embeddings"
                if representation
                == "embeddings"
                else "Transformations"
            )

            line = ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                label=label,
            )[0]

            ax.fill_between(
                x,
                low,
                high,
                alpha=0.2,
                color=line.get_color(),
            )

        ax.axhline(
            0,
            linewidth=0.8,
            color="black",
        )

        ax.set_xlabel(
            "Layer"
        )

        ax.set_ylabel(
            "Partial correlation"
        )

        ax.set_title(
            MODEL_NAMES[
                model_name
            ]
        )

        ax.legend(
            frameon=False
        )

        ax.spines[
            ["top", "right"]
        ].set_visible(False)

        fig.tight_layout()

        for suffix in [
            "png",
            "pdf",
            "svg",
        ]:

            fig.savefig(
                args.output
                / (
                    f"figure3b_"
                    f"{model_name}."
                    f"{suffix}"
                ),
                dpi=200,
            )

        plt.close(fig)

    print()
    print("=" * 70)
    print("Finished.")
    print(
        f"Results: {args.output}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()