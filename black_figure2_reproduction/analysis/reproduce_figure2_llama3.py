"""Auditable Figure 2 pipeline for pre-aligned, pre-lagged (TR, parcel) data.

Run --help; see ../FIGURE2_REPRODUCTION.md for scientific assumptions.
No truncation recovery, temporal trimming, or additional lagging is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from encoding_backend import configure_backend, standardize_train_test

ROOT = Path(__file__).resolve().parents[1]
# MODELS = ["ling_features", "glove", "zrep_magnitudes", "full", "full_z", "llama3_full"]
# MODEL_NAMES = ["Linguistic features", "GLoVe", "BERT: transformation magnitudes",
#                "BERT: embeddings", "BERT: transformations",
#                "Llama 3.2 3B: embeddings"]

# MODELS = [
#     "ling_features",
#     "glove",

#     "zrep_magnitudes",
#     "full",
#     "full_z",

#     "llama3_zrep_magnitudes",
#     "llama3_full",
#     "llama3_full_z",
# ]

# MODEL_NAMES = [
#     "Linguistic features",
#     "GLoVe",

#     "BERT: transformation magnitudes",
#     "BERT: embeddings",
#     "BERT: transformations",

#     "Llama 3.2 3B: transformation magnitudes",
#     "Llama 3.2 3B: embeddings",
#     "Llama 3.2 3B: transformations",
# ]

MODELS = [
    "ling_features",
    "glove",

    "zrep_magnitudes",
    "full",
    "full_z",

    "llama3_zrep_magnitudes",
    "llama3_full",
    "llama3_full_z",

    "llama4_zrep_magnitudes",
    "llama4_full",
    "llama4_full_z",
]

MODEL_NAMES = [
    "Linguistic features",
    "GLoVe",

    "BERT: transformation magnitudes",
    "BERT: embeddings",
    "BERT: transformations",

    "Llama 3.2 3B: transformation magnitudes",
    "Llama 3.2 3B: embeddings",
    "Llama 3.2 3B: transformations",

    "Llama 4 Scout: transformation magnitudes",
    "Llama 4 Scout: embeddings",
    "Llama 4 Scout: transformations",
]
ROI_NAMES = ["PostTemp", "AntTemp", "AngG", "IFG", "MFG", "IFGorb",
             "vmPFC", "dmPFC", "PMC", "HG", "V1"]
DISPLAY_IDS = [10, 1, 2, 3, 4, 6, 5, 7, 8, 9]
NUISANCE = ["nuisance_silent_", "nuisance_number_", "nuisance_phonemes_"]


def inspect_npy(path):
    result = {"path": str(path), "actual_bytes": path.stat().st_size}
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            #shape, order, dtype = np.lib.format._read_array_header(stream, version)
            shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
            expected = stream.tell() + int(np.prod(shape)) * dtype.itemsize
        result.update(shape=list(shape), dtype=str(dtype), expected_bytes=expected,
                      fortran_order=order)
        if dtype.hasobject:
            raise ValueError("Object arrays are unsupported")
        if expected != result["actual_bytes"]:
            raise ValueError(f"Expected {expected} bytes, found {result['actual_bytes']}")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError("Array must contain finite numeric values")
        result["valid"] = True
    except (ValueError, OSError, EOFError) as error:
        result.update(valid=False, error=str(error))
    return result


def validate_mapping(path):
    mapping = np.load(path, allow_pickle=False)
    if mapping.shape != (1000,) or not np.isfinite(mapping).all():
        raise ValueError("ROI mapping must be a finite (1000,) array")
    if not np.equal(mapping, mapping.astype(int)).all() or not np.isin(mapping, range(23)).all():
        raise ValueError("ROI mapping must have integer labels 0..22")
    if np.any(mapping[:500] > 11) or np.any((mapping[500:] > 0) & (mapping[500:] < 12)):
        raise ValueError("ROI labels disagree with Schaefer LH-first parcel ordering")
    if any(not np.any(mapping == label) for label in range(1, 23)):
        raise ValueError("One or more language/control ROIs are empty")
    return mapping.astype(int)


def audit(data_dir, mapping_path):
    files = [inspect_npy(p) for folder in ("X", "Y")
             for p in sorted((data_dir / folder).glob("*.npy"))]
    issues = []
    for story in ("black",):
        ys = [p for p in files if Path(p["path"]).parent.name == "Y"
              and Path(p["path"]).name.startswith(story + "_")]
        if not ys:
            issues.append(f"No response files for {story}")
        lengths = {p["shape"][0] for p in ys if len(p.get("shape", [])) == 2}
        if len(lengths) != 1:
            issues.append(f"Inconsistent or missing response lengths for {story}: {sorted(lengths)}")
        for p in ys:
            if len(p.get("shape", [])) == 2 and p["shape"][1] != 1000:
                issues.append(f"Expected (TR, 1000) responses: {p['path']}")
        for model in MODELS + NUISANCE:
            path = data_dir / "X" / f"{story}_{model}.npy"
            entry = next((f for f in files if f["path"] == str(path)), None)
            if entry is None:
                issues.append(f"Missing {path}")
            elif len(entry.get("shape", [])) != 2 or entry["shape"][0] not in lengths:
                issues.append(f"Feature/response TR mismatch: {path}")
    try:
        mapping = validate_mapping(mapping_path)
        roi_info = {str(i): int(np.sum(mapping == i)) for i in range(23)}
    except (ValueError, OSError) as error:
        issues.append(str(error))
        roi_info = None
    return {"files": files, "invalid_files": sum(not f["valid"] for f in files),
            "issues": issues, "roi_counts": roi_info,
            "ready": bool(files) and all(f["valid"] for f in files) and not issues}


def load_matrix(path):
    check = inspect_npy(path)
    if not check["valid"]:
        raise ValueError(f"Invalid data file {path}: {check['error']}")
    data = np.load(path, allow_pickle=False)
    if data.ndim != 2:
        raise ValueError(f"Expected a matrix at {path}")
    return data.astype("float32")


def select_embeddings(x, mode):
    if mode == "saved":
        return x
    # Existing exporter horizontally stacks four delays, each containing layers 0..12.
    if x.shape[1] == 4 * 13 * 768:
        return x.reshape(len(x), 4, 13, 768)[:, :, 1:, :].reshape(len(x), -1)
    if x.shape[1] == 4 * 12 * 768:
        return x
    raise ValueError("Cannot identify 12/13 BERT layers in four saved delay blocks")


def column_correlation(x, y):
    x = np.asarray(x, dtype="float64") - np.mean(x, axis=0)
    y = np.asarray(y, dtype="float64") - np.mean(y, axis=0)
    denominator = np.sqrt(np.sum(x*x, axis=0) * np.sum(y*y, axis=0))
    return np.divide(np.sum(x*y, axis=0), denominator,
                     out=np.full(x.shape[1], np.nan), where=denominator > 0)


def compute_isc(responses):
    """Self-inclusive group mean, evaluated on the same three outer test folds."""
    from sklearn.model_selection import KFold
    if len(responses) < 2:
        raise ValueError("ISC needs at least two subjects")
    mean = np.mean(np.stack(responses), axis=0, dtype="float64")
    folds = list(KFold(3, shuffle=False).split(mean))
    return np.stack([np.stack([column_correlation(y[test], mean[test])
                              for _, test in folds], axis=1) for y in responses])


def encode(bands, y, backend, n_iter=100, seed=0, solver="kernel", alphas=(1.0,)):
    """Four-band regression; return main-band r and log band weights per fold."""
    from himalaya.scoring import correlation_score
    from sklearn.model_selection import KFold
    from himalaya.ridge import GroupRidgeCV
    from himalaya.kernel_ridge import MultipleKernelRidgeCV
    scores, deltas = [], []
    for fold, (train, test) in enumerate(KFold(3, shuffle=False).split(y), 1):
        start = time.monotonic()
        pairs = [standardize_train_test(x[train], x[test]) for x in bands]
        train_bands, test_bands = map(list, zip(*pairs))
        params = dict(n_iter=n_iter, score_func=correlation_score, progress_bar=False,
                      alphas=np.asarray(alphas, dtype="float32"), n_targets_batch=512)
        common = dict(cv=KFold(3, shuffle=False), fit_intercept=False,
                      random_state=seed, solver_params=params)
        if solver == "kernel":
            # Algebraically equivalent to group ridge; smaller than a 37k-feature SVD.
            train_gpu = [backend.asarray(x) for x in train_bands]
            test_gpu = [backend.asarray(x) for x in test_bands]
            xtrain = backend.stack([x @ x.T for x in train_gpu])
            xtest = backend.stack([x @ t.T for x, t in zip(test_gpu, train_gpu)])
            model = MultipleKernelRidgeCV(kernels="precomputed", **common)
        else:
            xtrain, xtest = train_bands, test_bands
            model = GroupRidgeCV(groups="input", **common)
        model.fit(xtrain, y[train])
        prediction = backend.to_numpy(model.predict(xtest, split=True)[-1])
        scores.append(column_correlation(prediction, y[test]))
        deltas.append(backend.to_numpy(model.deltas_).T)
        print(f"  Fold {fold}/3: {time.monotonic()-start:.1f}s", flush=True)
    return np.stack(scores, axis=1), np.stack(deltas, axis=1)


def roi_scores(correlations, ceiling, mapping):
    """Mean folds -> Fisher z -> mean ROI -> ratio, matching ROI notebook function."""
    eps = np.finfo(float).eps
    r = np.arctanh(np.clip(correlations.mean(axis=1), -1+eps, 1-eps))
    c = np.arctanh(np.clip(ceiling.mean(axis=1), -1+eps, 1-eps))
    values = []
    for label in DISPLAY_IDS:
        mask = mapping == label
        numerator, denominator = np.mean(r[mask]), np.mean(c[mask])
        values.append(100 * numerator / denominator if denominator > 0 else np.nan)
    return np.asarray(values)


def summarize(rows, output, n_boot=10000, seed=0, stratified=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    summaries = []
    fig, ax = plt.subplots(figsize=(12, 5))
    #colors = ["black", "#999999", "#efa517", "#2454d7", "#e13c36", "#6a3d9a"]
    # colors = [
    #     "black",      # linguistic
    #     "#999999",    # GLoVe

    #     "#efa517",    # BERT magnitudes
    #     "#2454d7",    # BERT embeddings
    #     "#e13c36",    # BERT transformations

    #     "#33a02c",    # Llama magnitudes
    #     "#6a3d9a",    # Llama embeddings
    #     "#1f9e89",    # Llama transformations
    # ]
    colors = [
        "black",      # linguistic
        "#999999",    # GLoVe

        "#efa517",    # BERT magnitudes
        "#2454d7",    # BERT embeddings
        "#e13c36",    # BERT transformations

        "#33a02c",    # Llama 3 magnitudes
        "#6a3d9a",    # Llama 3 embeddings
        "#1f9e89",    # Llama 3 transformations

        "#bcbd22",    # Llama 4 magnitudes
        "#8c564b",    # Llama 4 embeddings
        "#17becf",    # Llama 4 transformations
    ]
    for m, model in enumerate(MODELS):
        for j, roi_id in enumerate(DISPLAY_IDS):
            selected = [r for r in rows if r["model"] == model and r["roi"] == ROI_NAMES[roi_id-1]]
            values = np.array([r["percent_noise_ceiling"] for r in selected])
            if len(values) == 0 or not np.isfinite(values).all():
                raise ValueError(f"Missing or undefined ROI scores for {model}, ROI {roi_id}")
            groups = ([np.array([k for k, r in enumerate(selected) if r['story'] == story])
                       for story in sorted({r['story'] for r in selected})]
                      if stratified else [np.arange(len(values))])
            boot = np.median(np.concatenate([values[rng.choice(g, (n_boot, len(g)))]
                                             for g in groups], axis=1), axis=1)
            median = float(np.median(values))
            low, high = np.percentile(boot, [2.5, 97.5])
            summaries.append(dict(model=model, roi=ROI_NAMES[roi_id-1], n=len(values),
                                  median=median, ci_low=float(low), ci_high=float(high)))
            #x = j + (m - (len(MODELS)-1)/2)*0.13
            x = j + (m - (len(MODELS) - 1) / 2) * 0.11
            ax.vlines(x, low, high, color=colors[m], lw=1.5)
            ax.plot(x, median, 'o', color=colors[m], ms=5,
                    label=MODEL_NAMES[m] if j == 0 else None)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(10), [ROI_NAMES[i-1] for i in DISPLAY_IDS], rotation=45, ha="right")
    ax.set_ylabel("Percent noise ceiling")
    ax.set_title("Figure 2 reproduction — I Knew You Were Black")
    ax.spines[["top", "right"]].set_visible(False)
    #ax.legend(frameon=False, fontsize=9)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"figure2.{suffix}", dpi=200)
    plt.close(fig)
    write_csv(output / "figure2_summary.csv", summaries)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "save_data")
    parser.add_argument("--mapping", type=Path, default=ROOT / "Parcel2ROI_new.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch", "torch_cuda"], default="auto")
    parser.add_argument("--solver", choices=["kernel", "primal"], default="kernel")
    parser.add_argument("--embedding-layers", choices=["paper", "saved"], default="paper")
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooled-bootstrap", action="store_true")
    parser.add_argument("--include-excluded", action="store_true", help="Include black sub-296 in plotted sample")
    parser.add_argument("--all-parcels", action="store_true", help="Fit all 1000 parcels; default fits only Figure 2 LH ROIs")
    parser.add_argument("--clip-percent", action="store_true", help="Clip ROI percentages to [-100, 100] as in plotting notebook")
    args = parser.parse_args()
    if args.n_iter < 1 or args.n_boot < 1 or any(a <= 0 for a in args.alphas):
        parser.error("Iterations, bootstrap samples, and alphas must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report = audit(args.data_dir, args.mapping)
    (args.output / "input_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audit: {len(report['files'])} arrays, {report['invalid_files']} invalid; ready={report['ready']}")
    if not report["ready"]:
        print(f"Restore complete inputs; see {args.output / 'input_audit.json'}", file=sys.stderr)
        return 2
    if args.command == "audit":
        return 0
    backend = configure_backend(args.backend)
    mapping = validate_mapping(args.mapping)
    parcel_ids = np.arange(1000) if args.all_parcels else np.flatnonzero(np.isin(mapping, DISPLAY_IDS))
    mapping = mapping[parcel_ids]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["actual_backend"] = backend.name
    for key in ("data_dir", "mapping", "output"):
        path = getattr(args, key).resolve()
        config[key] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    # Hash all inputs so cached subject results cannot mix datasets or analysis choices.
    config["input_sha256"] = {}
    for path in [args.mapping] + [Path(f['path']) for f in report['files']]:
        with path.open("rb") as stream:
            config["input_sha256"][str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    config["parcel_ids_zero_based"] = parcel_ids.tolist()
    config["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes() +
                                          Path(__file__).with_name('encoding_backend.py').read_bytes()).hexdigest()
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output contains a different run configuration; choose another --output")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for story in ("black",):
        paths = sorted((args.data_dir / "Y").glob(f"{story}_sub-*.npy"))
        responses = [load_matrix(p)[:, parcel_ids] for p in paths]
        ceilings = compute_isc(responses)
        np.savez_compressed(args.output / f"{story}_isc.npz", subjects=[p.stem for p in paths],
                            parcel_ids=parcel_ids, r=ceilings)
        nuisance = [load_matrix(args.data_dir / "X" / f"{story}_{name}.npy") for name in NUISANCE]
        for model in MODELS:
            features = load_matrix(args.data_dir / "X" / f"{story}_{model}.npy")
            if model == "full":
                features = select_embeddings(features, args.embedding_layers)
            included = [i for i, p in enumerate(paths)
                        if args.include_excluded or p.stem != "black_sub-296"]
            cache = args.output / f"{story}_{model}.npz"
            if cache.exists():
                with np.load(cache) as saved:
                    all_r = saved['r']
            else:
                # Subject-parcel targets are independent. Concatenation reuses the same
                # kernel decompositions without averaging subjects or sharing weights.
                print(f"Fitting {story} {model}: {len(included)} subjects, {len(parcel_ids)} parcels", flush=True)
                joint_y = np.concatenate([responses[i] for i in included], axis=1)
                all_r, deltas = encode(nuisance + [features], joint_y, backend, args.n_iter,
                                       args.seed, args.solver, args.alphas)
                all_r = all_r.reshape(len(included), len(parcel_ids), 3)
                np.savez_compressed(cache, r=all_r, log_band_weights=deltas.reshape(
                    len(included), len(parcel_ids), 3, 4),
                    subjects=[paths[i].stem for i in included], parcel_ids=parcel_ids)
            for batch_index, index in enumerate(included):
                path, r = paths[index], all_r[batch_index]
                scores = roi_scores(r, ceilings[index], mapping)
                for roi_id, score in zip(DISPLAY_IDS, scores):
                    rows.append(dict(story=story, subject=path.stem.split('_', 1)[1], model=model,
                                     roi=ROI_NAMES[roi_id-1], percent_noise_ceiling=float(
                                         np.clip(score, -100, 100) if args.clip_percent else score),
                                     unclipped_percent_noise_ceiling=float(score)))
    write_csv(args.output / "figure2_subject_scores.csv", rows)
    summarize(rows, args.output, args.n_boot, args.seed, not args.pooled_bootstrap)
    print(f"Saved Figure 2 and numerical results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
