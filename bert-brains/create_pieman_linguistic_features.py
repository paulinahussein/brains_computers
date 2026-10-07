#!/usr/bin/env python3
"""Create the 39 linguistic features used by the Figure 2 analysis.

Run this script from anywhere inside/on a copy of the bert-brains repository.
It parses the Pieman transcript with spaCy, aligns the parsed tokens to the 277
fMRI TRs, and writes both a plain encoding matrix and a labelled NPZ archive.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import spacy


DEP_LABELS = [
    "nsubj", "ROOT", "advmod", "prep", "det",
    "pobj", "aux", "dobj", "cc", "ccomp",
    "amod", "compound", "acomp", "poss", "xcomp",
    "conj", "relcl", "attr", "mark", "npadvmod",
    "advcl", "neg", "prt", "nummod", "intj",
]

POS_LABELS = [
    "PRON", "VERB", "NOUN", "DET", "AUX", "ADP", "ADV",
    "CCONJ", "ADJ", "PART", "PROPN", "SCONJ", "NUM", "INTJ",
]


def find_repo_root(start: Path) -> Path:
    """Find a parent containing both data/ and text_processing_and_transformers/."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "data").is_dir()
            and (candidate / "text_processing_and_transformers").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find the bert-brains repository. Use --repo-root explicitly."
    )


def load_alignment_words(path: Path) -> list[str]:
    # align.csv has no header; its first column contains the original words.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        words = [row[0].strip() for row in csv.reader(handle) if row and row[0].strip()]
    if not words:
        raise ValueError(f"No words found in {path}")
    return words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Path to bert-brains (normally detected automatically).",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="Installed English spaCy model (default: en_core_web_lg).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = (
        args.repo_root.expanduser().resolve()
        if args.repo_root
        else find_repo_root(Path.cwd().resolve())
    )
    data_dir = repo_root / "data" / "pieman"
    feature_dir = repo_root / "text_processing_and_transformers"
    align_path = data_dir / "align.csv"
    tr_path = data_dir / "tr_tokens.csv"

    for path in (align_path, tr_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")

    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model {args.spacy_model!r} is not installed.\n"
            f"Install it with: python -m spacy download {args.spacy_model}"
        ) from exc

    words = load_alignment_words(align_path)
    tr_data = pd.read_csv(tr_path, keep_default_na=False)
    if "tokens" not in tr_data.columns:
        raise ValueError(f"{tr_path} has no 'tokens' column")

    # This follows the original repository: parse the complete transcript once.
    doc = nlp(" ".join(words))
    parsed_tokens = [token for token in doc if token.pos_ != "SPACE"]

    # Tokenize each TR exactly as the original alignment loop does.
    tr_token_counts = [
        len(nlp.make_doc(str(text))) if str(text).strip() else 0
        for text in tr_data["tokens"]
    ]
    expected_tokens = sum(tr_token_counts)
    if len(parsed_tokens) != expected_tokens:
        raise ValueError(
            "Token alignment failed: the complete transcript produced "
            f"{len(parsed_tokens)} spaCy tokens, but the TR strings produced "
            f"{expected_tokens}. Use the same spaCy model/version as the original "
            "pipeline or inspect align.csv and tr_tokens.csv."
        )

    feature_rows: list[list[int]] = []
    aligned_rows: list[dict[str, object]] = []
    cursor = 0
    for tr_row, count in zip(tr_data.to_dict("records"), tr_token_counts):
        tokens = parsed_tokens[cursor : cursor + count]
        cursor += count
        dep_values = [token.dep_ for token in tokens]
        pos_values = [token.pos_ for token in tokens]
        feature_rows.append(
            [dep_values.count(label) for label in DEP_LABELS]
            + [pos_values.count(label) for label in POS_LABELS]
        )
        aligned_rows.append(
            {
                **tr_row,
                "lemma": [token.lemma_ for token in tokens],
                "pos": pos_values,
                "tag": [token.tag_ for token in tokens],
                "dep_rel": dep_values,
                "dep_head": [token.head.text for token in tokens],
                "dep_distance": [abs(token.i - token.head.i) for token in tokens],
            }
        )

    features = np.asarray(feature_rows, dtype=np.float32)
    expected_shape = (len(tr_data), len(DEP_LABELS) + len(POS_LABELS))
    if features.shape != expected_shape:
        raise AssertionError(f"Unexpected output shape: {features.shape} != {expected_shape}")
    if not np.isfinite(features).all():
        raise ValueError("Output contains NaN or infinite values")

    feature_dir.mkdir(parents=True, exist_ok=True)
    npy_path = feature_dir / "pieman_ling_features.npy"
    npz_path = feature_dir / "pieman_ling_features.npz"
    aligned_path = data_dir / "tr_tokens_new.csv"
    details_path = data_dir / "ling_features.csv"

    np.save(npy_path, features)
    np.savez(
        npz_path,
        ling_features=features,
        dep_list=np.asarray(DEP_LABELS),
        pos_list=np.asarray(POS_LABELS),
    )
    pd.DataFrame(aligned_rows).to_csv(aligned_path, index=False)
    pd.DataFrame(
        {
            "token": [token.text for token in parsed_tokens],
            "lemma": [token.lemma_ for token in parsed_tokens],
            "pos": [token.pos_ for token in parsed_tokens],
            "tag": [token.tag_ for token in parsed_tokens],
            "dep_rel": [token.dep_ for token in parsed_tokens],
            "dep_head": [token.head.text for token in parsed_tokens],
            "dep_distance": [abs(token.i - token.head.i) for token in parsed_tokens],
        }
    ).to_csv(details_path, index=False)

    print(f"Repository: {repo_root}")
    print(f"Words in align.csv: {len(words)}")
    print(f"TRs: {len(tr_data)} ({sum(c > 0 for c in tr_token_counts)} with text)")
    print(f"Aligned spaCy tokens: {expected_tokens}")
    print(f"Saved encoding matrix: {npy_path}")
    print(f"Saved labelled archive: {npz_path}")
    print(f"Shape: {features.shape}")
    print(f"Dtype: {features.dtype}")
    print(f"NaNs: {np.isnan(features).sum()}")


if __name__ == "__main__":
    main()
