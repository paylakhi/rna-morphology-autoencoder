#!/usr/bin/env python3

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

TECHNICAL_COLUMNS = {
    "PERT",
    "Compounds",
    "well",
    "Well",
    "sample_id",
    "Sample_ID",
    "Unnamed: 0",
    "index",
    "x_count_col",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--gene-expression",
        type=Path,
        default=Path("data/gene_expression.csv"),
        help="Preprocessed gene-expression CSV.",
    )

    parser.add_argument(
        "--morphology",
        type=Path,
        default=Path("data/morphology.csv"),
        help="Preprocessed morphology CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/elastic_net"),
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed. Default: {RANDOM_SEED}",
    )

    return parser.parse_args()


def clean_numeric_table(
    path: Path,
    table_name: str,
) -> pd.DataFrame:
    raw = pd.read_csv(path)

    metadata_columns = [
        column
        for column in raw.columns
        if str(column).lower().startswith("metadata_")
    ]

    explicit_technical = [
        column
        for column in raw.columns
        if column in TECHNICAL_COLUMNS
    ]

    cleaned = raw.drop(
        columns=list(dict.fromkeys(metadata_columns + explicit_technical)),
        errors="ignore",
    ).copy()

    cleaned = cleaned.apply(
        pd.to_numeric,
        errors="coerce",
    )

    cleaned = cleaned.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    cleaned = cleaned.dropna(
        axis=1,
        how="all",
    )

    cleaned = cleaned.fillna(0.0)

    constant_columns = [
        column
        for column in cleaned.columns
        if cleaned[column].nunique(dropna=False) <= 1
    ]

    cleaned = cleaned.drop(
        columns=constant_columns,
        errors="ignore",
    )

    if cleaned.shape[1] == 0:
        raise ValueError(
            f"No usable numeric columns remain in {table_name}."
        )

    print(
        f"{table_name} final shape:",
        cleaned.shape,
    )

    return cleaned


def make_fixed_split(
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_indices = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return train_idx, val_idx, test_idx


def mean_feature_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    values = []

    for column_index in range(
        observed.shape[1]
    ):
        y_true = observed[:, column_index]
        y_pred = predicted[:, column_index]

        if np.var(y_true) <= 1e-12:
            continue

        values.append(
            r2_score(
                y_true,
                y_pred,
            )
        )

    if not values:
        raise ValueError(
            "No non-constant morphology features were available for R²."
        )

    return float(
        np.mean(values)
    )


def save_predictions(
    output_dir: Path,
    feature_names: list[str],
    predicted: np.ndarray,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predicted_df = pd.DataFrame(
        predicted,
        columns=feature_names,
    )

    predicted_df.to_csv(
        output_dir
        / "predicted_morphology.csv",
        index=False,
    )


from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet

ELASTIC_NET_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

L1_RATIOS = [
    0.1,
    0.5,
    0.9,
]

MAX_ITERATIONS = 100000
TOLERANCE = 1e-4


def main() -> None:
    args = parse_args()

    X_df = clean_numeric_table(
        args.gene_expression,
        "Gene expression",
    )

    Y_df = clean_numeric_table(
        args.morphology,
        "Morphology",
    )

    if len(X_df) != len(Y_df):
        raise ValueError(
            "Gene-expression and morphology matrices must contain "
            "the same matched samples in identical row order."
        )

    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)

    train_idx, val_idx, test_idx = make_fixed_split(
        n_samples=len(X),
        seed=args.seed,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_val = Y[val_idx]

    best_alpha = None
    best_l1_ratio = None
    best_val_r2 = -np.inf

    for alpha in ELASTIC_NET_ALPHAS:
        for l1_ratio in L1_RATIOS:
            model = MultiTaskElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=MAX_ITERATIONS,
                tol=TOLERANCE,
                random_state=args.seed,
                selection="cyclic",
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    category=ConvergenceWarning,
                )

                model.fit(
                    X_train,
                    Y_train,
                )

            validation_predictions = model.predict(
                X_val
            )

            validation_r2 = mean_feature_r2(
                Y_val,
                validation_predictions,
            )

            if validation_r2 > best_val_r2:
                best_val_r2 = validation_r2
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    final_model = MultiTaskElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        fit_intercept=True,
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE,
        random_state=args.seed,
        selection="cyclic",
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=ConvergenceWarning,
        )

        final_model.fit(
            X_train,
            Y_train,
        )

    test_predictions = final_model.predict(
        X_test
    )

    save_predictions(
        output_dir=args.output_dir,
        feature_names=(
            Y_df.columns
            .astype(str)
            .tolist()
        ),
        predicted=test_predictions,
    )

    print("Elastic Net baseline complete.")
    print("Selected alpha:", best_alpha)
    print("Selected l1_ratio:", best_l1_ratio)
    print(
        "Saved:",
        args.output_dir / "predicted_morphology.csv",
    )


if __name__ == "__main__":
    main()
