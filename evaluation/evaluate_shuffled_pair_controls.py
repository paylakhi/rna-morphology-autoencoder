#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


DATASETS = ("LUAD", "LINCS", "CDRP", "TAORF")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate feature-level shuffled-pair control metrics "
            "for LUAD, LINCS, CDRP-bio, and TAORF."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/predictions"),
        help=(
            "Directory containing the observed and predicted morphology "
            "CSV files. Default: results/"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/shuffled_controls"),
        help=(
            "Directory in which output CSV files will be saved. "
            "Default: results/shuffled_controls"
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help=(
            "Datasets to process. Default: LUAD LINCS CDRP TAORF"
        ),
    )

    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=1000,
        help="Number of shuffled-pair permutations. Default: 1000",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility. Default: 42",
    )

    parser.add_argument(
        "--sample-id-column",
        type=str,
        default="sample_id",
        help="Name of the sample identifier column. Default: sample_id",
    )

    parser.add_argument(
        "--save-null-distributions",
        action="store_true",
        help=(
            "Save the complete feature-by-permutation null distributions. "
            "These files can be large."
        ),
    )

    return parser.parse_args()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:

    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)

    valid = np.isfinite(p_values)
    if not valid.any():
        return q_values

    valid_p = p_values[valid]
    number_of_tests = valid_p.size

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    ranks = np.arange(1, number_of_tests + 1, dtype=float)

    sorted_q = sorted_p * number_of_tests / ranks
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)

    restored_q = np.empty_like(sorted_q)
    restored_q[order] = sorted_q

    q_values[valid] = restored_q
    return q_values


def validate_positive_integer(value: int, name: str) -> None:
    """Raise an error when an integer argument is not positive."""
    if value < 1:
        raise ValueError(f"{name} must be at least 1. Received: {value}")


def load_and_align_data(
    observed_path: Path,
    predicted_path: Path,
    sample_id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, validate, and align observed and predicted morphology files.
    """
    if not observed_path.exists():
        raise FileNotFoundError(
            f"Observed morphology file was not found: {observed_path}"
        )

    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Predicted morphology file was not found: {predicted_path}"
        )

    observed_raw = pd.read_csv(observed_path)
    predicted_raw = pd.read_csv(predicted_path)

    if sample_id_column not in observed_raw.columns:
        raise ValueError(
            f"Column '{sample_id_column}' is missing from {observed_path}"
        )

    if sample_id_column not in predicted_raw.columns:
        raise ValueError(
            f"Column '{sample_id_column}' is missing from {predicted_path}"
        )

    observed = observed_raw.set_index(sample_id_column)
    predicted = predicted_raw.set_index(sample_id_column)

    observed.index = observed.index.astype(str)
    predicted.index = predicted.index.astype(str)

    if observed.index.duplicated().any():
        duplicated = observed.index[
            observed.index.duplicated()
        ].unique()[:5].tolist()

        raise ValueError(
            "Observed morphology file contains duplicated sample IDs. "
            f"Examples: {duplicated}"
        )

    if predicted.index.duplicated().any():
        duplicated = predicted.index[
            predicted.index.duplicated()
        ].unique()[:5].tolist()

        raise ValueError(
            "Predicted morphology file contains duplicated sample IDs. "
            f"Examples: {duplicated}"
        )

    missing_observed_samples = predicted.index.difference(observed.index)
    missing_predicted_samples = observed.index.difference(predicted.index)

    if len(missing_observed_samples) > 0:
        raise ValueError(
            f"{len(missing_observed_samples)} predicted sample IDs are "
            "missing from the observed file. Examples: "
            f"{missing_observed_samples[:5].tolist()}"
        )

    if len(missing_predicted_samples) > 0:
        raise ValueError(
            f"{len(missing_predicted_samples)} observed sample IDs are "
            "missing from the predicted file. Examples: "
            f"{missing_predicted_samples[:5].tolist()}"
        )

    missing_observed_features = predicted.columns.difference(
        observed.columns
    )
    missing_predicted_features = observed.columns.difference(
        predicted.columns
    )

    if len(missing_observed_features) > 0:
        raise ValueError(
            "Predicted features absent from the observed file: "
            f"{missing_observed_features[:10].tolist()}"
        )

    if len(missing_predicted_features) > 0:
        raise ValueError(
            "Observed features absent from the predicted file: "
            f"{missing_predicted_features[:10].tolist()}"
        )

    observed = observed.loc[predicted.index, predicted.columns]

    observed = observed.apply(pd.to_numeric, errors="coerce")
    predicted = predicted.apply(pd.to_numeric, errors="coerce")

    return observed, predicted


def calculate_feature_metrics(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate R², Pearson correlation, and Pearson p-value per feature.
    """
    records: list[dict[str, float | int | str]] = []

    for feature in predicted.columns:
        y_true = observed[feature].to_numpy(dtype=float)
        y_pred = predicted[feature].to_numpy(dtype=float)

        complete = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[complete]
        y_pred = y_pred[complete]

        n_samples = int(y_true.size)

        if n_samples < 3:
            records.append(
                {
                    "feature": feature,
                    "n": n_samples,
                    "r2": np.nan,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                }
            )
            continue

        y_true_is_constant = np.isclose(np.std(y_true), 0.0)
        y_pred_is_constant = np.isclose(np.std(y_pred), 0.0)

        r2 = (
            np.nan
            if y_true_is_constant
            else float(r2_score(y_true, y_pred))
        )

        if y_true_is_constant or y_pred_is_constant:
            pearson_r = np.nan
            pearson_p = np.nan
        else:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
            pearson_r = float(pearson_r)
            pearson_p = float(pearson_p)

        records.append(
            {
                "feature": feature,
                "n": n_samples,
                "r2": r2,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
            }
        )

    return pd.DataFrame.from_records(records)


def calculate_shuffled_null_distributions(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    n_shuffles: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate shuffled null distributions for R² and Pearson correlation.

    The same row permutation is applied to all predicted features within
    each shuffle so that the multivariate predicted sample profile remains
    intact while sample pairing is broken.
    """
    rng = np.random.default_rng(seed)

    number_of_features = predicted.shape[1]
    number_of_samples = predicted.shape[0]

    null_r2 = np.full(
        (n_shuffles, number_of_features),
        np.nan,
        dtype=float,
    )

    null_pearson_r = np.full(
        (n_shuffles, number_of_features),
        np.nan,
        dtype=float,
    )

    observed_values = observed.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)

    for shuffle_index in range(n_shuffles):
        permutation = rng.permutation(number_of_samples)
        shuffled_predicted = predicted_values[permutation, :]

        for feature_index in range(number_of_features):
            y_true = observed_values[:, feature_index]
            y_pred = shuffled_predicted[:, feature_index]

            complete = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_complete = y_true[complete]
            y_pred_complete = y_pred[complete]

            if y_true_complete.size < 3:
                continue

            y_true_is_constant = np.isclose(
                np.std(y_true_complete),
                0.0,
            )
            y_pred_is_constant = np.isclose(
                np.std(y_pred_complete),
                0.0,
            )

            if not y_true_is_constant:
                null_r2[shuffle_index, feature_index] = r2_score(
                    y_true_complete,
                    y_pred_complete,
                )

            if not y_true_is_constant and not y_pred_is_constant:
                null_pearson_r[shuffle_index, feature_index] = pearsonr(
                    y_true_complete,
                    y_pred_complete,
                ).statistic

    return null_r2, null_pearson_r


def empirical_upper_tail_p_value(
    observed_value: float,
    null_values: np.ndarray,
) -> float:
    """
    Calculate an upper-tail empirical permutation p-value.

    A +1 correction is used so that the estimated p-value is never zero.
    """
    if not np.isfinite(observed_value):
        return np.nan

    valid_null = null_values[np.isfinite(null_values)]

    if valid_null.size == 0:
        return np.nan

    return float(
        (np.sum(valid_null >= observed_value) + 1)
        / (valid_null.size + 1)
    )


def empirical_absolute_p_value(
    observed_value: float,
    null_values: np.ndarray,
) -> float:
    """
    Calculate a two-sided empirical p-value using absolute correlations.
    """
    if not np.isfinite(observed_value):
        return np.nan

    valid_null = null_values[np.isfinite(null_values)]

    if valid_null.size == 0:
        return np.nan

    return float(
        (
            np.sum(
                np.abs(valid_null) >= abs(observed_value)
            )
            + 1
        )
        / (valid_null.size + 1)
    )


def summarize_null_distributions(
    original_metrics: pd.DataFrame,
    null_r2: np.ndarray,
    null_pearson_r: np.ndarray,
) -> pd.DataFrame:
    """
    Combine original metrics with shuffled-null summary statistics.
    """
    results = original_metrics.copy()

    r2_empirical_p: list[float] = []
    pearson_empirical_p: list[float] = []

    for feature_index in range(len(results)):
        r2_empirical_p.append(
            empirical_upper_tail_p_value(
                observed_value=results.loc[feature_index, "r2"],
                null_values=null_r2[:, feature_index],
            )
        )

        pearson_empirical_p.append(
            empirical_absolute_p_value(
                observed_value=results.loc[
                    feature_index,
                    "pearson_r",
                ],
                null_values=null_pearson_r[:, feature_index],
            )
        )

    results["shuffled_r2_mean"] = np.nanmean(null_r2, axis=0)
    results["shuffled_r2_median"] = np.nanmedian(null_r2, axis=0)
    results["shuffled_r2_ci_lower"] = np.nanpercentile(
        null_r2,
        2.5,
        axis=0,
    )
    results["shuffled_r2_ci_upper"] = np.nanpercentile(
        null_r2,
        97.5,
        axis=0,
    )

    results["shuffled_pearson_r_mean"] = np.nanmean(
        null_pearson_r,
        axis=0,
    )
    results["shuffled_pearson_r_median"] = np.nanmedian(
        null_pearson_r,
        axis=0,
    )
    results["shuffled_pearson_r_ci_lower"] = np.nanpercentile(
        null_pearson_r,
        2.5,
        axis=0,
    )
    results["shuffled_pearson_r_ci_upper"] = np.nanpercentile(
        null_pearson_r,
        97.5,
        axis=0,
    )

    results["r2_empirical_p"] = r2_empirical_p
    results["pearson_empirical_p"] = pearson_empirical_p

    results["r2_empirical_q"] = benjamini_hochberg(
        results["r2_empirical_p"].to_numpy(dtype=float)
    )
    results["pearson_empirical_q"] = benjamini_hochberg(
        results["pearson_empirical_p"].to_numpy(dtype=float)
    )

    results = results.sort_values(
        by=["r2", "r2_empirical_q"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    return results


def null_distribution_to_long_format(
    features: Iterable[str],
    null_values: np.ndarray,
    metric_name: str,
) -> pd.DataFrame:
    """
    Convert a feature-by-permutation null array to long-form format.
    """
    feature_names = list(features)
    n_shuffles, n_features = null_values.shape

    if n_features != len(feature_names):
        raise ValueError(
            "The number of feature names does not match the null array."
        )

    return pd.DataFrame(
        {
            "shuffle": np.repeat(
                np.arange(1, n_shuffles + 1),
                n_features,
            ),
            "feature": np.tile(feature_names, n_shuffles),
            metric_name: null_values.reshape(-1),
        }
    )


def process_dataset(
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    n_shuffles: int,
    seed: int,
    sample_id_column: str,
    save_null_distributions: bool,
) -> pd.DataFrame:
    """
    Process one dataset and save feature-level shuffled-control results.
    """
    observed_path = input_dir / f"{dataset}_true_morphology.csv"
    predicted_path = input_dir / f"{dataset}_predicted.csv"

    observed, predicted = load_and_align_data(
        observed_path=observed_path,
        predicted_path=predicted_path,
        sample_id_column=sample_id_column,
    )

    original_metrics = calculate_feature_metrics(
        observed=observed,
        predicted=predicted,
    )

    null_r2, null_pearson_r = calculate_shuffled_null_distributions(
        observed=observed,
        predicted=predicted,
        n_shuffles=n_shuffles,
        seed=seed,
    )

    results = summarize_null_distributions(
        original_metrics=original_metrics,
        null_r2=null_r2,
        null_pearson_r=null_pearson_r,
    )

    results.insert(0, "dataset", dataset)
    results.insert(1, "n_shuffles", n_shuffles)
    results.insert(2, "seed", seed)

    output_path = (
        output_dir
        / f"shuffled_pair_results_by_feature_{dataset}.csv"
    )

    results.to_csv(output_path, index=False)

    if save_null_distributions:
        r2_long = null_distribution_to_long_format(
            features=predicted.columns,
            null_values=null_r2,
            metric_name="shuffled_r2",
        )
        r2_long.insert(0, "dataset", dataset)

        pearson_long = null_distribution_to_long_format(
            features=predicted.columns,
            null_values=null_pearson_r,
            metric_name="shuffled_pearson_r",
        )
        pearson_long.insert(0, "dataset", dataset)

        r2_long.to_csv(
            output_dir / f"shuffled_r2_null_{dataset}.csv",
            index=False,
        )

        pearson_long.to_csv(
            output_dir / f"shuffled_pearson_null_{dataset}.csv",
            index=False,
        )

    print(
        f"{dataset}: saved {len(results)} feature-level results to "
        f"{output_path}"
    )

    return results


def main() -> None:
    """Run shuffled-pair control evaluation for all requested datasets."""
    args = parse_arguments()

    validate_positive_integer(args.n_shuffles, "n_shuffles")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []

    for dataset_index, dataset in enumerate(args.datasets):
        dataset_seed = args.seed + dataset_index

        dataset_results = process_dataset(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            n_shuffles=args.n_shuffles,
            seed=dataset_seed,
            sample_id_column=args.sample_id_column,
            save_null_distributions=args.save_null_distributions,
        )

        all_results.append(dataset_results)

    combined_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_output_path = (
        output_dir
        / "shuffled_pair_results_by_feature_all_datasets.csv"
    )

    combined_results.to_csv(
        combined_output_path,
        index=False,
    )

    dataset_summary = (
        combined_results.groupby("dataset", as_index=False)
        .agg(
            number_of_features=("feature", "size"),
            mean_original_r2=("r2", "mean"),
            median_original_r2=("r2", "median"),
            mean_shuffled_r2=("shuffled_r2_mean", "mean"),
            median_shuffled_r2=("shuffled_r2_median", "median"),
            significant_r2_features=(
                "r2_empirical_q",
                lambda values: int((values < 0.05).sum()),
            ),
            significant_pearson_features=(
                "pearson_empirical_q",
                lambda values: int((values < 0.05).sum()),
            ),
        )
    )

    summary_output_path = (
        output_dir
        / "shuffled_pair_summary_all_datasets.csv"
    )

    dataset_summary.to_csv(
        summary_output_path,
        index=False,
    )

    print("\nCompleted shuffled-pair control evaluation.")
    print(f"Combined feature results: {combined_output_path}")
    print(f"Dataset summary: {summary_output_path}")


if __name__ == "__main__":
    main()
