
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Calculate feature-level R², Pearson correlation, "
            "p-values, and Benjamini-Hochberg FDR-adjusted q-values."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/LUAD"),
        help=(
            "Directory containing the predicted and observed morphology "
            "files. Default: results/"
        ),
    )

    parser.add_argument(
        "--predicted-file",
        type=str,
        default="predicted_morphology.csv",
        help=(
            "Name of the predicted morphology CSV file inside "
            "--output-dir. Default: predicted_morphology.csv"
        ),
    )

    parser.add_argument(
        "--observed-file",
        type=str,
        default="true_morphology.csv",
        help=(
            "Name of the observed morphology CSV file inside "
            "--output-dir. Default: true_morphology.csv"
        ),
    )

    parser.add_argument(
        "--results-file",
        type=str,
        default="r2_p_q_scores_by_feature.csv",
        help=(
            "Name of the output metrics CSV file. "
            "Default: r2_p_q_scores_by_feature.csv"
        ),
    )

    parser.add_argument(
        "--sample-id-column",
        type=str,
        default="sample_id",
        help=(
            "Name of the sample identifier column. "
            "Default: sample_id"
        ),
    )

    return parser.parse_args()


# ============================================================
# BENJAMINI-HOCHBERG FDR
# ============================================================
def benjamini_hochberg(
    p_values: np.ndarray,
) -> np.ndarray:
    """

    Parameters
    ----------
    p_values
        One-dimensional array of p-values.

    Returns
    -------
    numpy.ndarray
        Array of Benjamini-Hochberg-adjusted q-values in the same
        order as the input p-values.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    q_values = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p_values)

    if not valid.any():
        return q_values

    valid_p_values = p_values[valid]
    number_of_tests = valid_p_values.size

    order = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[order]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=float,
    )

    sorted_q_values = (
        sorted_p_values
        * number_of_tests
        / ranks
    )

    # Enforce the monotonicity required by the BH procedure.
    sorted_q_values = np.minimum.accumulate(
        sorted_q_values[::-1]
    )[::-1]

    sorted_q_values = np.clip(
        sorted_q_values,
        0.0,
        1.0,
    )

    restored_q_values = np.empty_like(
        sorted_q_values
    )

    restored_q_values[order] = sorted_q_values
    q_values[valid] = restored_q_values

    return q_values


# ============================================================
# INPUT VALIDATION
# ============================================================
def load_morphology_file(
    path: Path,
    sample_id_column: str,
    file_label: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{file_label} file was not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if sample_id_column not in dataframe.columns:
        raise ValueError(
            f"{file_label} file does not contain the required "
            f"sample identifier column '{sample_id_column}': {path}"
        )

    dataframe[sample_id_column] = (
        dataframe[sample_id_column]
        .astype(str)
    )

    dataframe = dataframe.set_index(
        sample_id_column
    )

    if dataframe.index.duplicated().any():
        duplicated_ids = (
            dataframe.index[
                dataframe.index.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{file_label} file contains duplicated sample IDs. "
            f"Examples: {duplicated_ids[:5]}"
        )

    if dataframe.shape[1] == 0:
        raise ValueError(
            f"{file_label} file does not contain any morphology "
            f"feature columns: {path}"
        )

    return dataframe


def validate_and_align_files(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:


    missing_observed_samples = (
        predicted.index.difference(
            observed.index
        )
    )

    if len(missing_observed_samples) > 0:
        raise ValueError(
            f"{len(missing_observed_samples)} predicted samples are "
            "missing from the observed morphology file. Examples: "
            f"{missing_observed_samples[:5].tolist()}"
        )

    predicted_only_features = (
        predicted.columns.difference(
            observed.columns
        )
    )

    observed_only_features = (
        observed.columns.difference(
            predicted.columns
        )
    )

    if len(predicted_only_features) > 0:
        raise ValueError(
            "The following predicted morphology features are absent "
            "from the observed file. Examples: "
            f"{predicted_only_features[:10].tolist()}"
        )

    if len(observed_only_features) > 0:
        raise ValueError(
            "The following observed morphology features are absent "
            "from the predicted file. Examples: "
            f"{observed_only_features[:10].tolist()}"
        )

    # Preserve the exact sample and feature ordering of the prediction file.
    observed = observed.loc[
        predicted.index,
        predicted.columns,
    ].copy()

    predicted = predicted.loc[
        predicted.index,
        predicted.columns,
    ].copy()

    observed = observed.apply(
        pd.to_numeric,
        errors="coerce",
    )

    predicted = predicted.apply(
        pd.to_numeric,
        errors="coerce",
    )

    return observed, predicted


# ============================================================
# FEATURE-LEVEL METRICS
# ============================================================
def calculate_feature_metrics(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
) -> pd.DataFrame:

    records: list[dict[str, object]] = []

    for feature in predicted.columns:
        y_true = observed[feature].to_numpy(
            dtype=float
        )

        y_pred = predicted[feature].to_numpy(
            dtype=float
        )

        complete = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
        )

        y_true = y_true[complete]
        y_pred = y_pred[complete]

        number_of_samples = int(
            y_true.size
        )

        true_is_constant = (
            number_of_samples > 0
            and np.isclose(
                np.std(y_true),
                0.0,
            )
        )

        predicted_is_constant = (
            number_of_samples > 0
            and np.isclose(
                np.std(y_pred),
                0.0,
            )
        )

        # R² requires at least two observations and nonconstant
        # observed values.
        if (
            number_of_samples >= 2
            and not true_is_constant
        ):
            r2 = float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
        else:
            r2 = np.nan

        # Pearson correlation requires at least three observations
        # and variation in both observed and predicted values.
        if (
            number_of_samples >= 3
            and not true_is_constant
            and not predicted_is_constant
        ):
            pearson_r, p_value = pearsonr(
                y_true,
                y_pred,
            )

            pearson_r = float(
                pearson_r
            )

            p_value = float(
                p_value
            )
        else:
            pearson_r = np.nan
            p_value = np.nan

        records.append(
            {
                "feature": feature,
                "n": number_of_samples,
                "r2": r2,
                "pearson_r": pearson_r,
                "p": p_value,
            }
        )

    results = pd.DataFrame.from_records(
        records
    )

    results["q"] = benjamini_hochberg(
        results["p"].to_numpy(
            dtype=float
        )
    )

    results = results.sort_values(
        by=["r2", "q"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(
        drop=True
    )

    return results


# ============================================================
# REPORTING
# ============================================================
def report_results(
    results: pd.DataFrame,
    output_path: Path,
    number_of_samples: int,
) -> None:
    """
    Print a concise evaluation summary.

    Parameters
    ----------
    results
        Feature-level evaluation results.
    output_path
        Path where the results table was saved.
    number_of_samples
        Number of held-out samples evaluated.
    """

    valid_r2 = int(
        results["r2"]
        .notna()
        .sum()
    )

    valid_pearson = int(
        results["pearson_r"]
        .notna()
        .sum()
    )

    significant_features = int(
        (
            results["q"] < 0.05
        ).sum()
    )

    print(
        f"Saved feature-level metrics to:\n{output_path}"
    )

    print(
        f"Held-out samples evaluated: {number_of_samples}"
    )

    print(
        f"Morphology features evaluated: {len(results)}"
    )

    print(
        f"Features with valid R² values: {valid_r2}"
    )

    print(
        "Features with valid Pearson correlations: "
        f"{valid_pearson}"
    )

    print(
        "Features with Pearson q < 0.05: "
        f"{significant_features}"
    )

    print(
        "\nTop five features ranked by R²:"
    )

    print(
        results.head().to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    """
    Run feature-level morphology prediction evaluation.
    """

    args = parse_args()

    output_directory = (
        args.output_dir
    )

    predicted_path = (
        output_directory
        / args.predicted_file
    )

    observed_path = (
        output_directory
        / args.observed_file
    )

    results_path = (
        output_directory
        / args.results_file
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    observed = load_morphology_file(
        path=observed_path,
        sample_id_column=args.sample_id_column,
        file_label="Observed morphology",
    )

    predicted = load_morphology_file(
        path=predicted_path,
        sample_id_column=args.sample_id_column,
        file_label="Predicted morphology",
    )

    observed, predicted = (
        validate_and_align_files(
            observed=observed,
            predicted=predicted,
        )
    )

    results = calculate_feature_metrics(
        observed=observed,
        predicted=predicted,
    )

    results.to_csv(
        results_path,
        index=False,
    )

    report_results(
        results=results,
        output_path=results_path,
        number_of_samples=len(predicted),
    )


if __name__ == "__main__":
    main()
