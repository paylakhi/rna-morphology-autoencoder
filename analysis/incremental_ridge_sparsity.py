#!/usr/bin/env python3

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# MANUSCRIPT FEATURES
# ============================================================================

OUR_17_FEATURES = [
    "Nuclei_RadialDistribution_RadialCV_Mito_1of4",
    "Cells_AreaShape_Zernike_3_1",
    "Cells_RadialDistribution_RadialCV_AGP_1of4",
    "Cells_RadialDistribution_RadialCV_AGP_4of4",
    "Nuclei_RadialDistribution_FracAtD_Mito_3of4",
    "Cells_AreaShape_Zernike_4_2",
    "Cells_RadialDistribution_RadialCV_Mito_1of4",
    "Nuclei_Granularity_5_DNA",
    "Cells_RadialDistribution_MeanFrac_AGP_3of4",
    "Cytoplasm_AreaShape_Zernike_4_4",
    "Cells_RadialDistribution_RadialCV_AGP_3of4",
    "Nuclei_RadialDistribution_RadialCV_DNA_1of4",
    "Cells_AreaShape_Zernike_2_2",
    "Cytoplasm_RadialDistribution_RadialCV_AGP_1of4",
    "Nuclei_RadialDistribution_RadialCV_AGP_3of4",
    "Nuclei_RadialDistribution_MeanFrac_Mito_3of4",
    "Cytoplasm_AreaShape_Zernike_9_9",
]


# ============================================================================
# DEFAULT ANALYSIS SETTINGS
# ============================================================================

RANDOM_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
TARGET_FRACTION = 0.90
RIDGE_ALPHAS = np.logspace(-4, 4, 17)

# Figure colors
NAVY = "#233B8B"
GRAY = "#A0A0A0"
DARK_GRAY = "#444444"


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the cross-validated incremental-R² Ridge sparsity analysis "
            "for the selected morphology features."
        )
    )

    parser.add_argument(
        "--gene-expression",
        type=Path,
        default=Path("data/gene_expression.csv"),
        help="Gene-expression CSV. Default: data/gene_expression.csv",
    )

    parser.add_argument(
        "--morphology",
        type=Path,
        default=Path("data/morphology.csv"),
        help="Morphology CSV. Default: data/morphology.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/incremental_ridge"),
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed. Default: {RANDOM_SEED}",
    )

    parser.add_argument(
        "--outer-folds",
        type=int,
        default=N_OUTER_FOLDS,
        help=f"Number of outer CV folds. Default: {N_OUTER_FOLDS}",
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=N_INNER_FOLDS,
        help=f"Maximum number of inner CV folds. Default: {N_INNER_FOLDS}",
    )

    parser.add_argument(
        "--target-fraction",
        type=float,
        default=TARGET_FRACTION,
        help="Fraction of full-model R² used for k90. Default: 0.90",
    )

    return parser.parse_args()


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data(
    gene_path: Path,
    morphology_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean matched gene-expression and morphology matrices."""

    gene_df = pd.read_csv(gene_path)
    morph_df = pd.read_csv(morphology_path)

    print("Gene-expression shape:", gene_df.shape)
    print("Morphology shape:", morph_df.shape)

    if gene_df.shape[0] != morph_df.shape[0]:
        raise ValueError(
            "Gene-expression and morphology files must contain the same "
            "number of samples and must already be aligned in the same row "
            "order. "
            f"Gene-expression rows={gene_df.shape[0]}, "
            f"morphology rows={morph_df.shape[0]}."
        )

    # Remove common technical columns from the predictor matrix if present.
    technical_columns = [
        column
        for column in gene_df.columns
        if (
            str(column).lower() == "x_count_col"
            or str(column).lower().startswith("metadata_")
        )
    ]

    if technical_columns:
        print("Removing technical predictor columns:", technical_columns)
        gene_df = gene_df.drop(
            columns=technical_columns,
            errors="ignore",
        )

    # Convert remaining values to numeric.
    X_all = gene_df.apply(
        pd.to_numeric,
        errors="coerce",
    )
    Y_all = morph_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove columns with no usable numeric values.
    X_all = X_all.dropna(axis=1, how="all")
    Y_all = Y_all.dropna(axis=1, how="all")

    # Remove constant predictors.
    gene_variance = X_all.var(
        axis=0,
        skipna=True,
    )
    constant_genes = gene_variance[
        gene_variance <= 1e-12
    ].index.tolist()

    if constant_genes:
        print(
            f"Removing {len(constant_genes)} constant predictor columns."
        )
        X_all = X_all.drop(columns=constant_genes)

    if X_all.shape[1] < 2:
        raise ValueError(
            "Too few usable gene-expression predictors for analysis."
        )

    print("\nFinal predictor matrix:", X_all.shape)
    print("Final morphology matrix:", Y_all.shape)

    return X_all, Y_all


def get_features_to_test(Y_all: pd.DataFrame) -> list[str]:
    features_to_test = [
        feature
        for feature in OUR_17_FEATURES
        if feature in Y_all.columns
    ]

    missing = [
        feature
        for feature in OUR_17_FEATURES
        if feature not in Y_all.columns
    ]

    print("\nFeatures available:", len(features_to_test))
    print("Features missing:", len(missing))

    for feature in missing:
        print("Missing:", feature)

    if not features_to_test:
        raise ValueError("None of the 17 manuscript morphology features were found.")

    return features_to_test


def choose_k_values(n_total_genes: int) -> list[int]:

    candidate_k_values = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        150,
        n_total_genes,
    ]

    k_values = sorted({
        int(k)
        for k in candidate_k_values
        if 1 <= k <= n_total_genes
    })

    if n_total_genes not in k_values:
        k_values.append(n_total_genes)

    return sorted(set(k_values))


# ============================================================================
# TRAINING-FOLD GENE RANKING
# ============================================================================

def rank_genes_by_training_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:

    y_numeric = pd.to_numeric(
        y_train,
        errors="coerce",
    ).to_numpy(dtype=float)

    ranking_rows = []

    for gene in X_train.columns:
        x_numeric = pd.to_numeric(
            X_train[gene],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(x_numeric)
            & np.isfinite(y_numeric)
        )

        x_valid = x_numeric[valid]
        y_valid = y_numeric[valid]

        if (
            len(x_valid) < 3
            or np.std(x_valid) <= 1e-12
            or np.std(y_valid) <= 1e-12
        ):
            correlation = 0.0
        else:
            try:
                correlation, _ = pearsonr(
                    x_valid,
                    y_valid,
                )
                if not np.isfinite(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0

        ranking_rows.append({
            "gene": gene,
            "training_pearson_r": correlation,
            "training_absolute_r": abs(correlation),
        })

    ranking_df = (
        pd.DataFrame(ranking_rows)
        .sort_values(
            ["training_absolute_r", "gene"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    ranking_df["training_rank"] = np.arange(
        1,
        len(ranking_df) + 1,
    )

    return ranking_df


# ============================================================================
# RIDGE MODEL WITH INNER-CV HYPERPARAMETER SELECTION
# ============================================================================

def build_ridge_search(
    n_training_samples: int,
    inner_folds_max: int,
    seed: int,
) -> GridSearchCV:
    """
    Build a median-imputation + standardization + Ridge pipeline.

    Preprocessing and Ridge-alpha selection are fitted only on the current
    outer training fold.
    """

    inner_folds = min(
        inner_folds_max,
        max(
            2,
            n_training_samples // 10,
        ),
    )

    if inner_folds >= n_training_samples:
        inner_folds = max(2, n_training_samples - 1)

    inner_cv = KFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(),
            ),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            "ridge__alpha": RIDGE_ALPHAS,
        },
        scoring="neg_mean_squared_error",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )


# ============================================================================
# OUTER CROSS-VALIDATION
# ============================================================================

def run_incremental_ridge(
    X_all: pd.DataFrame,
    Y_all: pd.DataFrame,
    features_to_test: list[str],
    k_values: list[int],
    output_dir: Path,
    seed: int,
    outer_folds: int,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete training-fold ranking and held-out Ridge analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    outer_cv = KFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_results: list[dict] = []
    gene_rankings: list[pd.DataFrame] = []
    failed_features: list[dict] = []

    n_total_genes = X_all.shape[1]

    for feature_number, feature in enumerate(
        features_to_test,
        start=1,
    ):
        print(
            f"\nFeature {feature_number}/{len(features_to_test)}: {feature}"
        )

        y_feature = pd.to_numeric(
            Y_all[feature],
            errors="coerce",
        )

        # Keep rows with a finite morphology target and at least one finite
        # RNA predictor.
        finite_rna = np.isfinite(
            X_all.to_numpy(dtype=float)
        ).any(axis=1)

        valid_rows = (
            np.isfinite(y_feature.to_numpy(dtype=float))
            & finite_rna
        )

        X_feature = (
            X_all.loc[valid_rows]
            .reset_index(drop=True)
        )

        y_feature = (
            y_feature.loc[valid_rows]
            .reset_index(drop=True)
        )

        if len(y_feature) < outer_folds * 3:
            failed_features.append({
                "morphology_feature": feature,
                "error": "Too few valid samples",
            })
            print("Skipped: too few valid samples.")
            continue

        try:
            for fold_number, (
                train_indices,
                test_indices,
            ) in enumerate(
                outer_cv.split(X_feature),
                start=1,
            ):
                X_train = X_feature.iloc[train_indices].copy()
                X_test = X_feature.iloc[test_indices].copy()
                y_train = y_feature.iloc[train_indices].copy()
                y_test = y_feature.iloc[test_indices].copy()

                # Rank genes using this outer training fold only.
                ranking_df = rank_genes_by_training_correlation(
                    X_train=X_train,
                    y_train=y_train,
                )

                ranking_df["morphology_feature"] = feature
                ranking_df["fold"] = fold_number
                gene_rankings.append(ranking_df)

                ordered_genes = ranking_df["gene"].tolist()

                for k in k_values:
                    selected_genes = ordered_genes[:k]

                    ridge_search = build_ridge_search(
                        n_training_samples=len(train_indices),
                        inner_folds_max=inner_folds,
                        seed=seed,
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ridge_search.fit(
                            X_train[selected_genes],
                            y_train,
                        )

                    predictions = ridge_search.predict(
                        X_test[selected_genes]
                    )

                    heldout_r2 = r2_score(
                        y_test,
                        predictions,
                    )

                    heldout_mse = mean_squared_error(
                        y_test,
                        predictions,
                    )

                    best_alpha = ridge_search.best_params_[
                        "ridge__alpha"
                    ]

                    fold_results.append({
                        "morphology_feature": feature,
                        "fold": fold_number,
                        "k_genes": k,
                        "percentage_genes": (
                            100.0 * k / n_total_genes
                        ),
                        "is_full_model": (
                            k == n_total_genes
                        ),
                        "heldout_r2": heldout_r2,
                        "heldout_mse": heldout_mse,
                        "best_alpha": best_alpha,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "selected_genes": "|".join(selected_genes),
                    })

                print(
                    f"  Fold {fold_number}/{outer_folds} completed."
                )

        except Exception as error:
            print("FAILED:", error)
            failed_features.append({
                "morphology_feature": feature,
                "error": str(error),
            })

    fold_results_df = pd.DataFrame(fold_results)

    gene_rankings_df = (
        pd.concat(
            gene_rankings,
            ignore_index=True,
        )
        if gene_rankings
        else pd.DataFrame()
    )

    failed_features_df = pd.DataFrame(failed_features)

    if fold_results_df.empty:
        raise RuntimeError(
            "No incremental-R² results were generated."
        )

    fold_results_df.to_csv(
        output_dir / "incremental_r2_fold_results.csv",
        index=False,
    )

    gene_rankings_df.to_csv(
        output_dir / "training_fold_gene_rankings.csv",
        index=False,
    )

    failed_features_df.to_csv(
        output_dir / "failed_features.csv",
        index=False,
    )

    return (
        fold_results_df,
        gene_rankings_df,
        failed_features_df,
    )


# ============================================================================
# SUMMARIZE HELD-OUT R² CURVES
# ============================================================================

def summarize_curves(
    fold_results_df: pd.DataFrame,
    n_total_genes: int,
    target_fraction: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fold-level performance and calculate relative performance."""

    curve_summary_df = (
        fold_results_df
        .groupby(
            [
                "morphology_feature",
                "k_genes",
                "percentage_genes",
            ],
            as_index=False,
        )
        .agg(
            mean_heldout_r2=("heldout_r2", "mean"),
            sd_heldout_r2=("heldout_r2", "std"),
            median_heldout_r2=("heldout_r2", "median"),
            mean_heldout_mse=("heldout_mse", "mean"),
            mean_best_alpha=("best_alpha", "mean"),
            n_folds=("fold", "nunique"),
        )
    )

    full_model_summary = (
        curve_summary_df[
            curve_summary_df["k_genes"] == n_total_genes
        ][
            [
                "morphology_feature",
                "mean_heldout_r2",
                "sd_heldout_r2",
            ]
        ]
        .rename(
            columns={
                "mean_heldout_r2": "full_model_mean_r2",
                "sd_heldout_r2": "full_model_sd_r2",
            }
        )
    )

    curve_summary_df = curve_summary_df.merge(
        full_model_summary,
        on="morphology_feature",
        how="left",
    )

    curve_summary_df["relative_to_full_r2"] = np.where(
        curve_summary_df["full_model_mean_r2"] > 0,
        (
            curve_summary_df["mean_heldout_r2"]
            / curve_summary_df["full_model_mean_r2"]
        ),
        np.nan,
    )

    curve_summary_df["target_90pct_full_r2"] = (
        target_fraction
        * curve_summary_df["full_model_mean_r2"]
    )

    # Display-only clipping; original relative values remain unchanged.
    curve_summary_df["relative_to_full_r2_display"] = (
        curve_summary_df["relative_to_full_r2"]
        .clip(lower=0, upper=1.05)
    )

    curve_summary_df.to_csv(
        output_dir / "incremental_r2_curve_summary.csv",
        index=False,
    )

    return curve_summary_df, full_model_summary


# ============================================================================
# k90 CALCULATION
# ============================================================================

def interpolate_k90(
    feature_curve: pd.DataFrame,
    target_fraction: float,
) -> float:

    curve = (
        feature_curve[
            [
                "k_genes",
                "relative_to_full_r2",
            ]
        ]
        .dropna()
        .sort_values("k_genes")
        .reset_index(drop=True)
    )

    if curve.empty:
        return np.nan

    k_values = curve["k_genes"].to_numpy(dtype=float)
    relative_values = curve[
        "relative_to_full_r2"
    ].to_numpy(dtype=float)

    crossing_indices = np.where(
        relative_values >= target_fraction
    )[0]

    if crossing_indices.size == 0:
        return np.nan

    upper_index = int(crossing_indices[0])

    if upper_index == 0:
        return float(k_values[0])

    lower_index = upper_index - 1

    k_lower = float(k_values[lower_index])
    k_upper = float(k_values[upper_index])
    r_lower = float(relative_values[lower_index])
    r_upper = float(relative_values[upper_index])

    if np.isclose(r_upper, r_lower):
        return k_upper

    interpolation_fraction = (
        (target_fraction - r_lower)
        / (r_upper - r_lower)
    )

    return float(
        k_lower
        + interpolation_fraction
        * (k_upper - k_lower)
    )


def calculate_k90_summary(
    curve_summary_df: pd.DataFrame,
    n_total_genes: int,
    target_fraction: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Calculate tested and interpolated k90 values for every feature."""

    rows = []

    for feature in sorted(
        curve_summary_df["morphology_feature"].unique()
    ):
        feature_curve = (
            curve_summary_df[
                curve_summary_df["morphology_feature"] == feature
            ]
            .sort_values("k_genes")
            .copy()
        )

        full_r2 = float(
            feature_curve["full_model_mean_r2"].iloc[0]
        )

        if (
            not np.isfinite(full_r2)
            or full_r2 <= 0
        ):
            k90_tested = np.nan
            k90_interpolated = np.nan
            reached = False
            note = (
                "Undefined because full-model held-out R2 "
                "is nonpositive"
            )
        else:
            eligible = feature_curve[
                feature_curve["relative_to_full_r2"]
                >= target_fraction
            ]

            if eligible.empty:
                k90_tested = np.nan
                k90_interpolated = np.nan
                reached = False
                note = (
                    "90% target not reached among tested "
                    "gene-set sizes"
                )
            else:
                k90_tested = int(
                    eligible["k_genes"].min()
                )
                k90_interpolated = interpolate_k90(
                    feature_curve,
                    target_fraction,
                )
                reached = True
                note = ""

        rows.append({
            "morphology_feature": feature,
            "full_model_mean_r2": full_r2,
            "target_fraction": target_fraction,
            "target_90pct_full_r2": (
                target_fraction * full_r2
                if np.isfinite(full_r2) and full_r2 > 0
                else np.nan
            ),
            "k90_tested": k90_tested,
            "k90_interpolated": k90_interpolated,
            "k90_interpolated_rounded": (
                int(np.round(k90_interpolated))
                if np.isfinite(k90_interpolated)
                else np.nan
            ),
            "k90_fraction_of_all_genes": (
                k90_interpolated / n_total_genes
                if np.isfinite(k90_interpolated)
                else np.nan
            ),
            "k90_percentage_of_all_genes": (
                100.0 * k90_interpolated / n_total_genes
                if np.isfinite(k90_interpolated)
                else np.nan
            ),
            "reached_90pct_target": reached,
            "note": note,
        })

    k90_summary_df = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "k90_percentage_of_all_genes",
                "full_model_mean_r2",
            ],
            ascending=[True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    k90_summary_df.to_csv(
        output_dir / "incremental_r2_k90_summary.csv",
        index=False,
    )

    return k90_summary_df


# ============================================================================
# FINAL TWO-PANEL FIGURE
# ============================================================================

def format_feature_label(feature_name: str) -> str:

    parts = str(feature_name).split("_")

    if len(parts) <= 2:
        return str(feature_name)

    return (
        "_".join(parts[:2])
        + "\n"
        + "_".join(parts[2:])
    )


def create_two_panel_figure(
    curve_summary_df: pd.DataFrame,
    k90_summary_df: pd.DataFrame,
    n_total_genes: int,
    target_fraction: float,
    output_dir: Path,
) -> None:

 
    desired_plot_k = [
        2,
        10,
        50,
        100,
        150,
        n_total_genes,
    ]

    selected_k_values = [
        k
        for k in desired_plot_k
        if k <= n_total_genes
        and k in set(curve_summary_df["k_genes"].unique())
    ]

    # Guarantee inclusion of the full-gene model.
    if (
        n_total_genes in set(curve_summary_df["k_genes"].unique())
        and n_total_genes not in selected_k_values
    ):
        selected_k_values.append(n_total_genes)

    selected_k_values = list(dict.fromkeys(selected_k_values))

    if len(selected_k_values) < 2:
        raise ValueError(
            "Too few directly tested gene-set sizes are available "
            "for the final figure."
        )

    if n_total_genes == 187:
        label_lookup = {
            2: "1%",
            10: "5%",
            50: "25%",
            100: "50%",
            150: "75%",
            187: "100%",
        }
    else:
        # For a different panel size, display rounded actual percentages.
        label_lookup = {
            k: f"{int(round(100 * k / n_total_genes))}%"
            for k in selected_k_values
        }

    positive_full_features = (
        k90_summary_df.loc[
            k90_summary_df["full_model_mean_r2"] > 0,
            "morphology_feature",
        ]
        .drop_duplicates()
        .tolist()
    )

    relative_plot_df = (
        curve_summary_df[
            (
                curve_summary_df["morphology_feature"].isin(
                    positive_full_features
                )
            )
            & (
                curve_summary_df["k_genes"].isin(
                    selected_k_values
                )
            )
        ]
        .dropna(
            subset=["relative_to_full_r2_display"]
        )
        .copy()
    )

    plot_x_by_k = {
        k: i
        for i, k in enumerate(selected_k_values)
    }

    relative_plot_df["plot_x"] = (
        relative_plot_df["k_genes"].map(plot_x_by_k)
    )

    relative_median_curve = (
        relative_plot_df
        .groupby(
            ["k_genes", "plot_x"],
            as_index=False,
        )
        .agg(
            median_relative_r2=(
                "relative_to_full_r2_display",
                "median",
            ),
            q25_relative_r2=(
                "relative_to_full_r2_display",
                lambda x: x.quantile(0.25),
            ),
            q75_relative_r2=(
                "relative_to_full_r2_display",
                lambda x: x.quantile(0.75),
            ),
        )
        .sort_values("plot_x")
        .reset_index(drop=True)
    )

    plot_k90_df = (
        k90_summary_df[
            (
                k90_summary_df["full_model_mean_r2"] > 0
            )
            & (
                k90_summary_df[
                    "k90_percentage_of_all_genes"
                ].notna()
            )
        ]
        .sort_values(
            "k90_percentage_of_all_genes",
            ascending=True,
        )
        .copy()
    )

    if plot_k90_df.empty:
        raise ValueError(
            "No valid k90 values were available for Panel B."
        )

    panel_b_labels = [
        format_feature_label(feature)
        for feature in plot_k90_df["morphology_feature"]
    ]

    # Equal-sized panels.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(19, 8.5),
        gridspec_kw={"width_ratios": [1, 1]},
        constrained_layout=True,
    )

    ax_curve = axes[0]
    ax_k90 = axes[1]

    # ---------------------------------------------------------------------
    # PANEL A
    # ---------------------------------------------------------------------

    for feature in positive_full_features:
        feature_curve = (
            relative_plot_df[
                relative_plot_df["morphology_feature"] == feature
            ]
            .sort_values("plot_x")
        )

        if feature_curve.empty:
            continue

        ax_curve.plot(
            feature_curve["plot_x"],
            feature_curve["relative_to_full_r2_display"],
            linewidth=1.3,
            alpha=0.28,
            color="#C9CDD3",
            solid_capstyle="round",
            zorder=1,
        )

    ax_curve.fill_between(
        relative_median_curve["plot_x"],
        relative_median_curve["q25_relative_r2"],
        relative_median_curve["q75_relative_r2"],
        color=NAVY,
        alpha=0.24,
        linewidth=0,
        label="Interquartile range",
        zorder=2,
    )

    ax_curve.plot(
        relative_median_curve["plot_x"],
        relative_median_curve["median_relative_r2"],
        marker="o",
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=0.9,
        linewidth=3.8,
        color=NAVY,
        label="Median across features",
        zorder=3,
    )

    ax_curve.axhline(
        y=target_fraction,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="90% of full-model $R^2$",
        zorder=2,
    )

    ax_curve.axhline(
        y=1.0,
        color=DARK_GRAY,
        linestyle=":",
        linewidth=1.3,
        label="Full-gene Ridge model",
        zorder=2,
    )

    ax_curve.set_xticks(
        np.arange(len(selected_k_values))
    )

    ax_curve.set_xticklabels(
        [
            label_lookup[k]
            for k in selected_k_values
        ],
        fontsize=14,
        fontweight="bold",
    )

    ax_curve.set_xlim(
        -0.30,
        len(selected_k_values) - 0.70,
    )

    ax_curve.set_ylim(0, 1.06)

    ax_curve.set_xlabel(
        "Percentage of top-ranked genes included",
        fontsize=15,
        fontweight="bold",
        labelpad=9,
    )

    ax_curve.set_ylabel(
        "Fraction of full-model held-out Ridge $R^2$",
        fontsize=15,
        fontweight="bold",
        labelpad=9,
    )

    ax_curve.set_title(
        "Relative held-out Ridge performance",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )

    ax_curve.tick_params(
        axis="both",
        labelsize=14,
        colors="black",
        width=1.5,
        length=5,
    )

    for tick_label in ax_curve.get_xticklabels():
        tick_label.set_fontweight("bold")

    for tick_label in ax_curve.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax_curve.grid(
        True,
        linestyle="--",
        linewidth=1.0,
        alpha=0.14,
    )

    legend = ax_curve.legend(
        frameon=True,
        fontsize=12,
        loc="lower right",
        handlelength=2.5,
        labelspacing=0.35,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(1.5)

    ax_curve.text(
        -0.12,
        1.055,
        "a",
        transform=ax_curve.transAxes,
        fontsize=22,
        fontweight="bold",
        verticalalignment="top",
    )

    # ---------------------------------------------------------------------
    # PANEL B
    # ---------------------------------------------------------------------

    y_positions = np.arange(len(plot_k90_df))

    ax_k90.barh(
        y_positions,
        plot_k90_df["k90_percentage_of_all_genes"],
        height=0.70,
        color=NAVY,
        alpha=0.95,
        zorder=3,
    )

    ax_k90.set_yticks(y_positions)

    ax_k90.set_yticklabels(
        panel_b_labels,
        fontsize=14,
        fontweight="bold",
        linespacing=0.92,
    )

    ax_k90.axvline(
        x=100,
        color=DARK_GRAY,
        linestyle=":",
        linewidth=1.8,
        zorder=2,
    )

    ax_k90.set_xlim(0, 103)

    ax_k90.set_xticks(
        [0, 20, 40, 60, 80, 100]
    )

    ax_k90.set_xticklabels(
        ["0%", "20%", "40%", "60%", "80%", "100%"],
        fontsize=14,
        fontweight="bold",
    )

    ax_k90.set_xlabel(
        "Percentage of top-ranked genes included",
        fontsize=15,
        fontweight="bold",
        labelpad=9,
    )

    ax_k90.set_ylabel("")

    ax_k90.set_title(
        "Gene panel required\n"
        "for 90% of full-model performance",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )

    ax_k90.tick_params(
        axis="both",
        labelsize=14,
        colors="black",
        width=1.5,
        length=5,
    )

    for tick_label in ax_k90.get_xticklabels():
        tick_label.set_fontweight("bold")

    for tick_label in ax_k90.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax_k90.grid(
        axis="x",
        linestyle="--",
        linewidth=1.0,
        alpha=0.18,
        zorder=0,
    )

    ax_k90.text(
        -0.12,
        1.055,
        "b",
        transform=ax_k90.transAxes,
        fontsize=22,
        fontweight="bold",
        verticalalignment="top",
    )

    # Darker plot borders.
    for ax in [ax_curve, ax_k90]:
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)

    # Force identical plotting-box dimensions.
    ax_curve.set_box_aspect(1)
    ax_k90.set_box_aspect(1)

    output_prefix = (
        output_dir
        / "incremental_ridge_sparsity_two_panel"
    )

    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_prefix.with_suffix(".svg"),
        bbox_inches="tight",
    )
    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    # Save exactly the data displayed in the two panels.
    relative_median_curve.to_csv(
        output_dir / "figure_panel_a_relative_summary.csv",
        index=False,
    )
    plot_k90_df.to_csv(
        output_dir / "figure_panel_b_k90_percentages.csv",
        index=False,
    )


# ============================================================================
# OVERALL SUMMARY
# ============================================================================

def save_overall_summary(
    k90_summary_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    eligible = k90_summary_df[
        k90_summary_df[
            "k90_percentage_of_all_genes"
        ].notna()
    ].copy()

    summary_df = pd.DataFrame([{
        "n_features_total": len(k90_summary_df),
        "n_features_positive_full_r2": int(
            (
                k90_summary_df["full_model_mean_r2"] > 0
            ).sum()
        ),
        "n_features_reaching_90pct": int(
            k90_summary_df["reached_90pct_target"].sum()
        ),
        "median_full_model_r2": (
            k90_summary_df.loc[
                k90_summary_df["full_model_mean_r2"] > 0,
                "full_model_mean_r2",
            ].median()
        ),
        "median_k90_genes": (
            eligible["k90_interpolated"].median()
            if not eligible.empty
            else np.nan
        ),
        "median_k90_percentage": (
            eligible["k90_percentage_of_all_genes"].median()
            if not eligible.empty
            else np.nan
        ),
        "q25_k90_percentage": (
            eligible["k90_percentage_of_all_genes"].quantile(0.25)
            if not eligible.empty
            else np.nan
        ),
        "q75_k90_percentage": (
            eligible["k90_percentage_of_all_genes"].quantile(0.75)
            if not eligible.empty
            else np.nan
        ),
    }])

    summary_df.to_csv(
        output_dir / "incremental_r2_overall_summary.csv",
        index=False,
    )

    return summary_df


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_all, Y_all = load_and_prepare_data(
        gene_path=args.gene_expression,
        morphology_path=args.morphology,
    )

    features_to_test = get_features_to_test(Y_all)

    n_total_genes = X_all.shape[1]
    k_values = choose_k_values(n_total_genes)

    print("\nAnalysis settings")
    print("-----------------")
    print("Samples:", X_all.shape[0])
    print("Total gene predictors:", n_total_genes)
    print("Morphology features:", len(features_to_test))
    print("Incremental k values:", k_values)
    print("Outer CV folds:", args.outer_folds)
    print("Inner CV folds (maximum):", args.inner_folds)
    print("Target fraction:", args.target_fraction)

    fold_results_df, _, failed_features_df = run_incremental_ridge(
        X_all=X_all,
        Y_all=Y_all,
        features_to_test=features_to_test,
        k_values=k_values,
        output_dir=args.output_dir,
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
    )

    curve_summary_df, _ = summarize_curves(
        fold_results_df=fold_results_df,
        n_total_genes=n_total_genes,
        target_fraction=args.target_fraction,
        output_dir=args.output_dir,
    )

    k90_summary_df = calculate_k90_summary(
        curve_summary_df=curve_summary_df,
        n_total_genes=n_total_genes,
        target_fraction=args.target_fraction,
        output_dir=args.output_dir,
    )

    overall_summary_df = save_overall_summary(
        k90_summary_df=k90_summary_df,
        output_dir=args.output_dir,
    )

    create_two_panel_figure(
        curve_summary_df=curve_summary_df,
        k90_summary_df=k90_summary_df,
        n_total_genes=n_total_genes,
        target_fraction=args.target_fraction,
        output_dir=args.output_dir,
    )

    print("\nAnalysis complete.")
    print("Successful features:", fold_results_df[
        "morphology_feature"
    ].nunique())
    print("Failed features:", len(failed_features_df))
    print("\nOverall summary:")
    print(overall_summary_df.to_string(index=False))
    print("\nOutputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
