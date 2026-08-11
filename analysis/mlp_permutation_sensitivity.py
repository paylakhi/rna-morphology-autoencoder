#!/usr/bin/env python3

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor


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
# SETTINGS
# ============================================================================

RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 20
TOP_N = 20

MLP_PARAMS = {
    "hidden_layer_sizes": (32,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.1,
    "batch_size": 8,
    "learning_rate_init": 0.0005,
    "max_iter": 3000,
    "early_stopping": True,
    "validation_fraction": 0.20,
    "n_iter_no_change": 50,
    "random_state": RANDOM_SEED,
}

NAVY = "#000080"
LIGHT_NAVY = "#B8B8D8"

# Representative features shown in Supplementary Figure S6.
REPRESENTATIVE_FEATURES = [
    "Cells_RadialDistribution_RadialCV_AGP_1of4",
    "Cells_AreaShape_Zernike_4_2",
]


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the nonlinear MLP permutation-importance sensitivity "
            "analysis for the 17 manuscript morphology features."
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
        "--gene-annotation",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping predictor column IDs to gene symbols. "
            "Expected columns: gene,gene_symbol."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mlp_sensitivity"),
        help="Output directory.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help=f"Number of top genes retained per feature. Default: {TOP_N}",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=N_SPLITS,
        help=f"Number of cross-validation folds. Default: {N_SPLITS}",
    )

    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=N_REPEATS,
        help=f"Permutation repeats per held-out fold. Default: {N_REPEATS}",
    )

    return parser.parse_args()


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_numeric_matrices(
    gene_path: Path,
    morphology_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    gene_df = pd.read_csv(gene_path)
    morph_df = pd.read_csv(morphology_path)

    print("Gene-expression shape:", gene_df.shape)
    print("Morphology shape:", morph_df.shape)

    # Remove common non-model columns if present.
    technical_columns = {
        "Metadata_Well",
        "PERT",
        "Compounds",
        "Metadata_well_position",
        "Metadata_Well_Position",
        "well",
        "Well",
        "sample_id",
        "Sample_ID",
        "Unnamed: 0",
        "index",
        "x_count_col",
    }

    gene_feature_columns = [
        column
        for column in gene_df.columns
        if (
            column not in technical_columns
            and pd.api.types.is_numeric_dtype(gene_df[column])
        )
    ]

    morph_feature_columns = [
        column
        for column in morph_df.columns
        if (
            column not in technical_columns
            and pd.api.types.is_numeric_dtype(morph_df[column])
        )
    ]

    X_all = gene_df[gene_feature_columns].copy()
    Y_all = morph_df[morph_feature_columns].copy()

    if X_all.empty:
        raise ValueError("No numeric gene-expression predictors were found.")

    if Y_all.empty:
        raise ValueError("No numeric morphology features were found.")

    print("Numeric gene predictors:", X_all.shape[1])
    print("Numeric morphology features:", Y_all.shape[1])

    return X_all, Y_all


def get_features_to_test(
    Y_all: pd.DataFrame,
) -> list[str]:
    missing = [
        feature
        for feature in OUR_17_FEATURES
        if feature not in Y_all.columns
    ]

    if missing:
        raise ValueError(
            "The morphology matrix is missing manuscript features:\n"
            + "\n".join(missing)
        )

    return OUR_17_FEATURES.copy()


def load_gene_annotation(
    annotation_path: Path | None,
) -> dict[str, str]:
    """Load an optional predictor-ID to gene-symbol mapping."""

    if annotation_path is None:
        return {}

    annotation_df = pd.read_csv(annotation_path)

    required = {"gene", "gene_symbol"}
    missing = required - set(annotation_df.columns)

    if missing:
        raise ValueError(
            "Gene annotation file is missing columns: "
            + ", ".join(sorted(missing))
        )

    mapping = {}

    for row in annotation_df[
        ["gene", "gene_symbol"]
    ].dropna(subset=["gene"]).itertuples(index=False):
        gene = str(row.gene)
        symbol = str(row.gene_symbol).strip()

        if symbol and symbol.lower() != "nan":
            # If multiple symbols are supplied, display the first.
            symbol = symbol.split("///")[0].strip()
            mapping[gene] = symbol

    return mapping


# ============================================================================
# ONE-FEATURE MLP ANALYSIS
# ============================================================================

def analyze_feature_with_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    feature_name: str,
    top_n: int,
    n_splits: int,
    n_repeats: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    X = X.copy()
    y = pd.to_numeric(
        y,
        errors="coerce",
    ).copy()

    valid_rows = (
        np.isfinite(
            y.to_numpy(dtype=float)
        )
        &
        np.isfinite(
            X.to_numpy(dtype=float)
        ).all(axis=1)
    )

    X_valid = (
        X.loc[valid_rows]
        .reset_index(drop=True)
    )

    y_valid = (
        y.loc[valid_rows]
        .reset_index(drop=True)
    )

    if len(y_valid) < n_splits * 2:
        raise ValueError(
            f"Too few valid samples for {feature_name}."
        )

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_importances = []
    fold_metrics = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        kfold.split(X_valid),
        start=1,
    ):
        X_train = X_valid.iloc[
            train_index
        ]

        X_test = X_valid.iloc[
            test_index
        ]

        y_train = y_valid.iloc[
            train_index
        ]

        y_test = y_valid.iloc[
            test_index
        ]

        model = MLPRegressor(
            **MLP_PARAMS
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=ConvergenceWarning,
            )
            model.fit(
                X_train,
                y_train,
            )

        predictions = model.predict(
            X_test
        )

        fold_metrics.append({
            "morphology_feature": feature_name,
            "fold": fold_number,
            "r2": r2_score(
                y_test,
                predictions,
            ),
            "mse": mean_squared_error(
                y_test,
                predictions,
            ),
            "n_train": len(train_index),
            "n_test": len(test_index),
            "n_iterations": model.n_iter_,
        })

        # Held-out permutation importance.
        importance = permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            scoring="neg_mean_squared_error",
            n_repeats=n_repeats,
            random_state=(
                RANDOM_SEED
                + fold_number
            ),
            n_jobs=-1,
        )

        fold_importances.append(
            pd.DataFrame({
                "gene": X_valid.columns,
                "importance": (
                    importance.importances_mean
                ),
                "importance_sd_within_fold": (
                    importance.importances_std
                ),
                "fold": fold_number,
            })
        )

    fold_importance_df = pd.concat(
        fold_importances,
        ignore_index=True,
    )

    mean_importance_df = (
        fold_importance_df
        .groupby(
            "gene",
            as_index=False,
        )
        .agg(
            mean_permutation_importance=(
                "importance",
                "mean",
            ),
            sd_permutation_importance=(
                "importance",
                "std",
            ),
            positive_importance_folds=(
                "importance",
                lambda values:
                    int(
                        (values > 0).sum()
                    ),
            ),
        )
        .sort_values(
            "mean_permutation_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    top_genes = (
        mean_importance_df
        .head(top_n)
        .copy()
    )

    top_genes["rank"] = np.arange(
        1,
        len(top_genes) + 1,
    )

    correlation_rows = []

    for row in top_genes.itertuples(
        index=False
    ):
        gene_values = (
            X_valid[row.gene]
            .to_numpy(dtype=float)
        )

        morphology_values = (
            y_valid.to_numpy(dtype=float)
        )

        pearson_r, pearson_p = pearsonr(
            gene_values,
            morphology_values,
        )

        spearman_rho, spearman_p = (
            spearmanr(
                gene_values,
                morphology_values,
            )
        )

        correlation_rows.append({
            "model": "MLPRegressor",
            "morphology_feature": feature_name,
            "category": assign_feature_category(
                feature_name
            ),
            "gene": row.gene,
            "rank": row.rank,
            "mean_permutation_importance":
                row.mean_permutation_importance,
            "sd_permutation_importance":
                row.sd_permutation_importance,
            "positive_importance_folds":
                row.positive_importance_folds,
            "pearson_r": pearson_r,
            "absolute_pearson_r": abs(
                pearson_r
            ),
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "absolute_spearman_rho": abs(
                spearman_rho
            ),
            "spearman_p": spearman_p,
            "n_samples": len(y_valid),
        })

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    return (
        pd.DataFrame(
            correlation_rows
        ),
        mean_importance_df,
        metrics_df,
    )


# ============================================================================
# RUN ALL 17 FEATURES
# ============================================================================

def run_all_features(
    X_all: pd.DataFrame,
    Y_all: pd.DataFrame,
    features_to_test: list[str],
    output_dir: Path,
    top_n: int,
    n_splits: int,
    n_repeats: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the MLP sensitivity analysis for all 17 morphology features."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_top_results = []
    all_importance_results = []
    all_metrics = []
    failed_features = []

    for index, feature in enumerate(
        features_to_test,
        start=1,
    ):
        print(
            f"\nFeature {index}/"
            f"{len(features_to_test)}: "
            f"{feature}"
        )

        try:
            (
                top_results,
                gene_importances,
                fold_metrics,
            ) = analyze_feature_with_mlp(
                X=X_all,
                y=Y_all[feature],
                feature_name=feature,
                top_n=top_n,
                n_splits=n_splits,
                n_repeats=n_repeats,
            )

            gene_importances[
                "morphology_feature"
            ] = feature

            gene_importances[
                "category"
            ] = assign_feature_category(
                feature
            )

            all_top_results.append(
                top_results
            )

            all_importance_results.append(
                gene_importances
            )

            all_metrics.append(
                fold_metrics
            )

            print("Completed.")

        except Exception as error:
            print("FAILED:", error)

            failed_features.append({
                "morphology_feature":
                    feature,
                "error":
                    str(error),
            })

    if not all_top_results:
        raise RuntimeError(
            "No MLP feature analyses completed successfully."
        )

    top_results_df = pd.concat(
        all_top_results,
        ignore_index=True,
    )

    importance_results_df = pd.concat(
        all_importance_results,
        ignore_index=True,
    )

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    failed_df = pd.DataFrame(
        failed_features
    )

    top_results_df.to_csv(
        output_dir
        / "MLP_top20_gene_results_all_17_features.csv",
        index=False,
    )

    importance_results_df.to_csv(
        output_dir
        / "MLP_all_gene_importances_all_17_features.csv",
        index=False,
    )

    metrics_df.to_csv(
        output_dir
        / "MLP_fold_metrics_all_17_features.csv",
        index=False,
    )

    failed_df.to_csv(
        output_dir
        / "MLP_failed_features.csv",
        index=False,
    )

    return (
        top_results_df,
        importance_results_df,
        metrics_df,
        failed_df,
    )


# ============================================================================
# SUMMARY OF MARGINAL CORRELATIONS
# ============================================================================

def save_correlation_summary(
    top_results_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Summarize marginal correlations among top MLP-ranked gene-feature pairs."""

    absolute_r = (
        top_results_df[
            "absolute_pearson_r"
        ]
        .dropna()
    )

    summary_df = pd.DataFrame({
        "metric": [
            "Number of morphology features",
            "Number of top-ranked gene-feature pairs",
            "Median absolute Pearson correlation",
            "Mean absolute Pearson correlation",
            "25th percentile absolute Pearson correlation",
            "75th percentile absolute Pearson correlation",
            "Maximum absolute Pearson correlation",
            "Percentage with |r| < 0.10",
            "Percentage with |r| < 0.20",
            "Percentage with |r| < 0.30",
        ],
        "value": [
            top_results_df[
                "morphology_feature"
            ].nunique(),
            len(top_results_df),
            absolute_r.median(),
            absolute_r.mean(),
            absolute_r.quantile(0.25),
            absolute_r.quantile(0.75),
            absolute_r.max(),
            100
            * (
                absolute_r < 0.10
            ).mean(),
            100
            * (
                absolute_r < 0.20
            ).mean(),
            100
            * (
                absolute_r < 0.30
            ).mean(),
        ],
    })

    summary_df.to_csv(
        output_dir
        / "mlp_top20_correlation_summary.csv",
        index=False,
    )

    return summary_df


# ============================================================================
# FIGURE HELPERS
# ============================================================================

def format_feature_title(
    feature_name: str,
) -> str:

    parts = feature_name.split("_")

    if len(parts) <= 2:
        return feature_name

    return (
        "_".join(parts[:2])
        + "\n"
        + "_".join(parts[2:])
    )


def add_linear_fit_with_ci(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Add an ordinary least-squares fit with an approximate 95% CI."""

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    x_grid = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    y_fit = (
        intercept
        + slope * x_grid
    )

    fitted_observed = (
        intercept
        + slope * x
    )

    residuals = (
        y
        - fitted_observed
    )

    dof = max(
        len(x) - 2,
        1,
    )

    residual_se = np.sqrt(
        np.sum(
            residuals ** 2
        )
        / dof
    )

    x_mean = np.mean(x)

    sxx = np.sum(
        (x - x_mean) ** 2
    )

    if sxx <= 0:
        return

    fit_se = residual_se * np.sqrt(
        (1.0 / len(x))
        +
        (
            (x_grid - x_mean) ** 2
            / sxx
        )
    )

    critical = stats.t.ppf(
        0.975,
        dof,
    )

    ax.fill_between(
        x_grid,
        y_fit - critical * fit_se,
        y_fit + critical * fit_se,
        color=NAVY,
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        x_grid,
        y_fit,
        color=NAVY,
        linewidth=2.0,
        zorder=3,
    )


def gene_display_label(
    gene: str,
    annotation_map: dict[str, str],
) -> str:
    """Return a gene symbol when available, otherwise the predictor ID."""

    return annotation_map.get(
        str(gene),
        str(gene),
    )


# ============================================================================
# REPRESENTATIVE FOUR-PANEL FIGURE
# ============================================================================

def create_representative_figure(
    X_all: pd.DataFrame,
    Y_all: pd.DataFrame,
    top_results_df: pd.DataFrame,
    annotation_map: dict[str, str],
    output_dir: Path,
) -> None:

    for feature in REPRESENTATIVE_FEATURES:
        if feature not in set(
            top_results_df[
                "morphology_feature"
            ]
        ):
            raise ValueError(
                "Representative feature is missing from MLP results: "
                f"{feature}"
            )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 10.5),
        constrained_layout=True,
    )

    panel_letters = [
        "a",
        "b",
        "c",
        "d",
    ]

    for row_index, feature in enumerate(
        REPRESENTATIVE_FEATURES
    ):
        feature_results = (
            top_results_df[
                top_results_df[
                    "morphology_feature"
                ] == feature
            ]
            .sort_values(
                "mean_permutation_importance",
                ascending=False,
            )
            .head(20)
            .copy()
        )

        if feature_results.empty:
            continue

        feature_results[
            "gene_label"
        ] = [
            gene_display_label(
                gene,
                annotation_map,
            )
            for gene in feature_results[
                "gene"
            ]
        ]

        # -----------------------------
        # permutation importance
        # -----------------------------
        ax_bar = axes[
            row_index,
            0,
        ]

        bar_df = (
            feature_results
            .sort_values(
                "mean_permutation_importance",
                ascending=True,
            )
        )

        ax_bar.barh(
            bar_df["gene_label"],
            bar_df[
                "mean_permutation_importance"
            ],
            height=0.72,
            color=NAVY,
            edgecolor=NAVY,
            linewidth=0.5,
        )

        ax_bar.axvline(
            x=0,
            color="black",
            linewidth=1.0,
        )

        ax_bar.set_xlabel(
            "permutation importance",
            fontsize=12,
            fontweight="bold",
        )

        ax_bar.set_ylabel("")

        ax_bar.set_title(
            format_feature_title(
                feature
            ),
            fontsize=13,
            fontweight="bold",
            pad=6,
        )

        ax_bar.tick_params(
            axis="x",
            labelsize=10,
        )

        ax_bar.tick_params(
            axis="y",
            labelsize=9,
            length=0,
        )

        for label in ax_bar.get_yticklabels():
            label.set_fontweight(
                "bold"
            )

        # -----------------------------
        #rank-1 marginal plot
        # -----------------------------
        rank1 = feature_results.iloc[0]

        gene = str(
            rank1["gene"]
        )

        gene_label = gene_display_label(
            gene,
            annotation_map,
        )

        x_values = pd.to_numeric(
            Y_all[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

        y_values = pd.to_numeric(
            X_all[gene],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
        )

        x_values = x_values[
            valid
        ]

        y_values = y_values[
            valid
        ]

        r_value, p_value = pearsonr(
            x_values,
            y_values,
        )

        ax_scatter = axes[
            row_index,
            1,
        ]

        ax_scatter.scatter(
            x_values,
            y_values,
            s=18,
            color=NAVY,
            alpha=0.68,
            edgecolors="none",
            zorder=2,
        )

        add_linear_fit_with_ci(
            ax=ax_scatter,
            x=x_values,
            y=y_values,
        )

        ax_scatter.text(
            0.03,
            0.97,
            (
                f"r = {r_value:.2f}\n"
                f"p = {p_value:.1e}"
            ),
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.8,
            },
        )

        ax_scatter.set_xlabel(
            f"{format_feature_title(feature)} (z)",
            fontsize=12,
            fontweight="bold",
        )

        ax_scatter.set_ylabel(
            f"{gene_label}\nexpression (z)",
            fontsize=12,
            fontweight="bold",
        )

        ax_scatter.tick_params(
            axis="both",
            labelsize=10,
        )

        for ax in (
            ax_bar,
            ax_scatter,
        ):
            for spine in (
                ax.spines.values()
            ):
                spine.set_linewidth(
                    0.9
                )
                spine.set_color(
                    "black"
                )

            ax.grid(False)

    for ax, letter in zip(
        axes.ravel(),
        panel_letters,
    ):
        ax.text(
            -0.12,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
        )

    output_prefix = (
        output_dir
        / "mlp_representative_gene_morphology_pairs"
    )

    fig.savefig(
        output_prefix.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        output_prefix.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        output_prefix.with_suffix(
            ".png"
        ),
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    np.random.seed(
        RANDOM_SEED
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_all, Y_all = (
        load_numeric_matrices(
            gene_path=args.gene_expression,
            morphology_path=args.morphology,
        )
    )

    features_to_test = (
        get_features_to_test(
            Y_all
        )
    )

    annotation_map = (
        load_gene_annotation(
            args.gene_annotation
        )
    )

    print("\nAnalysis settings")
    print("-----------------")
    print(
        "Gene predictors:",
        X_all.shape[1],
    )
    print(
        "Morphology features:",
        len(features_to_test),
    )
    print(
        "CV folds:",
        args.folds,
    )
    print(
        "Permutation repeats:",
        args.permutation_repeats,
    )
    print(
        "Top genes retained:",
        args.top_n,
    )
    print(
        "MLP parameters:",
        MLP_PARAMS,
    )

    (
        top_results_df,
        _,
        metrics_df,
        failed_df,
    ) = run_all_features(
        X_all=X_all,
        Y_all=Y_all,
        features_to_test=features_to_test,
        output_dir=args.output_dir,
        top_n=args.top_n,
        n_splits=args.folds,
        n_repeats=args.permutation_repeats,
    )

    correlation_summary_df = (
        save_correlation_summary(
            top_results_df=top_results_df,
            output_dir=args.output_dir,
        )
    )

    create_representative_figure(
        X_all=X_all,
        Y_all=Y_all,
        top_results_df=top_results_df,
        annotation_map=annotation_map,
        output_dir=args.output_dir,
    )

    print("\nAnalysis complete.")
    print(
        "Features successfully analyzed:",
        top_results_df[
            "morphology_feature"
        ].nunique(),
    )
    print(
        "Fold-metric rows:",
        len(metrics_df),
    )
    print(
        "Failed features:",
        len(failed_df),
    )
    print("\nCorrelation summary:")
    print(
        correlation_summary_df.to_string(
            index=False
        )
    )
    print(
        "\nOutputs saved to:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
