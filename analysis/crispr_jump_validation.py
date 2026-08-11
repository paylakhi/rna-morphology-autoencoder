#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================================
# DEFAULT COLUMN NAMES
# ============================================================================

DEFAULT_ID_COLUMN = "Metadata_JCP2022"
DEFAULT_GENE_COLUMN = "Metadata_Symbol"
DEFAULT_PLATE_COLUMN = "Metadata_Plate"
DEFAULT_WELL_COLUMN = "Metadata_Well"

DEFAULT_CONTROL_LABELS = (
    "non-targeting",
    "no-guide",
)


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--profiles",
        type=str,
        default="data/jump_crispr_profiles.parquet",
        help=(
            "Well-level JUMP CRISPR morphology profiles in Parquet format. "
            "A local path or parquet-compatible URL may be used."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/jump_crispr_metadata.csv"),
        help=(
            "CRISPR perturbation metadata containing the JUMP perturbation ID "
            "and gene symbol."
        ),
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/crispr_validation_pairs.csv"),
        help=(
            "CSV containing gene-feature pairs to test. Required columns: "
            "'gene' and 'feature'."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/crispr_validation"),
        help="Directory for result tables and individual validation plots.",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default=DEFAULT_ID_COLUMN,
        help="Perturbation-ID column shared by profiles and CRISPR metadata.",
    )

    parser.add_argument(
        "--gene-column",
        type=str,
        default=DEFAULT_GENE_COLUMN,
        help="Gene-symbol column in the CRISPR metadata.",
    )

    parser.add_argument(
        "--plate-column",
        type=str,
        default=DEFAULT_PLATE_COLUMN,
        help="Plate identifier column in the well-level profile table.",
    )

    parser.add_argument(
        "--well-column",
        type=str,
        default=DEFAULT_WELL_COLUMN,
        help=(
            "Well identifier column in the profile table. Used only for "
            "duplicate-well checking when present."
        ),
    )

    parser.add_argument(
        "--control-labels",
        nargs="+",
        default=list(DEFAULT_CONTROL_LABELS),
        help=(
            "Gene labels that identify negative controls in the CRISPR "
            "metadata."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used only for deterministic jitter in plots.",
    )

    return parser.parse_args()


# ============================================================================
# HELPERS
# ============================================================================

def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def safe_filename(value: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )
    return cleaned.strip("_")


def require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{table_name} is missing required columns: {missing}"
        )


def load_inputs(
    profiles_path: str,
    metadata_path: Path,
    pairs_path: Path,
    id_column: str,
    gene_column: str,
    plate_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    profiles = pd.read_parquet(
        profiles_path
    )

    metadata = pd.read_csv(
        metadata_path
    )

    pairs = pd.read_csv(
        pairs_path
    )

    require_columns(
        profiles,
        [
            id_column,
            plate_column,
        ],
        "CRISPR profiles",
    )

    require_columns(
        metadata,
        [
            id_column,
            gene_column,
        ],
        "CRISPR metadata",
    )

    require_columns(
        pairs,
        [
            "gene",
            "feature",
        ],
        "validation-pair table",
    )

    return profiles, metadata, pairs


def attach_gene_symbols(
    profiles: pd.DataFrame,
    metadata: pd.DataFrame,
    id_column: str,
    gene_column: str,
) -> pd.DataFrame:


    metadata_map = (
        metadata[
            [
                id_column,
                gene_column,
            ]
        ]
        .dropna(
            subset=[
                id_column,
                gene_column,
            ]
        )
        .copy()
    )

    metadata_map[id_column] = (
        metadata_map[id_column]
        .astype(str)
        .str.strip()
    )

    metadata_map[gene_column] = (
        metadata_map[gene_column]
        .map(normalize_gene)
    )

    mapping_counts = (
        metadata_map
        .groupby(id_column)[gene_column]
        .nunique()
    )

    ambiguous_ids = (
        mapping_counts[
            mapping_counts > 1
        ]
        .index
        .tolist()
    )

    if ambiguous_ids:
        raise ValueError(
            "Some perturbation IDs map to multiple gene/control labels. "
            f"Examples: {ambiguous_ids[:10]}"
        )

    metadata_map = (
        metadata_map
        .drop_duplicates(
            subset=[id_column],
            keep="first",
        )
        .rename(
            columns={
                gene_column: "_gene_label"
            }
        )
    )

    merged = profiles.copy()

    merged[id_column] = (
        merged[id_column]
        .astype(str)
        .str.strip()
    )

    # Avoid suffix confusion if the profile file already contains a gene column.
    merged = merged.drop(
        columns=[gene_column],
        errors="ignore",
    )

    merged = merged.merge(
        metadata_map,
        on=id_column,
        how="left",
        validate="many_to_one",
    )

    n_unmapped = int(
        merged["_gene_label"]
        .isna()
        .sum()
    )

    if n_unmapped:
        print(
            f"[WARNING] {n_unmapped} profile rows could not be mapped "
            "to a CRISPR gene/control label and will not be used."
        )

    return merged


def check_duplicate_wells(
    dataframe: pd.DataFrame,
    plate_column: str,
    well_column: str,
) -> None:

    if well_column not in dataframe.columns:
        return

    duplicate_mask = dataframe.duplicated(
        subset=[
            plate_column,
            well_column,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        n_duplicate_rows = int(
            duplicate_mask.sum()
        )

        print(
            "[WARNING] The profile table contains "
            f"{n_duplicate_rows} rows involved in duplicated plate/well IDs. "
            "Confirm that the input is truly a well-level aggregate profile."
        )


# ============================================================================
# ONE GENE-FEATURE COMPARISON
# ============================================================================

def test_gene_feature_pair(
    dataframe: pd.DataFrame,
    gene: str,
    feature: str,
    plate_column: str,
    control_labels: set[str],
) -> tuple[dict, np.ndarray, np.ndarray]:

    if feature not in dataframe.columns:
        raise KeyError(
            f"Morphology feature was not found in the profile table: {feature}"
        )

    gene_normalized = normalize_gene(
        gene
    )

    target = dataframe[
        dataframe["_gene_label"]
        == gene_normalized
    ].copy()

    if target.empty:
        raise ValueError(
            f"No CRISPR profile rows were found for gene '{gene_normalized}'."
        )

    target_plates = (
        target[plate_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not target_plates:
        raise ValueError(
            f"No plate identifiers were available for target gene "
            f"'{gene_normalized}'."
        )

    all_negative_controls = dataframe[
        dataframe["_gene_label"]
        .isin(
            control_labels
        )
    ].copy()

    plate_matched_controls = all_negative_controls[
        all_negative_controls[plate_column]
        .astype(str)
        .str.strip()
        .isin(
            target_plates
        )
    ].copy()

    if plate_matched_controls.empty:
        raise ValueError(
            f"No plate-matched negative controls were found for "
            f"'{gene_normalized}'. Target plates: {target_plates}"
        )

    target_values = (
        pd.to_numeric(
            target[feature],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    control_values = (
        pd.to_numeric(
            plate_matched_controls[feature],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if target_values.size == 0:
        raise ValueError(
            f"No finite target values were available for "
            f"{gene_normalized} / {feature}."
        )

    if control_values.size == 0:
        raise ValueError(
            f"No finite plate-matched control values were available for "
            f"{gene_normalized} / {feature}."
        )

    mw_result = mannwhitneyu(
        target_values,
        control_values,
        alternative="two-sided",
        method="auto",
    )

    result = {
        "gene": gene_normalized,
        "feature": feature,
        "n_target_wells": int(
            target_values.size
        ),
        "n_control_wells": int(
            control_values.size
        ),
        "n_target_plates": int(
            len(target_plates)
        ),
        "target_plates": "|".join(
            sorted(
                target_plates
            )
        ),
        "median_target": float(
            np.median(
                target_values
            )
        ),
        "median_control": float(
            np.median(
                control_values
            )
        ),
        "median_difference_target_minus_control": float(
            np.median(
                target_values
            )
            - np.median(
                control_values
            )
        ),
        "mean_target": float(
            np.mean(
                target_values
            )
        ),
        "mean_control": float(
            np.mean(
                control_values
            )
        ),
        "mannwhitney_u": float(
            mw_result.statistic
        ),
        "mannwhitney_p": float(
            mw_result.pvalue
        ),
    }

    return (
        result,
        target_values,
        control_values,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_gene_feature_pair(
    gene: str,
    feature: str,
    target_values: np.ndarray,
    control_values: np.ndarray,
    mannwhitney_p: float,
    output_dir: Path,
    seed: int,
) -> None:

    rng = np.random.default_rng(
        seed
    )

    fig, ax = plt.subplots(
        figsize=(4.3, 4.6)
    )

    data = [
        target_values,
        control_values,
    ]

    boxplot = ax.boxplot(
        data,
        positions=[
            1,
            2,
        ],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "linewidth": 1.6,
        },
        boxprops={
            "facecolor": "white",
            "linewidth": 1.2,
        },
        whiskerprops={
            "linewidth": 1.1,
        },
        capprops={
            "linewidth": 1.1,
        },
    )


    x_target = rng.normal(
        loc=1.0,
        scale=0.045,
        size=target_values.size,
    )

    x_control = rng.normal(
        loc=2.0,
        scale=0.045,
        size=control_values.size,
    )

    ax.scatter(
        x_target,
        target_values,
        s=18,
        alpha=0.55,
        edgecolors="none",
        zorder=3,
    )

    ax.scatter(
        x_control,
        control_values,
        s=18,
        alpha=0.55,
        edgecolors="none",
        zorder=3,
    )

    ax.set_xticks(
        [
            1,
            2,
        ]
    )

    ax.set_xticklabels(
        [
            f"{gene} CRISPR",
            "control",
        ],
        fontsize=10,
    )

    ax.set_ylabel(
        feature,
        fontsize=10,
    )

    ax.set_title(
        f"MW p={mannwhitney_p:.2e}",
        fontsize=12,
        pad=8,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    fig.tight_layout()

    base_name = (
        safe_filename(
            f"{gene}__{feature}__crispr_validation"
        )
    )

    for extension in (
        "pdf",
        "svg",
        "png",
    ):
        save_kwargs = {
            "bbox_inches": "tight",
        }

        if extension == "png":
            save_kwargs["dpi"] = 600

        fig.savefig(
            output_dir
            / f"{base_name}.{extension}",
            **save_kwargs,
        )

    plt.close(
        fig
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles, metadata, pairs = load_inputs(
        profiles_path=args.profiles,
        metadata_path=args.metadata,
        pairs_path=args.pairs,
        id_column=args.id_column,
        gene_column=args.gene_column,
        plate_column=args.plate_column,
    )

    profiles = attach_gene_symbols(
        profiles=profiles,
        metadata=metadata,
        id_column=args.id_column,
        gene_column=args.gene_column,
    )

    check_duplicate_wells(
        dataframe=profiles,
        plate_column=args.plate_column,
        well_column=args.well_column,
    )

    pairs = (
        pairs[
            [
                "gene",
                "feature",
            ]
        ]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    if pairs.empty:
        raise ValueError(
            "The validation-pair table contains no usable gene-feature pairs."
        )

    normalized_controls = {
        normalize_gene(
            label
        )
        for label in args.control_labels
    }

    result_rows = []
    failed_rows = []

    for pair_number, pair in enumerate(
        pairs.itertuples(
            index=False
        ),
        start=1,
    ):
        gene = str(
            pair.gene
        ).strip()

        feature = str(
            pair.feature
        ).strip()

        print(
            f"\n[{pair_number}/{len(pairs)}] "
            f"{gene} -> {feature}"
        )

        try:
            (
                result,
                target_values,
                control_values,
            ) = test_gene_feature_pair(
                dataframe=profiles,
                gene=gene,
                feature=feature,
                plate_column=args.plate_column,
                control_labels=normalized_controls,
            )

            result_rows.append(
                result
            )

            plot_gene_feature_pair(
                gene=result["gene"],
                feature=feature,
                target_values=target_values,
                control_values=control_values,
                mannwhitney_p=result["mannwhitney_p"],
                output_dir=args.output_dir,
                seed=args.seed + pair_number,
            )

            print(
                "  target wells:",
                result["n_target_wells"],
            )

            print(
                "  plate-matched control wells:",
                result["n_control_wells"],
            )

            print(
                "  Mann-Whitney p:",
                f'{result["mannwhitney_p"]:.6g}',
            )

        except Exception as error:
            print(
                "  FAILED:",
                error,
            )

            failed_rows.append({
                "gene": gene,
                "feature": feature,
                "error": str(
                    error
                ),
            })

    results_df = pd.DataFrame(
        result_rows
    )

    failed_df = pd.DataFrame(
        failed_rows
    )

    results_df.to_csv(
        args.output_dir
        / "crispr_validation_results.csv",
        index=False,
    )

    failed_df.to_csv(
        args.output_dir
        / "crispr_validation_failed_pairs.csv",
        index=False,
    )

    print(
        "\nAnalysis complete."
    )

    print(
        "Successful comparisons:",
        len(
            results_df
        ),
    )

    print(
        "Failed comparisons:",
        len(
            failed_df
        ),
    )

    print(
        "Outputs saved to:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
