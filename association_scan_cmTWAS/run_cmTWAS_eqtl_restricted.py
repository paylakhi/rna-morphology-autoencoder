#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests



# ============================================================================
# MANUSCRIPT MORPHOLOGY FEATURES
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
# UTILITIES
# ============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_donor_col(
    df: pd.DataFrame,
    donor_col_candidates: List[str] | None = None,
) -> pd.DataFrame:
    if donor_col_candidates is None:
        donor_col_candidates = [
            "donor",
            "IID",
            "iid",
            "sample",
            "sample_id",
            "Donor",
            "ID",
        ]

    df = df.copy()

    for c in donor_col_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "donor"})
            break

    if "donor" not in df.columns:
        raise ValueError(
            "Could not find a donor/sample identifier column. "
            "Please ensure the input contains a donor identifier."
        )

    df["donor"] = df["donor"].astype(str)

    if df["donor"].duplicated().any():
        duplicated = df.loc[df["donor"].duplicated(), "donor"].unique().tolist()
        raise ValueError(
            "Duplicate donor identifiers were found. "
            f"Examples: {duplicated[:10]}"
        )

    return df


def upper_gene_cols(expr: pd.DataFrame) -> pd.DataFrame:
    """Uppercase expression-gene columns while leaving 'donor' unchanged."""
    expr = expr.copy()
    expr.columns = [
        c if c == "donor" else str(c).strip().upper()
        for c in expr.columns
    ]

    gene_cols = [c for c in expr.columns if c != "donor"]
    if len(gene_cols) != len(set(gene_cols)):
        raise ValueError(
            "Uppercasing expression column names created duplicate gene names. "
            "Resolve duplicate gene identifiers before running the analysis."
        )

    return expr


def build_covariate_matrix(
    morph_df: pd.DataFrame,
    covar_numeric: List[str],
    covar_categorical: List[str],
) -> pd.DataFrame:
    parts = []

    if covar_numeric:
        missing = [c for c in covar_numeric if c not in morph_df.columns]
        if missing:
            raise ValueError(
                f"Missing numeric covariates in morphology file: {missing}"
            )

        c_num = morph_df[covar_numeric].copy()

        for c in c_num.columns:
            c_num[c] = pd.to_numeric(c_num[c], errors="coerce")

        parts.append(c_num)

    if covar_categorical:
        missing = [c for c in covar_categorical if c not in morph_df.columns]
        if missing:
            raise ValueError(
                f"Missing categorical covariates in morphology file: {missing}"
            )

        c_cat = pd.get_dummies(
            morph_df[covar_categorical].astype(str),
            drop_first=True,
            dtype=float,
        )

        parts.append(c_cat)

    if parts:
        covariates = pd.concat(parts, axis=1)
    else:
        covariates = pd.DataFrame(index=morph_df.index)

    # Add intercept.
    covariates.insert(0, "intercept", 1.0)

    # Remove completely missing covariates.
    covariates = covariates.loc[:, ~covariates.isna().all()]

    # Mean-impute remaining missing values in numeric covariates.
    for c in covariates.columns:
        if covariates[c].isna().any():
            covariates[c] = covariates[c].fillna(covariates[c].mean())

    return covariates.astype(float)


def residualize_matrix(
    x: np.ndarray,
    covariates: np.ndarray,
) -> np.ndarray:
    """Remove the linear effects of covariates from one or more variables."""
    c_pinv = np.linalg.pinv(covariates)
    return x - covariates @ (c_pinv @ x)


def assoc_scan_y_vs_many_x(
    y: np.ndarray,
    x: np.ndarray,
    df_resid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    sxx = np.sum(x * x, axis=0)
    valid = sxx > np.finfo(float).eps

    beta = np.full(x.shape[1], np.nan, dtype=float)
    se = np.full(x.shape[1], np.nan, dtype=float)
    tval = np.full(x.shape[1], np.nan, dtype=float)
    pval = np.full(x.shape[1], np.nan, dtype=float)

    if not np.any(valid):
        return beta, se, tval, pval

    x_valid = x[:, valid]

    xty = x_valid.T @ y
    beta_valid = xty / sxx[valid]

    yty = float(y @ y)
    rss_valid = yty - (xty ** 2) / sxx[valid]


    rss_valid = np.maximum(rss_valid, 0.0)

    mse_valid = rss_valid / df_resid
    se_valid = np.sqrt(mse_valid / sxx[valid])

    with np.errstate(divide="ignore", invalid="ignore"):
        t_valid = beta_valid / se_valid

    p_valid = 2.0 * stats.t.sf(
        np.abs(t_valid),
        df=df_resid,
    )

    beta[valid] = beta_valid
    se[valid] = se_valid
    tval[valid] = t_valid
    pval[valid] = p_valid

    return beta, se, tval, pval


def load_eqtl_genes(
    eqtl_path: str,
    p_cutoff: float,
) -> List[str]:
    eqtl = pd.read_csv(eqtl_path, sep="\t")

    colmap = {}

    for c in eqtl.columns:
        cl = str(c).strip().lower()

        if cl in {"gene", "gene_id"}:
            colmap[c] = "gene"

        if cl in {"p-value", "pvalue", "p", "pval"}:
            colmap[c] = "p"

    eqtl = eqtl.rename(columns=colmap)

    if "gene" not in eqtl.columns or "p" not in eqtl.columns:
        raise ValueError(
            "Could not identify gene and p-value columns in the eQTL file. "
            f"Columns found: {eqtl.columns.tolist()}"
        )

    eqtl["gene_upper"] = (
        eqtl["gene"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    eqtl["p"] = pd.to_numeric(
        eqtl["p"],
        errors="coerce",
    )

    eqtl_genes = (
        eqtl.loc[
            eqtl["p"] <= p_cutoff,
            "gene_upper",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(eqtl_genes)



# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an eQTL-restricted expression-morphology association scan "
            "for the 17 predefined manuscript morphology features."
        )
    )

    parser.add_argument(
        "--morph_csv",
        required=True,
        help="Morphology CSV: rows=samples/donors, columns=traits + donor.",
    )

    parser.add_argument(
        "--expr_csv",
        required=True,
        help="Expression CSV: rows=samples/donors, columns=genes + donor.",
    )

    parser.add_argument(
        "--eqtl_tsv",
        required=True,
        help="eQTL results table in tab-separated format.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--eqtl_p_cutoff",
        type=float,
        default=0.05,
        help="eQTL P-value cutoff used to define the tested gene set. Default: 0.05.",
    )

    parser.add_argument(
        "--min_n",
        type=int,
        default=30,
        help="Minimum number of donors required for each morphology trait. Default: 30.",
    )

    parser.add_argument(
        "--covar_num",
        nargs="*",
        default=[],
        help="Numeric covariate column names in the morphology CSV.",
    )

    parser.add_argument(
        "--covar_cat",
        nargs="*",
        default=[],
        help="Categorical covariate column names in the morphology CSV.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    morph = pd.read_csv(args.morph_csv)
    expr = pd.read_csv(args.expr_csv)

    morph = normalize_donor_col(morph)
    expr = normalize_donor_col(expr)
    expr = upper_gene_cols(expr)

    # ------------------------------------------------------------------------
    # Restrict analysis to the 17 high predicted morphology features
    # ------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in OUR_17_FEATURES
        if feature not in morph.columns
    ]

    if missing_features:
        raise ValueError(
            "The following high predicted morphology features are "
            "missing from the morphology CSV:\n"
            + "\n".join(missing_features)
        )

    target_traits = OUR_17_FEATURES.copy()

    # ------------------------------------------------------------------------
    # Select eQTL-prioritized genes that are present in expression data
    # ------------------------------------------------------------------------

    eqtl_genes = load_eqtl_genes(
        args.eqtl_tsv,
        args.eqtl_p_cutoff,
    )

    expr_genes_available = {
        c for c in expr.columns
        if c != "donor"
    }

    genes = [
        gene
        for gene in eqtl_genes
        if gene in expr_genes_available
    ]

    if len(genes) == 0:
        raise ValueError(
            "No eQTL-prioritized genes overlap with the expression matrix. "
            "Check whether both files use the same gene identifier type "
            "(for example, gene symbols versus Ensembl IDs)."
        )

    # ------------------------------------------------------------------------
    # Align donors
    # ------------------------------------------------------------------------

    common_donors = sorted(
        set(morph["donor"]).intersection(
            set(expr["donor"])
        )
    )

    if len(common_donors) == 0:
        raise ValueError(
            "No overlapping donors were found between morphology "
            "and expression files."
        )

    morph = (
        morph
        .set_index("donor")
        .loc[common_donors]
        .copy()
    )

    expr = (
        expr
        .set_index("donor")
        .loc[common_donors, genes]
        .copy()
    )

    # Require numeric expression values.
    expr = expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if expr.isna().any().any():
        missing_count = int(expr.isna().sum().sum())
        raise ValueError(
            f"Expression matrix contains {missing_count} missing/non-numeric "
            "values among the selected eQTL genes. Resolve these values before "
            "running the association analysis."
        )

    # ------------------------------------------------------------------------
    # Build covariate matrix
    # ------------------------------------------------------------------------

    morph_reset = morph.reset_index()

    covariates_df = build_covariate_matrix(
        morph_reset,
        args.covar_num,
        args.covar_cat,
    )

    covariates_df.index = morph.index
    covariates = covariates_df.values

    n_total, n_covar_columns = covariates.shape

    if n_total - n_covar_columns - 1 <= 5:
        raise ValueError(
            "Too few residual degrees of freedom after accounting for "
            f"covariates (n={n_total}, covariate columns including "
            f"intercept={n_covar_columns})."
        )

    x_all = expr.values.astype(float)

    # ------------------------------------------------------------------------
    # Association scan across the 17 high predicted morphology features
    # ------------------------------------------------------------------------

    all_results = []

    for trait in target_traits:

        y_raw = pd.to_numeric(
            morph[trait],
            errors="coerce",
        ).values.astype(float)

        ok = ~np.isnan(y_raw)
        n_trait = int(ok.sum())

        if n_trait < args.min_n:
            raise ValueError(
                f"{trait} has only {n_trait} non-missing donors, "
                f"below --min_n={args.min_n}."
            )

        y = y_raw[ok]
        c_sub = covariates[ok, :]
        x_sub_raw = x_all[ok, :]


        df_trait = n_trait - c_sub.shape[1] - 1

        if df_trait <= 5:
            raise ValueError(
                f"Too few residual degrees of freedom for {trait}: "
                f"df={df_trait}."
            )

        x_sub = residualize_matrix(
            x_sub_raw,
            c_sub,
        )

        y_resid = residualize_matrix(
            y.reshape(-1, 1),
            c_sub,
        ).ravel()

        beta, se, tval, pval = assoc_scan_y_vs_many_x(
            y_resid,
            x_sub,
            df_trait,
        )

        # Benjamini-Hochberg correction across tested genes
        qval = np.full_like(
            pval,
            np.nan,
            dtype=float,
        )

        valid_p = np.isfinite(pval)

        if np.any(valid_p):
            qval[valid_p] = multipletests(
                pval[valid_p],
                method="fdr_bh",
            )[1]

        trait_results = pd.DataFrame(
            {
                "trait": trait,
                "gene": genes,
                "beta": beta,
                "se": se,
                "t": tval,
                "p": pval,
                "q": qval,
                "n": n_trait,
                "df": df_trait,
            }
        )

        trait_results = trait_results.sort_values(
            "p",
            na_position="last",
        )

        all_results.append(trait_results)

        print(
            f"[OK] {trait}: "
            f"n={n_trait}, "
            f"genes tested={int(valid_p.sum())}, "
            f"q<0.05={int(np.nansum(qval < 0.05))}"
        )

    # ------------------------------------------------------------------------
    # Save Output
    # ------------------------------------------------------------------------

    final_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Preserve the selected trait order, then rank genes by P value.
    feature_order = {
        feature: i
        for i, feature in enumerate(target_traits)
    }

    final_results["_feature_order"] = final_results["trait"].map(feature_order)

    final_results = (
        final_results
        .sort_values(
            ["_feature_order", "p"],
            na_position="last",
        )
        .drop(columns="_feature_order")
        .reset_index(drop=True)
    )

    output_path = os.path.join(
        args.out_dir,
        "cmTWAS_17_features.sumstats.tsv",
    )

    final_results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print()
    print("[DONE]")
    print(f"Common donors: {len(common_donors)}")
    print(f"eQTL-prioritized genes available in expression: {len(genes)}")
    print(f"Morphology features analyzed: {len(target_traits)}")
    print(f"Total association rows: {len(final_results)}")
    print(f"Final results: {output_path}")


if __name__ == "__main__":
    main()
