#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import Autoencoder


# ============================================================
# DEFAULT ANALYSIS SETTINGS
# ============================================================

SEED = 42
N_BOOTSTRAPS = 5000
N_PERMUTATIONS = 5000
BOOTSTRAP_CI = 95.0
R2_THRESHOLD = 0.60
FDR_THRESHOLD = 0.05


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--donor-rna",
        type=Path,
        required=True,
        help="CSV containing donor RNA profiles.",
    )
    parser.add_argument(
        "--donor-morphology",
        type=Path,
        required=True,
        help="CSV containing matched donor morphology profiles.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help=(
            "LUAD training output directory containing preprocessing.joblib "
            "and two_stage_crossmodal_autoencoder.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/donor_transfer"),
        help="Directory for donor-transfer results.",
    )
    parser.add_argument(
        "--sample-id-column",
        type=str,
        default="sample_id",
        help="Identifier column shared by donor RNA and morphology files.",
    )
    parser.add_argument(
        "--rna-input-space",
        choices=["counts", "logcpm"],
        default="counts",
        help=(
            "Use 'counts' for raw counts requiring CPM + log1p. Use 'logcpm' "
            "if CPM + log1p has already been applied. LUAD-fitted scaling and "
            "variance selection are always reused without refitting."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used for inference.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=N_BOOTSTRAPS,
        help="Number of bootstrap resamples of individual donors.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help="Number of permutations for global feature-level significance.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=BOOTSTRAP_CI,
        help="Percentile confidence interval for bootstrap R².",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for bootstrap and permutation analyses.",
    )
    parser.add_argument(
        "--r2-threshold",
        type=float,
        default=R2_THRESHOLD,
        help="R² threshold used to highlight Figure 3 features.",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=FDR_THRESHOLD,
        help="FDR threshold used to highlight Figure 3 features.",
    )

    return parser.parse_args()


# ============================================================
# MULTIPLE-TESTING CORRECTION
# ============================================================


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction while preserving NaN values."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)

    valid = np.isfinite(p_values)
    if not valid.any():
        return q_values

    valid_p = p_values[valid]
    n_tests = valid_p.size
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    ranks = np.arange(1, n_tests + 1, dtype=float)

    sorted_q = sorted_p * n_tests / ranks
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)

    restored = np.empty_like(sorted_q)
    restored[order] = sorted_q
    q_values[valid] = restored

    return q_values


# ============================================================
# LOAD AND MATCH DONOR DATA
# ============================================================


def load_and_align_donors(
    rna_path: Path,
    morphology_path: Path,
    sample_id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Match RNA and morphology strictly by sample/donor identifier."""
    rna = pd.read_csv(rna_path, low_memory=False)
    morphology = pd.read_csv(morphology_path, low_memory=False)

    for label, dataframe in (
        ("donor RNA", rna),
        ("donor morphology", morphology),
    ):
        if sample_id_column not in dataframe.columns:
            raise KeyError(
                f"{label} is missing the required identifier column "
                f"'{sample_id_column}'."
            )

        dataframe[sample_id_column] = (
            dataframe[sample_id_column].astype(str).str.strip()
        )

        if dataframe[sample_id_column].duplicated().any():
            duplicated = dataframe.loc[
                dataframe[sample_id_column].duplicated(keep=False),
                sample_id_column,
            ].astype(str).unique()
            raise ValueError(
                f"{label} contains duplicated sample IDs. Examples: "
                f"{duplicated[:10].tolist()}"
            )

    common_ids = sorted(
        set(rna[sample_id_column]) & set(morphology[sample_id_column])
    )

    if not common_ids:
        raise ValueError(
            "No matched donor/sample IDs were found between RNA and morphology."
        )

    rna = (
        rna[rna[sample_id_column].isin(common_ids)]
        .set_index(sample_id_column)
        .loc[common_ids]
        .copy()
    )
    morphology = (
        morphology[morphology[sample_id_column].isin(common_ids)]
        .set_index(sample_id_column)
        .loc[common_ids]
        .copy()
    )

    if not rna.index.equals(morphology.index):
        raise RuntimeError("RNA and morphology donor ordering is not identical.")

    return rna, morphology, common_ids


# ============================================================
# DONOR RNA PREPROCESSING
# ============================================================


def preprocess_donor_rna(
    donor_rna: pd.DataFrame,
    rna_preprocessing: dict,
    expected_gene_names: list[str],
    input_space: str,
) -> np.ndarray:

    required_keys = {
        "pre_variance_names",
        "scaler",
        "selector",
        "selected_names",
    }
    missing_keys = required_keys - set(rna_preprocessing)
    if missing_keys:
        raise KeyError(
            "Saved RNA preprocessing is missing keys: "
            f"{sorted(missing_keys)}"
        )

    pre_variance_names = list(rna_preprocessing["pre_variance_names"])
    selected_names = list(rna_preprocessing["selected_names"])

    if selected_names != expected_gene_names:
        raise ValueError(
            "Saved selected RNA feature names do not match the gene order "
            "stored for the trained LUAD model."
        )

    numeric = (
        donor_rna.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    missing_genes = [
        gene for gene in pre_variance_names if gene not in numeric.columns
    ]
    if missing_genes:
        raise KeyError(
            "Donor RNA is missing genes required by the LUAD preprocessing. "
            f"Examples: {missing_genes[:10]}"
        )

    numeric = numeric.loc[:, pre_variance_names]

    if input_space == "counts":
        if (numeric < 0).any().any():
            raise ValueError(
                "Negative RNA values were detected. If RNA is already logCPM, "
                "use --rna-input-space logcpm."
            )

        library_size = numeric.sum(axis=1)
        if (library_size <= 0).any():
            raise ValueError(
                "At least one donor has a non-positive RNA library size."
            )

        numeric = np.log1p(
            numeric.div(library_size, axis=0) * 1e6
        )

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "Non-finite donor RNA values remain after preprocessing."
        )

    scaled = rna_preprocessing["scaler"].transform(numeric)
    transformed = (
        rna_preprocessing["selector"]
        .transform(scaled)
        .astype(np.float32)
    )

    if transformed.shape[1] != len(expected_gene_names):
        raise ValueError(
            "Processed donor RNA dimension does not match the trained RNA encoder."
        )

    return transformed


# ============================================================
# DONOR MORPHOLOGY PREPARATION
# ============================================================


def prepare_shared_donor_morphology(
    donor_morphology: pd.DataFrame,
    trained_morphology_names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:

    shared_features = [
        feature
        for feature in trained_morphology_names
        if feature in donor_morphology.columns
    ]

    if not shared_features:
        raise ValueError(
            "No morphology features are shared between the LUAD-trained model "
            "and the donor morphology dataset."
        )

    trained_feature_indices = np.asarray(
        [
            trained_morphology_names.index(feature)
            for feature in shared_features
        ],
        dtype=int,
    )

    observed = (
        donor_morphology.loc[:, shared_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    if not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError(
            "Non-finite values remain in shared donor morphology features."
        )

    observed_array = observed.to_numpy(dtype=float)

    return (
        observed_array.astype(np.float32),
        shared_features,
        trained_feature_indices,
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================


def load_trained_models(
    checkpoint_path: Path,
    n_genes: int,
    n_morphology_features: int,
    device: torch.device,
) -> tuple[Autoencoder, Autoencoder]:
    """Load the final RNA model and frozen morphology model for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {
        "rna_state_dict",
        "morphology_state_dict",
        "latent_dim",
    }
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise KeyError(
            "LUAD cross-modal checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    latent_dim = int(checkpoint["latent_dim"])

    rna_model = Autoencoder(
        input_dim=n_genes,
        latent_dim=latent_dim,
    ).to(device)
    morphology_model = Autoencoder(
        input_dim=n_morphology_features,
        latent_dim=latent_dim,
    ).to(device)

    rna_model.load_state_dict(checkpoint["rna_state_dict"])
    morphology_model.load_state_dict(checkpoint["morphology_state_dict"])

    rna_model.eval()
    morphology_model.eval()

    for model in (rna_model, morphology_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    return rna_model, morphology_model


# ============================================================
# FIXED-MODEL INFERENCE
# ============================================================


def predict_morphology(
    donor_rna_array: np.ndarray,
    rna_model: Autoencoder,
    morphology_model: Autoencoder,
    device: torch.device,
) -> np.ndarray:
    """RNA -> trained RNA encoder -> frozen morphology decoder."""
    donor_tensor = torch.tensor(
        donor_rna_array,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        latent = rna_model.encode(donor_tensor)
        prediction = morphology_model.decode(latent)

    return prediction.cpu().numpy().astype(np.float32)


# ============================================================
# R² HELPERS
# ============================================================


def vectorized_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted matrices must have equal shape.")

    residual_ss = np.sum((observed - predicted) ** 2, axis=0)
    observed_mean = np.mean(observed, axis=0)
    total_ss = np.sum((observed - observed_mean) ** 2, axis=0)

    r2 = np.full(observed.shape[1], np.nan, dtype=float)
    valid = total_ss > 0
    r2[valid] = 1.0 - residual_ss[valid] / total_ss[valid]

    return r2


# ============================================================
# GLOBAL + INDIVIDUAL-LEVEL BOOTSTRAP EVALUATION
# ============================================================


def bootstrap_individual_level_r2(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_bootstraps: int,
    ci: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")
    if not 0.0 < ci < 100.0:
        raise ValueError("bootstrap CI must be between 0 and 100.")

    n_donors, n_features = observed.shape
    if n_donors < 3:
        raise ValueError("At least three matched donors are required.")

    rng = np.random.default_rng(seed)
    bootstrap_r2 = np.full(
        (n_bootstraps, n_features),
        np.nan,
        dtype=float,
    )

    for bootstrap_index in range(n_bootstraps):
        sampled_indices = rng.integers(
            0,
            n_donors,
            size=n_donors,
        )
        bootstrap_r2[bootstrap_index] = vectorized_r2(
            observed[sampled_indices],
            predicted[sampled_indices],
        )

    alpha = (100.0 - ci) / 2.0

    return (
        np.nanmean(bootstrap_r2, axis=0),
        np.nanpercentile(bootstrap_r2, alpha, axis=0),
        np.nanpercentile(bootstrap_r2, 100.0 - alpha, axis=0),
    )


# ============================================================
# POPULATION-LEVEL PERMUTATION TEST
# ============================================================


def permutation_p_values(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_r2: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """
    One-sided empirical p-values for global feature-wise R².

    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    rng = np.random.default_rng(seed)
    n_donors = observed.shape[0]
    exceedances = np.zeros(observed.shape[1], dtype=np.int64)
    valid = np.isfinite(observed_r2)

    for _ in range(n_permutations):
        permuted_indices = rng.permutation(n_donors)
        permuted_r2 = vectorized_r2(
            observed,
            predicted[permuted_indices],
        )
        exceedances += (
            valid
            & np.isfinite(permuted_r2)
            & (permuted_r2 >= observed_r2)
        )

    p_values = np.full(observed.shape[1], np.nan, dtype=float)
    p_values[valid] = (
        exceedances[valid] + 1.0
    ) / (
        n_permutations + 1.0
    )

    return p_values


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_path = args.training_dir / "preprocessing.joblib"
    checkpoint_path = (
        args.training_dir / "two_stage_crossmodal_autoencoder.pt"
    )

    for required_path in (preprocessing_path, checkpoint_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required LUAD training output not found: {required_path}"
            )

    preprocessing = joblib.load(preprocessing_path)

    required_preprocessing_keys = {
        "gene_names",
        "morphology_names",
        "rna",
        "morphology",
    }
    missing = required_preprocessing_keys - set(preprocessing)
    if missing:
        raise KeyError(
            "preprocessing.joblib is missing required keys: "
            f"{sorted(missing)}"
        )

    trained_gene_names = list(preprocessing["gene_names"])
    trained_morphology_names = list(preprocessing["morphology_names"])

    # --------------------------------------------------------
    # 1. Match donor RNA and morphology by identifier.
    # --------------------------------------------------------
    donor_rna, donor_morphology, sample_ids = load_and_align_donors(
        rna_path=args.donor_rna,
        morphology_path=args.donor_morphology,
        sample_id_column=args.sample_id_column,
    )

    # --------------------------------------------------------
    # 2. Reuse the LUAD-fitted RNA preprocessing.
    # --------------------------------------------------------
    donor_rna_array = preprocess_donor_rna(
        donor_rna=donor_rna,
        rna_preprocessing=preprocessing["rna"],
        expected_gene_names=trained_gene_names,
        input_space=args.rna_input_space,
    )

    # --------------------------------------------------------
    # 3. Prepare observed donor morphology for evaluation only.
    # --------------------------------------------------------
    (
        observed_morphology,
        shared_morphology_names,
        shared_feature_indices,
    ) = prepare_shared_donor_morphology(
        donor_morphology=donor_morphology,
        trained_morphology_names=trained_morphology_names,
    )

    # --------------------------------------------------------
    # 4. Select device and load the fixed trained model.
    # --------------------------------------------------------
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    rna_model, morphology_model = load_trained_models(
        checkpoint_path=checkpoint_path,
        n_genes=len(trained_gene_names),
        n_morphology_features=len(trained_morphology_names),
        device=device,
    )

    # --------------------------------------------------------
    # 5. ONE fixed-model inference pass.
    # --------------------------------------------------------
    full_prediction = predict_morphology(
        donor_rna_array=donor_rna_array,
        rna_model=rna_model,
        morphology_model=morphology_model,
        device=device,
    )

    predicted_morphology = full_prediction[:, shared_feature_indices]

    if predicted_morphology.shape != observed_morphology.shape:
        raise RuntimeError(
            "Observed and predicted donor morphology matrices do not match."
        )

    predicted_dataframe = pd.DataFrame(
        predicted_morphology,
        columns=shared_morphology_names,
    )
    predicted_dataframe.insert(
        0,
        args.sample_id_column,
        sample_ids,
    )
    predicted_dataframe.to_csv(
        args.output_dir / "global_predicted.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Global / population-level feature-wise R².
    # --------------------------------------------------------
    global_r2 = vectorized_r2(
        observed_morphology,
        predicted_morphology,
    )

    # --------------------------------------------------------
    # 7. Individual-level robustness by donor bootstrap.
    # --------------------------------------------------------
    (
        indiv_mean_r2,
        indiv_ci_lower,
        indiv_ci_upper,
    ) = bootstrap_individual_level_r2(
        observed=observed_morphology,
        predicted=predicted_morphology,
        n_bootstraps=args.n_bootstraps,
        ci=args.bootstrap_ci,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # 8. Global permutation significance + BH-FDR.
    # --------------------------------------------------------
    global_p = permutation_p_values(
        observed=observed_morphology,
        predicted=predicted_morphology,
        observed_r2=global_r2,
        n_permutations=args.n_permutations,
        seed=args.seed + 1,
    )
    global_q = benjamini_hochberg(global_p)

    # --------------------------------------------------------
    # 9. Compact Figure 3 summary.
    # --------------------------------------------------------
    results = pd.DataFrame(
        {
            "feature": shared_morphology_names,
            "indiv_mean_r2": indiv_mean_r2,
            "global_r2": global_r2,
            "global_p": global_p,
            "global_q": global_q,
        }
    )

    results["highlight"] = (
        (results["indiv_mean_r2"] > args.r2_threshold)
        & (results["global_r2"] > args.r2_threshold)
        & (results["global_q"] < args.fdr_threshold)
    )

    results = results.sort_values(
        by=["highlight", "global_r2"],
        ascending=[False, False],
    ).reset_index(drop=True)

    results.to_csv(
        args.output_dir / "global_individual_r2_with_pq.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 10. Console summary only; no additional output files.
    # --------------------------------------------------------
    selected = results[results["highlight"]]

    print("\nDonor-transfer analysis complete.")
    print(f"Matched donors/samples: {len(sample_ids)}")
    print(
        "Shared morphology features evaluated: "
        f"{len(shared_morphology_names)}"
    )
    print(f"Bootstrap donor resamples: {args.n_bootstraps}")
    print(f"Population-level permutations: {args.n_permutations}")
    print("Model retraining or donor-specific fitting: NO")
    print("Predictions regenerated during bootstrap: NO")
    print(
        "NOTE: indiv_mean_r2 is the MEAN feature-wise R² across "
        "bootstrap resamples of individual donors."
    )
    print(
        "Features satisfying global R² > "
        f"{args.r2_threshold}, indiv_mean_r2 > {args.r2_threshold}, "
        f"and q < {args.fdr_threshold}: {len(selected)}"
    )
    print("\nSaved outputs:")
    print(args.output_dir / "global_predicted.csv")
    print(args.output_dir / "global_individual_r2_with_pq.csv")


if __name__ == "__main__":
    main()
