#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import Autoencoder, initialize_weights


SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
BATCH_SIZE = 16
LATENT_DIM = 150
MAX_EPOCHS = 200
PATIENCE = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 2.0

GENE_MEAN_LOGCPM_THRESHOLD = 5.0
GENE_VAR_THRESHOLD = 0.01
MORPH_CORR_THRESHOLD = 0.90

MORPHOLOGY_EXCLUDE_COLUMNS = {
    "Metadata_Plate",
    "Metadata_Assay_Plate_Barcode",
    "Metadata_NCBIGeneID",
    "Col",
    "InsertLength",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphology-csv", type=Path, required=True)
    parser.add_argument("--rna-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_match(
    morphology_path: Path,
    rna_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    morph_raw = pd.read_csv(morphology_path, low_memory=False)
    rna_raw = pd.read_csv(rna_path, low_memory=False)

    if "id" not in rna_raw.columns:
        raise KeyError("RNA file must contain an 'id' column.")
    if "Metadata_well_position" not in morph_raw.columns:
        raise KeyError(
            "Morphology file must contain 'Metadata_well_position'."
        )

    rna_raw["Metadata_Well"] = (
        rna_raw["id"].astype(str).str.split(":").str[-1].str.lower()
    )
    morph_raw["Metadata_well_position"] = (
        morph_raw["Metadata_well_position"].astype(str).str.lower()
    )

    common_wells = sorted(
        set(rna_raw["Metadata_Well"])
        & set(morph_raw["Metadata_well_position"])
    )
    if not common_wells:
        raise ValueError("No overlapping wells were found.")

    rna_matched = (
        rna_raw[rna_raw["Metadata_Well"].isin(common_wells)]
        .drop_duplicates("Metadata_Well")
        .sort_values("Metadata_Well")
        .reset_index(drop=True)
    )
    morph_matched = (
        morph_raw[
            morph_raw["Metadata_well_position"].isin(common_wells)
        ]
        .drop_duplicates("Metadata_well_position")
        .sort_values("Metadata_well_position")
        .reset_index(drop=True)
    )

    if not np.array_equal(
        rna_matched["Metadata_Well"].to_numpy(),
        morph_matched["Metadata_well_position"].to_numpy(),
    ):
        raise ValueError("RNA and morphology rows are not aligned.")

    for column in rna_matched.columns:
        if column.endswith("_at"):
            rna_matched[column] = pd.to_numeric(
                rna_matched[column],
                errors="coerce",
            )

    rna_numeric = rna_matched.select_dtypes(include="number").copy()
    morph_numeric = morph_matched.select_dtypes(include="number").copy()

    remove_columns = [
        column
        for column in morph_numeric.columns
        if (
            column.startswith("Metadata_")
            or column in MORPHOLOGY_EXCLUDE_COLUMNS
        )
    ]
    morph_numeric = morph_numeric.drop(
        columns=remove_columns,
        errors="ignore",
    )

    rna_numeric = rna_numeric.loc[
        :, rna_numeric.isna().mean() < 0.20
    ]
    morph_numeric = morph_numeric.loc[
        :, morph_numeric.isna().mean() < 0.20
    ]

    combined = pd.concat(
        [
            rna_numeric.add_prefix("RNA__"),
            morph_numeric.add_prefix("MORPH__"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    complete = combined.notna().all(axis=1)

    metadata = (
        morph_matched.loc[complete]
        .copy()
        .reset_index(drop=True)
    )
    metadata["sample_id"] = metadata["Metadata_well_position"]

    return (
        rna_numeric.loc[complete].reset_index(drop=True),
        morph_numeric.loc[complete].reset_index(drop=True),
        metadata,
    )


def split_indices(n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)

    train_idx, temporary_idx = train_test_split(
        indices,
        test_size=TEST_FRAC + VAL_FRAC,
        random_state=SEED,
        shuffle=True,
    )
    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=TEST_FRAC / (TEST_FRAC + VAL_FRAC),
        random_state=SEED,
        shuffle=True,
    )
    return train_idx, validation_idx, test_idx


def preprocess_rna(
    dataframe: pd.DataFrame,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
):
    row_sums = dataframe.sum(axis=1).replace(0, np.nan)
    log_cpm = np.log1p(
        dataframe.div(row_sums, axis=0) * 1e6
    ).replace([np.inf, -np.inf], np.nan)

    train_means = log_cpm.iloc[train_idx].mean()
    log_cpm = log_cpm.fillna(train_means).fillna(0)

    expressed = (
        log_cpm.iloc[train_idx].mean()
        > GENE_MEAN_LOGCPM_THRESHOLD
    )
    log_cpm = log_cpm.loc[:, expressed]

    pre_variance_names = list(log_cpm.columns)
    scaler = StandardScaler()

    train = scaler.fit_transform(log_cpm.iloc[train_idx])
    validation = scaler.transform(log_cpm.iloc[validation_idx])
    test = scaler.transform(log_cpm.iloc[test_idx])

    selector = VarianceThreshold(threshold=GENE_VAR_THRESHOLD)
    train = selector.fit_transform(train)
    validation = selector.transform(validation)
    test = selector.transform(test)

    selected_names = [
        name
        for name, selected in zip(
            pre_variance_names,
            selector.get_support(),
        )
        if selected
    ]

    preprocessing = {
        "scaler": scaler,
        "selector": selector,
        "pre_variance_names": pre_variance_names,
        "selected_names": selected_names,
        "gene_mean_logcpm_threshold": GENE_MEAN_LOGCPM_THRESHOLD,
        "gene_variance_threshold": GENE_VAR_THRESHOLD,
    }

    return (
        train.astype(np.float32),
        validation.astype(np.float32),
        test.astype(np.float32),
        selected_names,
        preprocessing,
    )


def preprocess_morphology(
    dataframe: pd.DataFrame,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
):
    train_df = dataframe.iloc[train_idx].copy()
    validation_df = dataframe.iloc[validation_idx].copy()
    test_df = dataframe.iloc[test_idx].copy()

    correlation = train_df.corr().abs()
    upper = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    dropped = [
        column
        for column in upper.columns
        if (upper[column] > MORPH_CORR_THRESHOLD).any()
    ]

    train_df = train_df.drop(columns=dropped)
    validation_df = validation_df.drop(columns=dropped)
    test_df = test_df.drop(columns=dropped)

    feature_names = list(train_df.columns)
    scaler = StandardScaler()

    train = scaler.fit_transform(train_df).astype(np.float32)
    validation = scaler.transform(validation_df).astype(np.float32)
    test = scaler.transform(test_df).astype(np.float32)

    preprocessing = {
        "scaler": scaler,
        "dropped_correlated": dropped,
        "feature_names": feature_names,
        "morphology_correlation_threshold": MORPH_CORR_THRESHOLD,
        "explicitly_excluded_columns": sorted(
            MORPHOLOGY_EXCLUDE_COLUMNS
        ),
    }

    return train, validation, test, feature_names, preprocessing


def make_loader(
    morphology: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(morphology, dtype=torch.float32)
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(
    model: Autoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for (morphology,) in loader:
            morphology = morphology.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstruction, _ = model(morphology)
            loss = criterion(reconstruction, morphology)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )
                optimizer.step()

            total += float(loss.item())
            batches += 1

    return total / max(batches, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    rna, morphology, metadata = load_and_match(
        args.morphology_csv,
        args.rna_csv,
    )
    train_idx, validation_idx, test_idx = split_indices(len(metadata))

    (
        rna_train,
        rna_validation,
        rna_test,
        gene_names,
        rna_preprocessing,
    ) = preprocess_rna(
        rna,
        train_idx,
        validation_idx,
        test_idx,
    )

    (
        morphology_train,
        morphology_validation,
        morphology_test,
        morphology_names,
        morphology_preprocessing,
    ) = preprocess_morphology(
        morphology,
        train_idx,
        validation_idx,
        test_idx,
    )

    print(f"RNA features: {rna_train.shape[1]}")
    print(f"Morphology features: {morphology_train.shape[1]}")
    print(
        "Train/validation/test samples: "
        f"{len(train_idx)}/{len(validation_idx)}/{len(test_idx)}"
    )

    split_table = pd.DataFrame(
        {
            "sample_id": metadata["sample_id"],
            "split": "unassigned",
        }
    )
    split_table.loc[train_idx, "split"] = "train"
    split_table.loc[validation_idx, "split"] = "validation"
    split_table.loc[test_idx, "split"] = "test"
    split_table.to_csv(
        args.output_dir / "sample_split_assignments.csv",
        index=False,
    )

    np.savez_compressed(
        args.output_dir / "prepared_data.npz",
        rna_train=rna_train,
        rna_validation=rna_validation,
        rna_test=rna_test,
        morphology_train=morphology_train,
        morphology_validation=morphology_validation,
        morphology_test=morphology_test,
        test_sample_ids=metadata.iloc[test_idx][
            "sample_id"
        ].astype(str).to_numpy(),
    )

    joblib.dump(
        {
            "rna": rna_preprocessing,
            "morphology": morphology_preprocessing,
            "gene_names": gene_names,
            "morphology_names": morphology_names,
            "train_idx": train_idx,
            "validation_idx": validation_idx,
            "test_idx": test_idx,
            "metadata_sample_ids": metadata["sample_id"].tolist(),
        },
        args.output_dir / "preprocessing.joblib",
    )

    model = Autoencoder(
        input_dim=morphology_train.shape[1],
        latent_dim=LATENT_DIM,
    ).to(device)
    model.apply(initialize_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    train_loader = make_loader(morphology_train, shuffle=True)
    validation_loader = make_loader(
        morphology_validation,
        shuffle=False,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_morphology_reconstruction": train_loss,
                "validation_morphology_reconstruction": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train MSE={train_loss:.6f} | "
            f"validation MSE={validation_loss:.6f}"
        )

        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No valid morphology checkpoint was saved.")

    model.load_state_dict(best_state)

    checkpoint = {
        "state_dict": best_state,
        "latent_dim": LATENT_DIM,
        "autoencoder_type": "single_layer_linear",
        "best_validation_loss": best_loss,
    }
    torch.save(
        checkpoint,
        args.output_dir / "best_morphology_autoencoder.pt",
    )

    pd.DataFrame(history).to_csv(
        args.output_dir / "morphology_training_history.csv",
        index=False,
    )

    with open(
        args.output_dir / "stage1_configuration.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "seed": SEED,
                "latent_dim": LATENT_DIM,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            handle,
            indent=2,
        )

    print(
        "Saved morphology checkpoint from epoch "
        f"{best_epoch}: "
        f"{args.output_dir / 'best_morphology_autoencoder.pt'}"
    )


if __name__ == "__main__":
    main()
