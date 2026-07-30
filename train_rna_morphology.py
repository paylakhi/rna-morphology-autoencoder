#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import (
    Autoencoder,
    LatentDiscriminator,
    freeze_module,
    initialize_weights,
)


SEED = 42
BATCH_SIZE = 16
LATENT_DIM = 150
DISC_HIDDEN_DIM = 128
DISC_DROPOUT = 0.10

MAX_EPOCHS = 200
PATIENCE = 40
LR_RNA = 1e-4
LR_DISC = 1e-4
WD_RNA = 1e-5
WD_DISC = 1e-5

ALPHA_RNA_RECON = 0.1
LAMBDA_ADV = 1.0
DISC_STEPS = 1
GRAD_CLIP_NORM = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(
    rna: np.ndarray,
    morphology: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(rna, dtype=torch.float32),
        torch.tensor(morphology, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def modality_accuracy(
    logits_rna: torch.Tensor,
    logits_morphology: torch.Tensor,
) -> torch.Tensor:
    predicted_rna = (torch.sigmoid(logits_rna) >= 0.5).float()
    predicted_morphology = (
        torch.sigmoid(logits_morphology) >= 0.5
    ).float()

    return 0.5 * (
        (
            predicted_rna == torch.zeros_like(predicted_rna)
        ).float().mean()
        +
        (
            predicted_morphology
            == torch.ones_like(predicted_morphology)
        ).float().mean()
    )


def alignment_epoch(
    rna_model: Autoencoder,
    morphology_reference: Autoencoder,
    discriminator: LatentDiscriminator,
    loader: DataLoader,
    mse: nn.Module,
    bce: nn.Module,
    device: torch.device,
    optimizer_rna: Optional[optim.Optimizer] = None,
    optimizer_discriminator: Optional[optim.Optimizer] = None,
) -> Dict[str, float]:
    training = (
        optimizer_rna is not None
        and optimizer_discriminator is not None
    )

    rna_model.train(training)
    discriminator.train(training)
    morphology_reference.eval()

    totals = {
        "total_rna": 0.0,
        "rna_reconstruction": 0.0,
        "adversarial_alignment": 0.0,
        "discriminator": 0.0,
        "discriminator_accuracy": 0.0,
    }
    batches = 0

    for rna, morphology in loader:
        rna = rna.to(device)
        morphology = morphology.to(device)

        with torch.no_grad():
            morphology_latent = morphology_reference.encode(
                morphology
            )

        if training:
            for _ in range(DISC_STEPS):
                optimizer_discriminator.zero_grad(set_to_none=True)

                with torch.no_grad():
                    detached_rna_latent = rna_model.encode(rna)

                logits_rna = discriminator(detached_rna_latent)
                logits_morphology = discriminator(morphology_latent)

                discriminator_loss = 0.5 * (
                    bce(
                        logits_rna,
                        torch.zeros_like(logits_rna),
                    )
                    + bce(
                        logits_morphology,
                        torch.ones_like(logits_morphology),
                    )
                )

                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    GRAD_CLIP_NORM,
                )
                optimizer_discriminator.step()

            optimizer_rna.zero_grad(set_to_none=True)

            reconstructed_rna, rna_latent = rna_model(rna)
            logits_rna_for_encoder = discriminator(rna_latent)

            rna_reconstruction_loss = mse(
                reconstructed_rna,
                rna,
            )
            adversarial_alignment_loss = bce(
                logits_rna_for_encoder,
                torch.ones_like(logits_rna_for_encoder),
            )

            total_rna_loss = (
                ALPHA_RNA_RECON * rna_reconstruction_loss
                + LAMBDA_ADV * adversarial_alignment_loss
            )

            total_rna_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                rna_model.parameters(),
                GRAD_CLIP_NORM,
            )
            optimizer_rna.step()

        else:
            with torch.no_grad():
                reconstructed_rna, rna_latent = rna_model(rna)
                logits_rna = discriminator(rna_latent)
                logits_morphology = discriminator(
                    morphology_latent
                )

                discriminator_loss = 0.5 * (
                    bce(
                        logits_rna,
                        torch.zeros_like(logits_rna),
                    )
                    + bce(
                        logits_morphology,
                        torch.ones_like(logits_morphology),
                    )
                )

                rna_reconstruction_loss = mse(
                    reconstructed_rna,
                    rna,
                )
                adversarial_alignment_loss = bce(
                    logits_rna,
                    torch.ones_like(logits_rna),
                )
                total_rna_loss = (
                    ALPHA_RNA_RECON * rna_reconstruction_loss
                    + LAMBDA_ADV * adversarial_alignment_loss
                )

        with torch.no_grad():
            metric_logits_rna = discriminator(rna_latent)
            metric_logits_morphology = discriminator(
                morphology_latent
            )
            accuracy = modality_accuracy(
                metric_logits_rna,
                metric_logits_morphology,
            )

        values = {
            "total_rna": float(total_rna_loss.item()),
            "rna_reconstruction": float(
                rna_reconstruction_loss.item()
            ),
            "adversarial_alignment": float(
                adversarial_alignment_loss.item()
            ),
            "discriminator": float(discriminator_loss.item()),
            "discriminator_accuracy": float(accuracy.item()),
            }

        for key, value in values.items():
            totals[key] += value
        batches += 1

    return {
        key: value / max(batches, 1)
        for key, value in totals.items()
    }


def export_effective_gene_morphology_weights(
    rna_model: Autoencoder,
    morphology_reference: Autoencoder,
    gene_names: list[str],
    morphology_names: list[str],
    output_dir: Path,
) -> None:
    """Export end-to-end linear gene-to-morphology coefficients.

    For the prediction pathway

        RNA -> RNA encoder -> morphology decoder -> morphology,

    the effective coefficient matrix is

        W_effective = W_morphology_decoder @ W_RNA_encoder.

    Rows correspond to morphology features and columns correspond to genes.
    Relative coefficients are normalized within each morphology feature so
    that the largest absolute coefficient has magnitude 100.
    """

    with torch.no_grad():
        rna_encoder_weights = (
            rna_model.encoder.network.weight
            .detach()
            .cpu()
            .numpy()
        )
        rna_encoder_bias = (
            rna_model.encoder.network.bias
            .detach()
            .cpu()
            .numpy()
        )
        morphology_decoder_weights = (
            morphology_reference.decoder.network.weight
            .detach()
            .cpu()
            .numpy()
        )
        morphology_decoder_bias = (
            morphology_reference.decoder.network.bias
            .detach()
            .cpu()
            .numpy()
        )

    expected_rna_shape = (LATENT_DIM, len(gene_names))
    expected_morphology_shape = (len(morphology_names), LATENT_DIM)

    if rna_encoder_weights.shape != expected_rna_shape:
        raise ValueError(
            "RNA encoder weight shape does not match the gene names: "
            f"weights={rna_encoder_weights.shape}, "
            f"expected={expected_rna_shape}."
        )

    if morphology_decoder_weights.shape != expected_morphology_shape:
        raise ValueError(
            "Morphology decoder weight shape does not match the morphology "
            f"feature names: weights={morphology_decoder_weights.shape}, "
            f"expected={expected_morphology_shape}."
        )

    effective_weights = (
        morphology_decoder_weights @ rna_encoder_weights
    )

    # The end-to-end intercept is not used for gene ranking, but it is saved
    # for completeness because it is part of the full linear mapping.
    effective_intercept = (
        morphology_decoder_weights @ rna_encoder_bias
        + morphology_decoder_bias
    )

    max_absolute_weight = np.max(
        np.abs(effective_weights),
        axis=1,
        keepdims=True,
    )
    safe_denominator = np.where(
        max_absolute_weight > 0,
        max_absolute_weight,
        1.0,
    )
    relative_weights = (
        effective_weights / safe_denominator
    ) * 100.0

    effective_dataframe = pd.DataFrame(
        effective_weights,
        index=morphology_names,
        columns=gene_names,
    )
    effective_dataframe.index.name = "morphology_feature"
    effective_dataframe.to_csv(
        output_dir / "effective_gene_morphology_weights.csv"
    )

    relative_dataframe = pd.DataFrame(
        relative_weights,
        index=morphology_names,
        columns=gene_names,
    )
    relative_dataframe.index.name = "morphology_feature"
    relative_dataframe.to_csv(
        output_dir / "relative_effective_gene_morphology_weights.csv"
    )

    intercept_dataframe = pd.DataFrame(
        {
            "morphology_feature": morphology_names,
            "effective_intercept": effective_intercept,
        }
    )
    intercept_dataframe.to_csv(
        output_dir / "effective_morphology_intercepts.csv",
        index=False,
    )

    # Long-form table is convenient for ranking and plotting genes for each
    # morphology feature.
    long_records: list[pd.DataFrame] = []
    for feature_index, feature_name in enumerate(morphology_names):
        feature_table = pd.DataFrame(
            {
                "morphology_feature": feature_name,
                "gene": gene_names,
                "effective_coefficient": effective_weights[
                    feature_index,
                    :,
                ],
                "relative_coefficient_percent": relative_weights[
                    feature_index,
                    :,
                ],
            }
        )
        feature_table["absolute_effective_coefficient"] = np.abs(
            feature_table["effective_coefficient"]
        )
        feature_table["absolute_relative_coefficient_percent"] = np.abs(
            feature_table["relative_coefficient_percent"]
        )
        feature_table = feature_table.sort_values(
            "absolute_effective_coefficient",
            ascending=False,
        ).reset_index(drop=True)
        feature_table["rank"] = np.arange(
            1,
            len(feature_table) + 1,
        )
        long_records.append(feature_table)

    long_dataframe = pd.concat(long_records, ignore_index=True)
    long_dataframe.to_csv(
        output_dir / "effective_gene_morphology_weights_long.csv",
        index=False,
    )

    top20_dataframe = (
        long_dataframe[long_dataframe["rank"] <= 20]
        .copy()
        .reset_index(drop=True)
    )
    top20_dataframe.to_csv(
        output_dir / "top20_genes_by_morphology_feature.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir / "effective_gene_morphology_weights.npz",
        effective_weights=effective_weights,
        relative_weights=relative_weights,
        effective_intercept=effective_intercept,
        gene_names=np.asarray(gene_names, dtype=str),
        morphology_names=np.asarray(morphology_names, dtype=str),
    )


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data_path = args.output_dir / "prepared_data.npz"
    preprocessing_path = args.output_dir / "preprocessing.joblib"
    morphology_checkpoint_path = (
        args.output_dir / "best_morphology_autoencoder.pt"
    )

    for required_path in (
        data_path,
        preprocessing_path,
        morphology_checkpoint_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required Stage 1 output not found: {required_path}"
            )

    arrays = np.load(data_path, allow_pickle=False)
    preprocessing = joblib.load(preprocessing_path)

    rna_train = arrays["rna_train"]
    rna_validation = arrays["rna_validation"]
    rna_test = arrays["rna_test"]
    morphology_train = arrays["morphology_train"]
    morphology_validation = arrays["morphology_validation"]
    morphology_test = arrays["morphology_test"]
    test_sample_ids = arrays["test_sample_ids"]

    gene_names = list(preprocessing["gene_names"])
    morphology_names = list(preprocessing["morphology_names"])

    morphology_checkpoint = torch.load(
        morphology_checkpoint_path,
        map_location=device,
    )

    morphology_reference = Autoencoder(
        input_dim=morphology_train.shape[1],
        latent_dim=LATENT_DIM,
    ).to(device)
    morphology_reference.load_state_dict(
        morphology_checkpoint["state_dict"]
    )
    freeze_module(morphology_reference)

    rna_model = Autoencoder(
        input_dim=rna_train.shape[1],
        latent_dim=LATENT_DIM,
    ).to(device)
    discriminator = LatentDiscriminator(
        latent_dim=LATENT_DIM,
        hidden_dim=DISC_HIDDEN_DIM,
        dropout=DISC_DROPOUT,
    ).to(device)

    rna_model.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    train_loader = make_loader(
        rna_train,
        morphology_train,
        shuffle=True,
    )
    validation_loader = make_loader(
        rna_validation,
        morphology_validation,
        shuffle=False,
    )
    test_loader = make_loader(
        rna_test,
        morphology_test,
        shuffle=False,
    )

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    optimizer_rna = optim.Adam(
        rna_model.parameters(),
        lr=LR_RNA,
        weight_decay=WD_RNA,
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=LR_DISC,
        weight_decay=WD_DISC,
    )

    best_loss = float("inf")
    best_rna_state = None
    best_discriminator_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = alignment_epoch(
            rna_model,
            morphology_reference,
            discriminator,
            train_loader,
            mse,
            bce,
            device,
            optimizer_rna,
            optimizer_discriminator,
        )
        validation_metrics = alignment_epoch(
            rna_model,
            morphology_reference,
            discriminator,
            validation_loader,
            mse,
            bce,
            device,
        )

        row = {"epoch": epoch}
        row.update(
            {
                f"train_{key}": value
                for key, value in train_metrics.items()
            }
        )
        row.update(
            {
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            }
        )
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"RNA rec={train_metrics['rna_reconstruction']:.6f} | "
            f"adv={train_metrics['adversarial_alignment']:.6f} | "
            f"disc acc={train_metrics['discriminator_accuracy']:.3f} | "
            f"validation objective="
            f"{validation_metrics['total_rna']:.6f}"
        )

        if validation_metrics["total_rna"] < best_loss - 1e-7:
            best_loss = validation_metrics["total_rna"]
            best_rna_state = copy.deepcopy(rna_model.state_dict())
            best_discriminator_state = copy.deepcopy(
                discriminator.state_dict()
            )
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if (
        best_rna_state is None
        or best_discriminator_state is None
    ):
        raise RuntimeError("No valid Stage 2 checkpoint was saved.")

    rna_model.load_state_dict(best_rna_state)
    discriminator.load_state_dict(best_discriminator_state)

    stage2_checkpoint = {
        "rna_state_dict": best_rna_state,
        "discriminator_state_dict": best_discriminator_state,
        "morphology_checkpoint": str(morphology_checkpoint_path),
        "latent_dim": LATENT_DIM,
        "discriminator_hidden_dim": DISC_HIDDEN_DIM,
        "discriminator_dropout": DISC_DROPOUT,
        "autoencoder_type": "single_layer_linear",
        "n_conditions": 0,
        "best_validation_objective": best_loss,
        "alpha_rna_reconstruction": ALPHA_RNA_RECON,
        "lambda_adversarial": LAMBDA_ADV,
    }

    torch.save(
        stage2_checkpoint,
        args.output_dir / "best_rna_morphology_alignment.pt",
    )

    combined_checkpoint = {
        "rna_state_dict": best_rna_state,
        "morphology_state_dict": morphology_checkpoint["state_dict"],
        "discriminator_state_dict": best_discriminator_state,
        "latent_dim": LATENT_DIM,
        "discriminator_hidden_dim": DISC_HIDDEN_DIM,
        "discriminator_dropout": DISC_DROPOUT,
        "autoencoder_type": "single_layer_linear",
        "n_conditions": 0,
    }

    torch.save(
        combined_checkpoint,
        args.output_dir / "two_stage_crossmodal_autoencoder.pt",
    )

    pd.DataFrame(history).to_csv(
        args.output_dir / "rna_alignment_training_history.csv",
        index=False,
    )

    rna_model.eval()
    morphology_reference.eval()

    export_effective_gene_morphology_weights(
        rna_model=rna_model,
        morphology_reference=morphology_reference,
        gene_names=gene_names,
        morphology_names=morphology_names,
        output_dir=args.output_dir,
    )

    with torch.no_grad():
        rna_tensor = torch.tensor(
            rna_test,
            dtype=torch.float32,
            device=device,
        )
        predicted_morphology = (
            morphology_reference.decode(
                rna_model.encode(rna_tensor)
            )
            .cpu()
            .numpy()
        )

    predicted_dataframe = pd.DataFrame(
        predicted_morphology,
        columns=morphology_names,
    )
    predicted_dataframe.insert(
        0,
        "sample_id",
        test_sample_ids,
    )
    predicted_dataframe.to_csv(
        args.output_dir / "predicted_morphology.csv",
        index=False,
    )

    true_dataframe = pd.DataFrame(
        morphology_test,
        columns=morphology_names,
    )
    true_dataframe.insert(
        0,
        "sample_id",
        test_sample_ids,
    )
    true_dataframe.to_csv(
        args.output_dir / "true_morphology.csv",
        index=False,
    )

    test_metrics = alignment_epoch(
        rna_model,
        morphology_reference,
        discriminator,
        test_loader,
        mse,
        bce,
        device,
    )

    with open(
        args.output_dir / "stage2_configuration.json",
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
                "alpha_rna_reconstruction": ALPHA_RNA_RECON,
                "lambda_adversarial": LAMBDA_ADV,
                "test_alignment_metrics": test_metrics,
            },
            handle,
            indent=2,
        )

    print(
        "Saved Stage 2 checkpoint from epoch "
        f"{best_epoch}: "
        f"{args.output_dir / 'best_rna_morphology_alignment.pt'}"
    )
    print(
        "Saved predicted morphology: "
        f"{args.output_dir / 'predicted_morphology.csv'}"
    )
    print(
        "Saved true morphology: "
        f"{args.output_dir / 'true_morphology.csv'}"
    )
    print(
        "Saved effective gene-to-morphology weights: "
        f"{args.output_dir / 'effective_gene_morphology_weights.csv'}"
    )
    print(
        "Saved top-20 genes for each morphology feature: "
        f"{args.output_dir / 'top20_genes_by_morphology_feature.csv'}"
    )



if __name__ == "__main__":
    main()
