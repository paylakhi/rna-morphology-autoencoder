#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


NAVY = "#000080"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-dir", type=Path, required=True)
    p.add_argument("--weights-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--features-file", type=Path, default=None)
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def slug(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("_")


def load_data(training_dir):
    prep = joblib.load(training_dir / "preprocessing.joblib")
    a = np.load(training_dir / "prepared_data.npz")

    X = np.vstack([a["rna_train"], a["rna_validation"], a["rna_test"]])
    Y = np.vstack([
        a["morphology_train"],
        a["morphology_validation"],
        a["morphology_test"],
    ])

    return X, Y, list(prep["gene_names"]), list(prep["morphology_names"])


def choose_features(path, available):
    if path is None:
        return available

    features = (
        pd.read_csv(path, header=None)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
        .tolist()
    )

    missing = [x for x in features if x not in available]
    if missing:
        raise ValueError(f"Features not found: {missing}")

    return features


def correlations(X, y, genes):
    rows = []

    for j, gene in enumerate(genes):
        r, p = pearsonr(X[:, j], y)
        rows.append([gene, r, p])

    out = pd.DataFrame(rows, columns=["gene", "pearson_r", "pearson_p"])
    out["pearson_q"] = multipletests(out["pearson_p"], method="fdr_bh")[1]
    return out


def plot_weights(top, feature, path):
    d = top.sort_values("relative_coefficient_percent")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.barh(
        d["gene"],
        d["relative_coefficient_percent"],
        color=NAVY,
        height=0.72,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Relative effective coefficient (%)")
    ax.set_ylabel("")
    ax.set_title(feature, fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(top, feature, X, y, genes, path):
    cols = 4
    rows = math.ceil(len(top) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 2.7 * rows),
        squeeze=False,
    )

    gene_index = {g: i for i, g in enumerate(genes)}

    for ax, row in zip(axes.flat, top.itertuples(index=False)):
        x = X[:, gene_index[row.gene]]

        ax.scatter(x, y, s=14, alpha=0.7, color=NAVY)

        if np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, m * xx + b, color="black", linewidth=1)

        ax.set_title(row.gene, fontsize=9)
        ax.text(
            0.04,
            0.96,
            f"r = {row.pearson_r:.2f}\nq = {row.pearson_q:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_xlabel("Gene expression", fontsize=8)
        ax.set_ylabel("Morphology", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)

    for ax in axes.flat[len(top):]:
        ax.axis("off")

    fig.suptitle(feature, fontsize=11)
    plt.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, genes, morph_features = load_data(args.training_dir)
    weights = pd.read_csv(args.weights_csv)

    required = {
        "morphology_feature",
        "gene",
        "effective_coefficient",
        "relative_coefficient_percent",
    }
    if not required.issubset(weights.columns):
        raise ValueError(f"weights CSV must contain: {sorted(required)}")

    features = choose_features(args.features_file, morph_features)
    all_top = []

    for feature in features:
        if feature not in set(weights["morphology_feature"]):
            raise ValueError(f"{feature} not found in weights file.")

        fi = morph_features.index(feature)
        corr = correlations(X, Y[:, fi], genes)

        w = weights[weights["morphology_feature"] == feature].copy()
        w["abs_weight"] = w["effective_coefficient"].abs()

        top = (
            w.sort_values("abs_weight", ascending=False)
            .head(args.top_n)
            .merge(corr, on="gene", how="left")
        )

        top["rank"] = np.arange(1, len(top) + 1)
        all_top.append(top)

        out = args.output_dir / slug(feature)
        out.mkdir(exist_ok=True)

        top.to_csv(out / "top_genes.csv", index=False)

        plot_weights(
            top,
            feature,
            out / "top_gene_weights",
        )

        plot_correlations(
            top,
            feature,
            X,
            Y[:, fi],
            genes,
            out / "top_gene_correlations",
        )

        print(f"[OK] {feature}")

    pd.concat(all_top, ignore_index=True).to_csv(
        args.output_dir / "top_genes_all_features.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
