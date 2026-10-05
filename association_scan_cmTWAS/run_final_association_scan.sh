#!/usr/bin/env bash
set -euo pipefail

# No additional numeric or categorical covariates were used.

: "${MORPH_CSV:?Set MORPH_CSV}"
: "${EXPR_CSV:?Set EXPR_CSV}"
: "${EQTL_TSV:?Set EQTL_TSV}"
: "${OUT_DIR:?Set OUT_DIR}"

mkdir -p "${OUT_DIR}"

python association_scan_cmTWAS/run_cmTWAS_eqtl_restricted.py \
  --morph_csv "${MORPH_CSV}" \
  --expr_csv "${EXPR_CSV}" \
  --eqtl_tsv "${EQTL_TSV}" \
  --out_dir "${OUT_DIR}" \
  --eqtl_p_cutoff 0.05 \
  --min_n 30
