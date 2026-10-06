# Cellular morphology emerges from polygenic, distributed transcriptional variation

This repository contains the analysis code, final analysis configurations, and shareable derived results associated with the manuscript:

**Cellular morphology emerges from polygenic, distributed transcriptional variation**

**Authors:** Seyedehzahra Paylakhi, Rafael Geurgas, Antionette Yasko, Robbee Wedow, and Matthew Tegtmeyer

## Overview

This study investigates whether cellular morphology can be predicted from distributed transcriptional variation.

The analysis integrates:

- matched L1000 gene-expression and Cell Painting profiles from four perturbation datasets;
- matched RNA-seq and Cell Painting profiles from human iPSC donors;
- independent CRISPR Cell Painting profiles for validation; and
- previously generated cis-eQTL results for complementary genetic analyses.

The cross-modal model was trained on perturbation data and evaluated for transfer to natural inter-individual variation in a cohort of 100 genetically diverse iPSC donors.

---

# Data availability

## Perturbation datasets

The LUAD, LINCS, TA-ORF, and CDRP-bio datasets used for model training and perturbation-level evaluation were obtained from the multimodal resource described by:

Haghighi M, Caicedo JC, Cimini BA, Carpenter AE, Singh S.  
**High-dimensional gene expression and morphology profiles of cells across 28,000 genetic and chemical perturbations.**  
*Nature Methods* (2022), 19:1550–1557.

DOI: https://doi.org/10.1038/s41592-022-01667-0

The preprocessed matched L1000 gene-expression and Cell Painting profiles are publicly available through the Registry of Open Data on AWS / Cell Painting Gallery.

**Cell Painting Gallery dataset:** `cpg0003-rosetta`

Registry of Open Data on AWS:  
https://registry.opendata.aws/cellpainting-gallery/

Original Haghighi et al. repository:  
https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods

The public preprocessed profiles can be obtained from:

```bash
aws s3 sync \
  --no-sign-request \
  s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/preprocessed_data/ \
  ./preprocessed_data/
```

The four perturbation resources analyzed in this study are the LUAD, LINCS, TA-ORF, and CDRP-bio matched Cell Painting/L1000 datasets described by Haghighi et al.

Because these source data are already publicly maintained by the original data providers, copies of the complete original perturbation datasets are not duplicated in this repository.

---

## Human donor Cell Painting and genomic resources

The human iPSC donor morphology resource used in the donor-level analyses was derived from the cohort described by:

Tegtmeyer M, Arora J, Asgari S, et al.  
**High-dimensional phenotyping to define the genetic basis of cellular morphology.**  
*Nature Communications* (2024), 15:347.

DOI: https://doi.org/10.1038/s41467-023-44045-w

Raw Cell Painting images from this study are publicly available through the Cell Painting Gallery.

**Cell Painting Gallery dataset:** `cpg0022-cmqtl`

Registry of Open Data on AWS:  
https://registry.opendata.aws/cellpainting-gallery/

The original analysis code associated with the cmQTL study is available at:

https://github.com/broadinstitute/cmQTL

Whole-genome sequencing data associated with this donor resource were deposited in NCBI dbGaP under accession:

**dbGaP:** `phs002032.v1.p1`

Study page:  
https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs002032.v1.p1

Individual-level genomic data are subject to controlled-access requirements and are **not redistributed through this repository**.

---

## Donor transcriptomic and cis-eQTL resources

Donor transcriptomic and genetic resources used in the study are associated with previously published human stem-cell donor resources, including:

Wells MF, Nemesh J, Ghosh S, et al.  
**Natural variation in gene expression and viral susceptibility revealed by neural progenitor cell villages.**  
*Cell Stem Cell* (2023), 30:312–332.e13.

DOI: https://doi.org/10.1016/j.stem.2023.01.010

The Wells et al. study describes access to sequencing and genomic data from relevant hiPSC resources through AnVIL/dbGaP, including accession:

**dbGaP:** `phs002032`

AnVIL study/access information:  
https://anvilproject.org/data/studies/phs002032

Individual-level sequencing and genomic data subject to controlled-access requirements are **not redistributed through this repository**.

For the present study, analysis-ready matched donor RNA-expression, morphology, and cis-eQTL inputs were derived from previously generated donor resources. These analysis-ready individual-level inputs are not distributed through this public repository. The repository instead provides the analysis code, executed configurations, and shareable derived statistical outputs required to document the analyses performed after preparation of those inputs.

---

# Data redistribution and access restrictions

This repository distinguishes between source data and shareable derived analysis results.

### Public source data

Publicly available datasets are accessed from and maintained by their original repositories. These include:

- Haghighi et al. perturbation data (`cpg0003-rosetta`);
- Tegtmeyer et al. raw Cell Painting images (`cpg0022-cmqtl`).

These source datasets are not duplicated in this repository.

### Controlled-access or non-redistributed donor data

Individual-level genomic and sequencing data subject to controlled-access requirements are not redistributed through this repository.

Likewise, analysis-ready matched donor-level RNA-expression, morphology, and genetic/eQTL input matrices are not presented here as unrestricted public datasets.

Researchers requiring controlled-access source data should obtain the corresponding data through the original data providers and applicable access procedures.

The public repository instead preserves the code, analysis configuration, and shareable derived statistical outputs necessary to document how the reported analyses were performed.

---

# Code and software

Analyses were performed using Python 3.11 with:

- NumPy 1.26
- pandas 2.2
- SciPy 1.11
- scikit-learn 1.5
- PyTorch 2.3
- Matplotlib 3.9

