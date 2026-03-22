# InSilicoEvo
# InSilicoEvo

**A fully computational pipeline for bidirectional enzyme thermal adaptation**  
*ISEF 2026 — Computational Biology & Bioinformatics (CBIO)*

---

## What does this project do?

Enzymes are proteins that speed up chemical reactions. Their performance is
tightly linked to temperature: most enzymes only work well within a narrow
temperature range. Engineering enzymes to work better at different
temperatures usually requires expensive and time-consuming laboratory
experiments.

This project presents **InSilicoEvo**, a fully computational (in silico)
pipeline that predicts and generates thermally adapted enzyme variants
without any wet-lab work. The pipeline is designed to be **generalizable**:
it can be applied to different enzyme families in both directions — making
an enzyme work better in cold environments, or better in hot ones.

Two case studies were used to demonstrate the pipeline:
- **TfCut2** (a cutinase) — cold adaptation: reducing optimal working temperature
- **Pa-LipA** (a lipase) — heat adaptation: increasing optimal working temperature

---

## Pipeline Overview

The pipeline consists of four main steps:

### Step 1 — Data Collection & Feature Engineering

The [BRENDA](https://www.brenda-enzymes.org/) enzyme database (3.5 GB, accessed
via Zenodo) was used as the data source. Enzymes were filtered by **host
organism ecological niche** (psychrophilic, mesophilic, or thermophilic)
rather than raw temperature values, to prevent bias in the machine learning model.

From each enzyme sequence, **456 biochemical and physicochemical features**
were calculated using [Biopython](https://biopython.org/)'s `ProteinAnalysis`
module. These features cover:

| Category | # Features |
|---|---|
| Global protein properties (aromaticity, hydrophobicity, pI, etc.) | 10 |
| Amino acid frequencies | 20 |
| Dipeptide frequencies | 400 |
| Optimal growth temperature (OGT) | 1 |
| Physicochemical + structural derived | 25 |
| **Total** | **456** |

---

### Step 2 — Machine Learning Model for Topt Prediction

[PyCaret](https://pycaret.org/) AutoML was used to systematically compare
five regression algorithms for predicting enzyme optimal temperature (Topt):

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Extra Trees Regressor

Each model was trained and evaluated under the same protocol. The
**single best-performing model** was selected based on RMSE and R²,
and used in all downstream prediction steps.

> This is a model comparison and selection approach, not ensemble learning.

---

### Step 3 — In Silico Directed Evolution

Guided random mutations were introduced into specific regions of the
target enzyme's amino acid sequence to generate thousands of variants.
The trained ML model then predicted the **Topt** of each variant.

Variants whose predicted Topt moved in the desired direction (colder or
hotter) were kept for the next stage.

---

### Step 4 — Multi-Layer Validation

Promising variants passed through two independent validation layers:

**Layer 1 — Evolutionary plausibility (ESM-2)**  
[ESM-2](https://github.com/facebookresearch/esm) is a protein language model
trained on ~250 million natural protein sequences. It scores how
biologically realistic a given amino acid sequence is. Variants with low
plausibility scores are deprioritized.

**Layer 2 — Structural confidence (ESMFold)**  
[ESMFold](https://github.com/facebookresearch/esm) predicts the 3D structure
of each variant and produces **pLDDT** scores (per-residue confidence):

| pLDDT | Confidence |
|---|---|
| ≥ 90 | Very High |
| 70 – 90 | Confident |
| 50 – 70 | Moderate |
| < 50 | Low |

Variants with high mean pLDDT and well-folded predicted structures
are prioritized as final candidates.

---
