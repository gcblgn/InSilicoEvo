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
- **Tc-Cut1** (a cutinase) — cold adaptation: reducing optimal working temperature
- **Pa-LipA** (a lipase) — heat adaptation: increasing optimal working temperature

---

## Pipeline Overview

The pipeline consists of four main steps:

### Step 1 — Data Collection & Feature Engineering

The [BRENDA](https://www.brenda-enzymes.org/) enzyme database (3.5 GB, accessed
via Zenodo) was used as the data source. Enzymes were filtered by **host
organism ecological niche** (psychrophilic, mesophilic, or thermophilic)
rather than raw temperature values, to prevent bias in the machine learning model.

From each enzyme sequence, a comprehensive set of **biochemical and physicochemical features**
were calculated using [Biopython](https://biopython.org/)'s `ProteinAnalysis`
module. These features cover several categories, including:

- **Global protein properties** — aromaticity, hydrophobicity, instability index, isoelectric point (pI), molecular weight, and more
- **Amino acid frequencies** — relative occurrence of each of the 20 standard amino acids
- **Dipeptide frequencies** — relative occurrence of all possible two-amino-acid combinations
- **Optimal growth temperature (OGT)** — the temperature at which the host organism grows best

The exact number of features may vary depending on the enzyme and the
filters applied during preprocessing.
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

## Repository Structure
```
InSilicoEvo/
│
├── 0_extract_data.py              # Data loading and organism-based filtering
├── 1_calculate_enzyme_features.py # 456-feature extraction with Biopython
├── 2_train_model_automlV3.py      # PyCaret AutoML model comparison & selection
├── 3_directed_evolutionV5.py      # Variant generation + Topt prediction
├── 5_esmfold_structure_prediction.py  # ESMFold structural validation
│
└── enzyme_feature_lib.py          # Shared feature calculation library
```

---

## Requirements
```
biopython
pycaret[full]
xgboost
lightgbm
catboost
torch
transformers
esm
pandas
numpy
scikit-learn
```

---

## Reproducibility

- Data: BRENDA SQLite database, version 2018.2 (available on [Zenodo](https://zenodo.org/))
- All scripts are self-contained and runnable in sequence (0 → 5)
- No proprietary tools or wet-lab data required

---

## Future Directions

- Rosetta (ΔΔG) as a third validation layer
- GROMACS molecular dynamics as a fourth validation layer
- pH optimization as an additional case study
- Web or CLI interface for broader accessibility

---

## Competition

**ISEF 2026** | Fair ID: TUR002 | Project ID: CBIO002  
Category: Computational Biology and Bioinformatics


