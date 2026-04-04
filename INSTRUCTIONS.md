# INSTRUCTIONS.md

## Purpose

This file explains how to reproduce the current project workflow and how the repository is organized.

The canonical front-facing runner is:

```text
reproduce_project.ipynb
```

That notebook is the main file a user or LLM should run to reproduce the current checkpoint.

---

## Repository structure

```text
CS-4365/
├── README.md
├── INSTRUCTIONS.md
├── requirements.txt
├── .gitignore
├── reproduce_project.ipynb
├── results/
└── src/
    ├── __init__.py
    ├── calibration.py
    ├── evaluate.py
    ├── llm_prep.py
    ├── load_data.py
    ├── preprocess.py
    ├── temporal_split.py
    ├── thresholding.py
    └── models/
        ├── __init__.py
        ├── logistic.py
        ├── mlp_model.py
        └── xgboost_model.py
```

---

## What each file does

### `reproduce_project.ipynb`
The single canonical notebook for reproducing the current project workflow.

### `src/load_data.py`
Loads the dataset from CSV or Parquet, validates required columns, parses dates, normalizes text fields, and creates time-derived columns such as `year`.

### `src/preprocess.py`
Defines the structured feature set and builds preprocessing logic for the tabular models.

### `src/temporal_split.py`
Creates the temporal train/validation/test split used by the notebook.

### `src/thresholding.py`
Tunes classification thresholds on the validation set.

### `src/evaluate.py`
Evaluates year-by-year model performance and computes AUC/F1 outputs.

### `src/calibration.py`
Computes year-by-year calibration outputs including Brier score, ECE, reliability tables, reliability diagrams, and calibration trend plots.

### `src/models/logistic.py`
Builds and fits the logistic regression pipeline.

### `src/models/xgboost_model.py`
Builds and fits the XGBoost pipeline.

### `src/models/mlp_model.py`
Builds and fits the small MLP pipeline.

### `src/llm_prep.py`
Builds and exports a prompt-ready sample for a future LLM evaluation workflow. It does not call an LLM; it only prepares the data.

---

## Exact workflow implemented by the notebook

The notebook currently performs the following steps:
1. clone the repository and install dependencies
2. mount Google Drive
3. load and clean the Lending Club dataset
4. build the structured feature set
5. create the temporal split:
   - train: years <= 2013
   - validation: 2014
   - test: 2015-2018
6. prepare and export a future LLM evaluation sample
7. train logistic regression, XGBoost, and MLP
8. tune thresholds on the validation year
9. evaluate on future years using AUC and F1
10. evaluate on future years using ECE and Brier score
11. generate reliability diagrams
12. compare calibration drift against accuracy drift
13. save final plots and CSV summaries to `results/`

---

## Dataset required to run the notebook

This repository does not include the Lending Club dataset file itself. Before running the notebook, download the curated Lending Club dataset from Zenodo record 11295916 and place it in a location accessible from your runtime.

Recommended Colab location:

```text
/content/drive/MyDrive/datasets/lending_club.csv
```

Then update `DATA_PATH` in the configuration cell of `reproduce_project.ipynb`. Parquet is preferred when available.

---

## Google Colab reproduction steps

### 1. Open Google Colab
Start a fresh Colab session.

### 2. Open `reproduce_project.ipynb`
Use the notebook in this repository as the main runner.

### 3. Make sure the dataset is already in Google Drive
Place the Lending Club dataset somewhere in Drive and set the notebook's `DATA_PATH` accordingly.

### 4. Run the notebook from top to bottom
The notebook handles:
- cloning the repository
- installing dependencies
- mounting Google Drive
- loading the dataset
- building features
- creating splits
- training and evaluating the three models
- exporting final outputs

---

## Generated outputs

A successful run writes outputs to:

```text
results/
```

The main artifacts include:

### Final comparison artifacts
- `temporal_metrics_all_models.csv`
- `temporal_calibration_all_models.csv`
- `performance_calibration_comparison.csv`
- `reliability_bins_all_models.csv`
- `drift_comparison_table.csv`
- `summary_table.csv`
- `best_thresholds.csv`
- `comparison_auc_by_year.png`
- `comparison_f1_by_year.png`
- `comparison_auc_by_time_gap.png`
- `comparison_f1_by_time_gap.png`
- `comparison_ece_by_year.png`
- `comparison_brier_by_year.png`
- `comparison_ece_by_time_gap.png`
- `comparison_brier_by_time_gap.png`

### Per-model artifacts
- `temporal_metrics_logreg.csv`
- `temporal_metrics_xgboost.csv`
- `temporal_metrics_mlp.csv`
- `temporal_calibration_logreg.csv`
- `temporal_calibration_xgboost.csv`
- `temporal_calibration_mlp.csv`
- `reliability_bins_logreg.csv`
- `reliability_bins_xgboost.csv`
- `reliability_bins_mlp.csv`
- `logreg_validation_threshold_search.csv`
- `xgboost_validation_threshold_search.csv`
- `mlp_validation_threshold_search.csv`
- per-model AUC/F1 plots
- per-model reliability diagrams
- per-model ECE plots
- per-model Brier score plots
- per-model accuracy-vs-calibration drift plots

### Future extension artifacts
- `llm_eval_sample.csv`
- `llm_eval_sample.jsonl`

---

## What an LLM should read first

An LLM trying to understand or reproduce this repository should read files in this order:
1. `README.md`
2. `INSTRUCTIONS.md`
3. `reproduce_project.ipynb`
4. `src/load_data.py`
5. `src/preprocess.py`
6. `src/temporal_split.py`
7. `src/thresholding.py`
8. `src/evaluate.py`
9. `src/calibration.py`
10. `src/models/logistic.py`
11. `src/models/xgboost_model.py`
12. `src/models/mlp_model.py`
13. `src/llm_prep.py`

---

## Current intended scope

The current implemented checkpoint includes:
- three structured-feature models
- a temporal train/validation/test workflow
- threshold tuning on validation
- future-year evaluation on 2015-2018 using AUC and F1
- future-year calibration evaluation using ECE and Brier score
- reliability diagrams
- calibration-vs-accuracy drift comparison
- export of a prompt-ready LLM evaluation sample

Future checkpoints can extend the same pipeline with drift metrics beyond calibration and LLM-based evaluation.
