# Stability vs Drift in Credit Risk Modeling

This project studies how credit-risk models behave under temporal distribution shift using a curated Lending Club dataset.

The current codebase supports a reproducible benchmark with three structured-feature models:
- logistic regression
- XGBoost
- a small MLP

The workflow uses a temporal train/validation/test split, selects classification thresholds on a validation year, evaluates on future years, analyzes probability calibration over time, and saves the resulting tables and plots.

It also exports a prompt-ready sample for a future LLM-based extension.

## Main entry point

The canonical front-facing runner for the project is:

```text
reproduce_project.ipynb
```

## Current pipeline

The notebook currently does the following:
1. load and clean the Lending Club dataset
2. build the structured feature set
3. create a temporal split:
   - train: years <= 2013
   - validation: 2014
   - test: 2015-2018
4. train logistic regression, XGBoost, and MLP
5. tune each model's threshold on the validation year
6. evaluate future-year performance using AUC and F1
7. evaluate future-year calibration using ECE and Brier score
8. generate reliability diagrams by test year
9. compare calibration drift against accuracy drift
10. save results to `results/`
11. export an LLM evaluation sample for later checkpoints

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

## Main generated outputs

A successful run writes outputs to `results/`.

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
