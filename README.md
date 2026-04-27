# Stability vs Drift in Credit Risk Modeling

This project studies how credit-risk models behave under temporal distribution shift using a curated Lending Club dataset. The central question is:

> When credit-risk models are trained on earlier Lending Club loans and evaluated on later loans, how do predictive performance, probability calibration, dataset drift, and feature reliance change over time?

The project uses a deployment-realistic forward-in-time evaluation rather than a random IID train/test split. The main benchmark compares structured-feature models across future years, then extends the analysis with dataset drift metrics, feature-reliance artifacts, and a sample-matched LLM evaluation.

## Main entry point

The canonical runner is:

```text
reproduce_project.ipynb
```

The notebook is intended to be a thin orchestration layer. Reusable project logic lives in `src/`.

## Project pipeline

The final notebook performs the following workflow:

1. Clone the repository and install dependencies.
2. Mount Google Drive.
3. Load and clean the Lending Club dataset.
4. Build structured feature groups.
5. Create a temporal train/validation/test split:
   - train: years `<= 2013`
   - validation: `2014`
   - test: `2015–2018`
6. Compute dataset drift:
   - default base-rate shift
   - Population Stability Index (PSI)
7. Export prompt-ready LLM examples.
8. Load or regenerate a 2,500-row LLM temporal evaluation:
   - 500 validation rows from 2014
   - 500 test rows from each future year 2015–2018
9. Train, threshold-tune, and evaluate structured models:
   - logistic regression
   - XGBoost
   - small MLP
10. Compare structured models and the LLM on the same sampled rows.
11. Extract feature-reliance artifacts for logistic regression and XGBoost.
12. Save final tables and plots to `results/`.

## Models evaluated

### Structured models

The primary full-dataset benchmark evaluates:

- logistic regression
- XGBoost
- small MLP

Each model is trained on earlier loans, tuned on the 2014 validation year, and evaluated on future years 2015–2018.

### LLM evaluator

The project also includes a sample-based LLM extension using `gpt-4.1-nano`.

The LLM receives prompt-formatted borrower text and selected structured loan context, returns a default-risk probability, and is evaluated under the same temporal structure as the structured models. Because full LLM inference over the complete validation/test set would require over one million API calls, the LLM evaluation uses a representative 2,500-row temporal sample. The structured models are also evaluated on this exact same sampled set for a fair sample-matched comparison.

Saved LLM predictions are included in `results/`, so an OpenAI API key is not required unless regenerating those predictions.

## Metrics and analyses

The project reports:

- AUC by future year
- F1 by future year
- validation-tuned thresholds
- Brier score
- Expected Calibration Error (ECE)
- reliability-bin summaries
- default base-rate shift
- PSI by feature and year
- logistic-regression coefficient-based feature reliance
- XGBoost feature importance
- top-feature overlap between logistic regression and XGBoost
- sample-matched LLM vs structured-model comparison

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
    ├── drift.py
    ├── evaluate.py
    ├── feature_stability.py
    ├── llm_eval.py
    ├── llm_prep.py
    ├── load_data.py
    ├── preprocess.py
    ├── sample_matched_eval.py
    ├── temporal_split.py
    ├── thresholding.py
    └── models/
        ├── __init__.py
        ├── logistic.py
        ├── mlp_model.py
        └── xgboost_model.py
```

## Important result artifacts

The `results/` directory contains the generated CSV and PNG artifacts used in the final report.

Key structured-model artifacts include:

```text
summary_table.csv
temporal_metrics_all_models.csv
temporal_calibration_all_models.csv
performance_calibration_comparison.csv
best_thresholds.csv
comparison_auc_by_year.png
comparison_f1_by_year.png
comparison_brier_by_year.png
comparison_ece_by_year.png
```

Key drift artifacts include:

```text
base_rate_by_year.csv
base_rate_by_year.png
psi_by_feature_year.csv
psi_top_features.csv
psi_heatmap_top_features.png
drift_summary_by_year.csv
final_performance_calibration_drift_summary.csv
```

Key feature-reliance artifacts include:

```text
logreg_feature_importance.csv
xgboost_feature_importance.csv
logreg_transformed_feature_importance.csv
xgboost_transformed_feature_importance.csv
feature_importance_overlap.csv
top_logreg_features.png
top_xgboost_features.png
```

Key LLM and sample-matched comparison artifacts include:

```text
llm_temporal_eval_batch1_predictions.csv
llm_temporal_eval_predictions.csv
llm_temporal_eval_metrics.csv
llm_temporal_eval_threshold_search.csv
sample_matched_predictions.csv
sample_matched_yearly_metrics.csv
sample_matched_model_summary.csv
sample_matched_threshold_search.csv
sample_matched_auc_by_year.png
sample_matched_f1_by_year.png
sample_matched_brier_by_year.png
sample_matched_ece_by_year.png
```

## Dataset

The raw Lending Club dataset is not included in this repository because of size. To reproduce the notebook, download the curated Lending Club dataset from Zenodo record `11295916`:

```text
https://zenodo.org/records/11295916
```

The notebook expects the dataset at this default Colab path:

```text
/content/drive/MyDrive/datasets/lending_club.csv
```

If using a different location, update `DATA_PATH` in the configuration cell of `reproduce_project.ipynb`.

## Reproduction notes

By default, the notebook uses saved LLM prediction artifacts from `results/` and does not call the OpenAI API.

To regenerate the LLM predictions, set this flag in the notebook configuration cell:

```python
RUN_LLM_EVAL_INFERENCE = True
```

Then store an OpenAI API key in Google Colab Secrets with the exact name:

```text
OPENAI_API_KEY
```

Do not hard-code API keys in the notebook or repository.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The required packages include:

```text
pandas
numpy
matplotlib
scikit-learn
pyarrow
xgboost
openai
tqdm
```

## Project scope

This repository is the final implementation of the CS 4365 project. It includes the original structured temporal benchmark, calibration analysis, drift analysis, feature-reliance analysis, and an LLM-based sample-matched temporal evaluation.