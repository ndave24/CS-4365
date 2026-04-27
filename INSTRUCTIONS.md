# INSTRUCTIONS.md

## Purpose

This file explains how to reproduce the final project workflow and how the repository is organized. It is written for both human readers and AI agents.

The canonical runner is:

```text
reproduce_project.ipynb
```

Run this notebook to reproduce the project outputs.

The notebook is designed as a thin orchestration layer. Core logic lives in `src/`.

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

---

## What each file does

### `reproduce_project.ipynb`

The main reproducibility notebook. It clones the repository, mounts Google Drive, loads the dataset, runs the temporal benchmark, computes drift and feature-reliance artifacts, loads or regenerates LLM predictions, and saves final result tables and plots.

### `src/load_data.py`

Loads the curated Lending Club dataset from CSV or Parquet, validates required columns, parses date fields, normalizes text fields, and creates time-derived fields such as `year`.

### `src/preprocess.py`

Defines the structured feature benchmark and builds preprocessing logic for mixed numeric/categorical tabular data.

### `src/temporal_split.py`

Creates the temporal train/validation/test split:

```text
train: years <= 2013
validation: 2014
test: 2015–2018
```

### `src/thresholding.py`

Tunes classification thresholds on the validation set, mainly by searching for the threshold that maximizes F1.

### `src/evaluate.py`

Evaluates future-year predictive performance for structured models and saves AUC/F1 outputs.

### `src/calibration.py`

Computes calibration metrics and reliability artifacts, including Brier score, Expected Calibration Error (ECE), reliability-bin tables, and calibration plots.

### `src/drift.py`

Computes dataset drift artifacts:

- default base-rate shift by year
- Population Stability Index (PSI) by feature and future year
- top drifting features
- year-level drift summaries
- PSI heatmap and base-rate plots

### `src/feature_stability.py`

Extracts feature-reliance artifacts for interpretable structured models:

- logistic-regression coefficient magnitudes
- XGBoost feature importances
- aggregation from transformed one-hot features back to original feature columns
- top-feature overlap between logistic regression and XGBoost

### `src/llm_prep.py`

Builds prompt-ready LLM evaluation rows from borrower text fields and selected structured loan context. It does not call an LLM by itself.

### `src/llm_eval.py`

Handles the LLM temporal evaluation workflow:

- builds a representative LLM validation/test sample
- calls the OpenAI API when regeneration is enabled
- saves predictions with resume/checkpoint support
- evaluates LLM probabilities by year
- tunes the LLM threshold on the 2014 validation sample

### `src/sample_matched_eval.py`

Runs a sample-matched comparison between the LLM and structured models. It evaluates logistic regression, XGBoost, MLP, and the LLM on the exact same sampled validation/test rows.

### `src/models/logistic.py`

Builds and fits the logistic regression pipeline.

### `src/models/xgboost_model.py`

Builds and fits the XGBoost pipeline.

### `src/models/mlp_model.py`

Builds and fits the small MLP pipeline.

---

## Dataset required to run the notebook

This repository does not include the Lending Club dataset file itself.

Download the curated Lending Club dataset from Zenodo record `11295916`:

```text
https://zenodo.org/records/11295916
```

Recommended Colab path:

```text
/content/drive/MyDrive/datasets/lending_club.csv
```

The notebook’s configuration cell currently uses:

```python
DATA_PATH = Path("/content/drive/MyDrive/datasets/lending_club.csv")
```

If you place the dataset elsewhere, update `DATA_PATH` in the configuration cell before running the notebook.

Parquet is also supported by the loader. If using a Parquet file, update `DATA_PATH` accordingly.

---

## Google Colab reproduction steps

### 1. Open the notebook

Open:

```text
reproduce_project.ipynb
```

in Google Colab.

### 2. Make sure the dataset is in Google Drive

Place the dataset at:

```text
/content/drive/MyDrive/datasets/lending_club.csv
```

or update `DATA_PATH` in the configuration cell.

### 3. Run the notebook

Run the notebook top to bottom.

The notebook will:

1. clone the GitHub repository,
2. install dependencies from `requirements.txt`,
3. mount Google Drive,
4. load and clean the dataset,
5. build structured features,
6. create the temporal split,
7. compute dataset drift,
8. export prompt-ready LLM samples,
9. load saved LLM predictions from `results/`,
10. train and evaluate logistic regression, XGBoost, and MLP,
11. run the sample-matched LLM comparison,
12. compute feature-reliance artifacts,
13. save final outputs to `results/`.

---

## OpenAI API instructions for regenerating LLM predictions

The final repository already includes saved LLM prediction artifacts in `results/`. Therefore, an OpenAI API key is **not required** to inspect the results or rerun the notebook in its default mode.

By default, the notebook contains:

```python
RUN_LLM_EVAL_INFERENCE = False
```

This means the notebook loads saved predictions from:

```text
results/llm_temporal_eval_batch1_predictions.csv
```

and does not call the OpenAI API.

### To regenerate LLM predictions

Only do this if you intentionally want to rerun LLM inference.

1. In Google Colab, open the left sidebar.
2. Click the key-shaped **Secrets** panel.
3. Add a secret named exactly:

```text
OPENAI_API_KEY
```

4. Paste your OpenAI API key as the value.
5. Enable notebook access to the secret.
6. In the notebook configuration cell, set:

```python
RUN_LLM_EVAL_INFERENCE = True
```

7. Run the LLM evaluation section.

The notebook uses:

```python
LLM_MODEL = "gpt-4.1-nano"
```

The LLM evaluation sample contains:

```text
500 validation rows from 2014
500 test rows from 2015
500 test rows from 2016
500 test rows from 2017
500 test rows from 2018
2,500 total rows
```

The LLM inference function includes checkpointing and resume support. It saves progress to:

```text
results/llm_temporal_eval_batch1_predictions.csv
```

Do not hard-code API keys in the notebook, source files, or repository.

---

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

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

If running in Colab, the notebook installs the dependencies automatically after cloning the repository.

---

## Exact final workflow implemented by the notebook

The final notebook performs the following steps:

1. **Clone repo and install dependencies**
   - Clones `ndave24/CS-4365`
   - Installs `requirements.txt`

2. **Mount Google Drive**
   - Gives Colab access to the Lending Club dataset.

3. **Set up imports**
   - Imports all reusable logic from `src/`.

4. **Configuration**
   - Sets `DATA_PATH`
   - Sets `RESULTS_DIR`
   - Sets temporal split years
   - Sets model configs
   - Sets LLM config and API-regeneration flag

5. **Load and clean dataset**
   - Reads the curated Lending Club data.
   - Parses date and target fields.

6. **Build structured feature groups**
   - Selects numeric and categorical features for the structured models.

7. **Create temporal split**
   - Train: years `<= 2013`
   - Validation: `2014`
   - Test: `2015–2018`

8. **Dataset drift analysis**
   - Computes default base-rate shift.
   - Computes PSI by feature/year.
   - Saves drift CSVs and plots.

9. **Export prompt-ready LLM sample**
   - Saves `llm_eval_sample.csv`
   - Saves `llm_eval_sample.jsonl`

10. **LLM temporal evaluation**
   - Builds the 2,500-row LLM evaluation sample.
   - Loads saved LLM predictions unless regeneration is enabled.
   - Evaluates LLM AUC, F1, Brier score, ECE, and threshold behavior by year.

11. **Train and evaluate structured models**
   - Trains logistic regression, XGBoost, and MLP.
   - Tunes thresholds on the 2014 validation year.
   - Evaluates by test year.

12. **Sample-matched comparison**
   - Compares the LLM and structured models on the exact same sampled rows.

13. **Feature-reliance analysis**
   - Extracts logistic coefficients and XGBoost importances.
   - Aggregates transformed features back to original feature names.
   - Computes top-feature overlap.

14. **Final tables and plots**
   - Saves combined performance, calibration, drift, LLM, and feature-reliance artifacts.

15. **Final output check**
   - Prints saved outputs and checks that expected final artifacts exist.

---

## Generated outputs

All outputs are written to:

```text
results/
```

### Core structured-model outputs

```text
summary_table.csv
best_thresholds.csv
temporal_metrics_all_models.csv
temporal_metrics_logreg.csv
temporal_metrics_xgboost.csv
temporal_metrics_mlp.csv
comparison_auc_by_year.png
comparison_f1_by_year.png
comparison_auc_by_time_gap.png
comparison_f1_by_time_gap.png
```

### Calibration outputs

```text
temporal_calibration_all_models.csv
temporal_calibration_logreg.csv
temporal_calibration_xgboost.csv
temporal_calibration_mlp.csv
reliability_bins_all_models.csv
reliability_bins_logreg.csv
reliability_bins_xgboost.csv
reliability_bins_mlp.csv
performance_calibration_comparison.csv
comparison_brier_by_year.png
comparison_brier_by_time_gap.png
comparison_ece_by_year.png
comparison_ece_by_time_gap.png
```

### Drift outputs

```text
base_rate_by_year.csv
base_rate_by_year.png
psi_by_feature_year.csv
psi_top_features.csv
psi_heatmap_top_features.png
drift_summary_by_year.csv
final_performance_calibration_drift_summary.csv
```

### Feature-reliance outputs

```text
logreg_feature_importance.csv
xgboost_feature_importance.csv
logreg_transformed_feature_importance.csv
xgboost_transformed_feature_importance.csv
feature_importance_overlap.csv
top_logreg_features.png
top_xgboost_features.png
```

### LLM outputs

```text
llm_eval_sample.csv
llm_eval_sample.jsonl
llm_temporal_eval_batch1_input_template.csv
llm_temporal_eval_batch1_predictions.csv
llm_temporal_eval_predictions.csv
llm_temporal_eval_metrics.csv
llm_temporal_eval_threshold_search.csv
```

### Sample-matched comparison outputs

```text
sample_matched_predictions.csv
sample_matched_yearly_metrics.csv
sample_matched_model_summary.csv
sample_matched_threshold_search.csv
sample_matched_auc_by_year.png
sample_matched_f1_by_year.png
sample_matched_brier_by_year.png
sample_matched_ece_by_year.png
```

---

## What an AI agent should read first

An AI agent trying to understand or modify this repository should read files in this order:

1. `README.md`
2. `INSTRUCTIONS.md`
3. `reproduce_project.ipynb`
4. `src/load_data.py`
5. `src/preprocess.py`
6. `src/temporal_split.py`
7. `src/thresholding.py`
8. `src/evaluate.py`
9. `src/calibration.py`
10. `src/drift.py`
11. `src/feature_stability.py`
12. `src/llm_prep.py`
13. `src/llm_eval.py`
14. `src/sample_matched_eval.py`
15. `src/models/logistic.py`
16. `src/models/xgboost_model.py`
17. `src/models/mlp_model.py`

---

## Notes and limitations

- The Lending Club dataset itself is not committed to the repository.
- The saved LLM predictions are included in `results/`, but regenerating them requires an OpenAI API key and API billing.
- The structured models are evaluated on the full future-year test set.
- The LLM is evaluated on a representative 2,500-row temporal sample because full LLM inference over all validation/test rows would require over one million API calls.
- The sample-matched comparison evaluates the structured models and LLM on the exact same sampled rows, making the LLM extension directly comparable within its sampled setting.
- API keys and secrets should never be committed to the repository.