# Stability vs Drift in Credit Risk Modeling

This project studies how credit-risk models behave under temporal distribution shift using a curated Lending Club dataset.

The current codebase supports a reproducible benchmark with three structured-feature models:

- logistic regression
- XGBoost
- a small MLP

The workflow uses a temporal train/validation/test split, selects classification thresholds on a validation year, evaluates on future years, and saves the resulting tables and plots.

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
7. save results to `results/`
8. export an LLM evaluation sample for later checkpoints

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