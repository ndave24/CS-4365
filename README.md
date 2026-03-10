# Credit Risk Temporal Drift in Lending Club Data

This project studies how credit-risk models behave under **temporal distribution shift**.

The current baseline trains a logistic regression model on earlier Lending Club loans and evaluates it on later years to measure how predictive performance changes over time.

## Current implemented baseline

The current codebase supports:

- loading and cleaning the curated Lending Club dataset
- selecting a structured feature set
- constructing a temporal train/test split
- fitting a baseline logistic regression model
- evaluating future-year performance using AUC and F1
- saving temporal performance plots and CSV summaries

## Repository structure

```text
CS-4365/
├── README.md
├── INSTRUCTIONS.md
├── requirements.txt
├── .gitignore
├── data/
├── experiments/
│   └── run_baseline_logreg.py
├── notebooks/
│   └── exploration.ipynb
├── results/
└── src/
    ├── load_data.py
    ├── preprocess.py
    ├── temporal_split.py
    ├── evaluate.py
    └── models/
        └── logistic.py