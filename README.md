# Credit Risk Temporal Drift in Lending Club Data

This project studies how credit-risk models degrade under **temporal distribution shift**.  
Models are trained on earlier Lending Club loans and evaluated on later years to measure how predictive performance changes over time.

## Project Goal

The main question is:

> If a credit-risk model is trained on historical loan applications, how well does it generalize to future borrowers as the temporal gap increases?

For the current baseline, the project:
- loads and cleans the curated Lending Club dataset
- constructs strict temporal train/test splits
- trains a baseline logistic regression model
- evaluates performance on future years using AUC and F1
- saves temporal performance plots and metrics

## Repository Structure

```text
CS-4365/
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
├── experiments/
│   └── run_baseline_logreg.py
├── notebooks/
│   └── exploration.ipynb
├── results/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── load_data.py
    ├── preprocess.py
    ├── temporal_split.py
    ├── evaluate.py
    └── models/
        ├── __init__.py
        └── logistic.py
