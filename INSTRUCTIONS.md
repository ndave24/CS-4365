# INSTRUCTIONS.md

## Goal of this repository

This repository contains the current baseline implementation for a project on **temporal drift in credit-risk modeling** using the curated Lending Club dataset.

The current implemented goal is to:

1. load and clean the dataset
2. define the modeling feature set
3. create a strict temporal split
4. train a baseline logistic regression model
5. evaluate future-year performance using AUC and F1
6. save the resulting CSV summaries and plots

For the current checkpoint, the intended deliverable is a **working baseline model plus an initial temporal degradation curve**.

---

## Repository structure

```text
CS-4365/
├── README.md
├── INSTRUCTIONS.md
├── requirements.txt
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
```

---

## What each file does

### `src/load_data.py`
Loads the raw dataset from CSV or Parquet, validates required columns, parses dates, standardizes text fields, and creates time-derived columns such as `year`.

### `src/preprocess.py`
Chooses which columns are used as model features and constructs the preprocessing logic for tabular models.

### `src/temporal_split.py`
Creates the train/test split by year.

### `src/models/logistic.py`
Builds and fits the baseline logistic regression pipeline.

### `src/evaluate.py`
Evaluates the fitted model year-by-year and generates temporal plots.

### `experiments/run_baseline_logreg.py`
This is the canonical script for reproducing the checkpoint baseline result.

### `notebooks/exploration.ipynb`
This is the interactive notebook version of the same workflow. It calls the reusable code in `src/` rather than duplicating the project logic.

---

## Dataset required to run this repository

This repository does **not** include the dataset file itself.

To reproduce the results, you must first obtain the **curated Lending Club dataset from Zenodo record 11295916**. This is the cleaned Lending Club granting-model dataset used for the project.

The dataset should then be stored in one of these ways:

### Option A: Google Drive (recommended for Colab)
Store it at a path such as:

```text
/content/drive/MyDrive/datasets/lending_club.parquet
```

or

```text
/content/drive/MyDrive/datasets/lending_club.csv
```

### Option B: Local repository data folder
Store it at:

```text
data/lending_club.parquet
```

or

```text
data/lending_club.csv
```

Parquet is preferred because it loads much faster than CSV.

---

## Exact reproduction workflow in Google Colab

This is the intended workflow for reproducing the current project result.

### Step 1: Open Google Colab

Open a new Colab notebook, or upload `notebooks/exploration.ipynb`.

### Step 2: Clone the repository

Run:

```python
!git clone https://github.com/ndave24/CS-4365.git
%cd CS-4365
```

### Step 3: Install dependencies

Run:

```python
!pip install -r requirements.txt
```

### Step 4: Mount Google Drive

Run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Step 5: Make sure the dataset is already in Google Drive

Before running the experiment, the user must have already uploaded the curated Lending Club dataset to Google Drive.

Recommended location:

```text
/content/drive/MyDrive/datasets/lending_club.parquet
```

If only a CSV is available, it can also be used directly.

### Step 6: Open the notebook

Open:

```text
notebooks/exploration.ipynb
```

Inside the notebook, set:

```python
USE_GOOGLE_DRIVE = True
```

and set the dataset path to something like:

```python
DATA_PATH = Path("/content/drive/MyDrive/datasets/lending_club.parquet")
```

If needed, also set the repo root manually:

```python
REPO_ROOT = Path("/content/CS-4365")
```

Then run the notebook cells from top to bottom.

### Step 7: Generated outputs

A successful run should generate outputs in `results/`, including:

- `temporal_metrics.csv`
- `split_summary.csv`
- `test_year_summary.csv`
- `temporal_auc_f1.png`
- `temporal_auc_f1_time_gap.png`

These are the main artifacts used to document the current checkpoint result.

---

## Exact command-line reproduction workflow

From the repository root, after the dataset has been placed locally in `data/`, run:

```bash
python experiments/run_baseline_logreg.py --data-path data/lending_club.parquet --train-end-year 2014 --test-years 2015 2016 2017 2018
```

If using CSV instead:

```bash
python experiments/run_baseline_logreg.py --data-path data/lending_club.csv --train-end-year 2014 --test-years 2015 2016 2017 2018
```

This script should load the dataset, construct the temporal split, fit the logistic regression baseline, evaluate each test year, and write the output files to `results/`.

---

## Exact dataset loading step

### In the notebook

The notebook should contain a cell like:

```python
from pathlib import Path

USE_GOOGLE_DRIVE = True
REPO_ROOT = Path("/content/CS-4365")
DATA_PATH = Path("/content/drive/MyDrive/datasets/lending_club.parquet")
```

Then later:

```python
df = load_and_clean_dataset(DATA_PATH)
```

### In the script

The experiment script receives the dataset path as an argument:

```bash
python experiments/run_baseline_logreg.py --data-path data/lending_club.parquet --train-end-year 2014 --test-years 2015 2016 2017 2018
```

Internally, it calls the loading function from `src/load_data.py`.

---

## What an LLM should do first when reading this repository

An LLM trying to understand or reproduce the repository should follow this order:

1. read `README.md` for a short overview
2. read `INSTRUCTIONS.md` for the exact execution workflow
3. inspect `experiments/run_baseline_logreg.py` to see the canonical end-to-end pipeline
4. inspect the reusable implementation files in `src/` in this order:
   - `load_data.py`
   - `preprocess.py`
   - `temporal_split.py`
   - `models/logistic.py`
   - `evaluate.py`

The current main execution paths are:

- interactive path: `notebooks/exploration.ipynb`
- canonical reproducible path: `experiments/run_baseline_logreg.py`

---

## Expected current scope

At the moment, the implemented baseline is **structured-feature logistic regression only**.

The codebase is intentionally organized so that future extensions can add:

- XGBoost
- MLP
- calibration metrics
- drift metrics such as PSI
- text embedding experiments using `title` and `desc`

Those future additions should reuse the same dataset loading, preprocessing, splitting, and evaluation structure already implemented here.