from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.load_data import DatasetSchema
from src.temporal_split import get_test_subsets_by_year


def safe_roc_auc_score(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """
    Compute ROC AUC safely.
    Returns np.nan if only one class is present.
    """
    y_true = pd.Series(y_true)

    if y_true.nunique(dropna=True) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_prob))

def _validate_feature_groups_for_eval(feature_groups: Dict) -> None:
    """
    Validate that feature_groups contains the keys needed for evaluation.
    """
    if "feature_cols" not in feature_groups:
        raise ValueError("feature_groups must contain a 'feature_cols' key.")

    if len(feature_groups["feature_cols"]) == 0:
        raise ValueError("feature_groups['feature_cols'] is empty.")

def predict_positive_probability(
    model,
    df: pd.DataFrame,
    feature_groups: Dict,
) -> np.ndarray:
    """
    Predict positive-class probabilities for any fitted model/pipeline
    that implements predict_proba(X).
    """
    _validate_feature_groups_for_eval(feature_groups)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict_proba(X).")

    X = df[feature_groups["feature_cols"]].copy()
    probs = np.asarray(model.predict_proba(X))

    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(
            "model.predict_proba(X) must return shape (n_samples, 2+) "
            "for binary classification."
        )

    return probs[:, 1]


def compute_binary_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binary classification metrics from probabilities.
    """
    y_true = pd.Series(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    auc = safe_roc_auc_score(y_true, y_prob)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "auc": auc,
        "f1": f1,
    }


def evaluate_dataframe(
    model,
    df: pd.DataFrame,
    feature_groups: Dict,
    schema: DatasetSchema = DatasetSchema(),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a fitted model on a single dataframe.
    """
    if schema.target_col not in df.columns:
        raise ValueError(
            f"Target column '{schema.target_col}' not found in evaluation dataframe."
        )

    if len(df) == 0:
        raise ValueError("Cannot evaluate on an empty dataframe.")

    y_true = df[schema.target_col].copy().astype(int)
    y_prob = predict_positive_probability(
        model=model,
        df=df,
        feature_groups=feature_groups,
    )
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )

    return {
        "n_rows": int(len(df)),
        "default_rate": float(y_true.mean()),
        "pred_positive_rate": float(np.mean(y_pred)),
        "auc": metrics["auc"],
        "f1": metrics["f1"],
    }


def evaluate_temporal_by_year(
    model,
    test_df: pd.DataFrame,
    feature_groups: Dict,
    schema: DatasetSchema = DatasetSchema(),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate a fitted model on each test year separately.
    """
    if "year" not in test_df.columns:
        raise ValueError("test_df must contain a 'year' column.")

    year_subsets = get_test_subsets_by_year(test_df)

    rows = []
    for year, year_df in year_subsets.items():
        metrics = evaluate_dataframe(
            model=model,
            df=year_df,
            feature_groups=feature_groups,
            schema=schema,
            threshold=threshold,
        )

        rows.append(
            {
                "year": int(year),
                "n_rows": metrics["n_rows"],
                "default_rate": metrics["default_rate"],
                "pred_positive_rate": metrics["pred_positive_rate"],
                "auc": metrics["auc"],
                "f1": metrics["f1"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return results_df


def add_time_gap_column(
    results_df: pd.DataFrame,
    train_end_year: int,
) -> pd.DataFrame:
    """
    Add a time-gap column relative to the training cutoff year.
    """
    if "year" not in results_df.columns:
        raise ValueError("results_df must contain a 'year' column.")

    results_df = results_df.copy()
    results_df["time_gap"] = results_df["year"] - train_end_year
    return results_df


def save_temporal_metrics(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save temporal metrics dataframe to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)
    return output_path


def plot_temporal_metrics(
    results_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    use_time_gap: bool = False,
    title: str = "Temporal Performance",
) -> None:
    """
    Plot AUC and F1 against year or time gap.
    """
    required_cols = {"auc", "f1"}
    missing = [col for col in required_cols if col not in results_df.columns]
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    x_col = "time_gap" if use_time_gap else "year"
    if x_col not in results_df.columns:
        raise ValueError(f"results_df must contain '{x_col}' to plot with it.")

    plot_df = results_df.sort_values(x_col).reset_index(drop=True)

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df[x_col], plot_df["auc"], marker="o", label="AUC")
    plt.plot(plot_df[x_col], plot_df["f1"], marker="o", label="F1")
    plt.xlabel("Time Gap" if use_time_gap else "Test Year")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()


def summarize_temporal_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact summary table for logging/reporting.
    """
    required_cols = {"year", "n_rows", "default_rate", "auc", "f1"}
    missing = [col for col in required_cols if col not in results_df.columns]
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    summary_df = results_df.copy()
    return summary_df[
        ["year", "n_rows", "default_rate", "auc", "f1"]
    ].sort_values("year").reset_index(drop=True)