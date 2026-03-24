from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def predict_positive_probability(
    model,
    df: pd.DataFrame,
    feature_groups: Dict,
) -> np.ndarray:
    """
    Predict positive-class probabilities for any fitted model/pipeline
    that implements predict_proba(X).
    """
    if "feature_cols" not in feature_groups:
        raise ValueError("feature_groups must contain a 'feature_cols' key.")

    X = df[feature_groups["feature_cols"]].copy()
    probs = np.asarray(model.predict_proba(X))

    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(
            "model.predict_proba(X) must return shape (n_samples, 2+) "
            "for binary classification."
        )

    return probs[:, 1]


def make_threshold_grid(
    y_prob: np.ndarray,
    min_threshold: float = 0.001,
    max_threshold: float = 0.50,
    n_fixed: int = 250,
    n_quantiles: int = 250,
) -> np.ndarray:
    """
    Build a threshold grid using both a fixed grid and probability quantiles.
    """
    fixed_grid = np.linspace(min_threshold, max_threshold, n_fixed)
    quantile_grid = np.quantile(y_prob, np.linspace(0.0, 1.0, n_quantiles))

    thresholds = np.unique(
        np.clip(np.concatenate([fixed_grid, quantile_grid]), 1e-6, 1 - 1e-6)
    )
    return thresholds


def search_best_f1_threshold(
    y_true,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Search thresholds and return the best one by F1 score.
    """
    y_true = pd.Series(y_true).astype(int).to_numpy()

    if thresholds is None:
        thresholds = make_threshold_grid(y_prob)

    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "pred_positive_rate": float(y_pred.mean()),
                "pred_positives": int(y_pred.sum()),
            }
        )

    threshold_df = (
        pd.DataFrame(rows)
        .sort_values(["f1", "threshold"], ascending=[False, True])
        .reset_index(drop=True)
    )

    best_row = threshold_df.iloc[0].to_dict()
    return best_row, threshold_df


def tune_threshold_on_validation(
    model,
    val_df: pd.DataFrame,
    feature_groups: Dict,
    target_col: str = "Default",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Tune a classification threshold on a validation dataframe.
    """
    if target_col not in val_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in val_df.")

    y_true = val_df[target_col].copy().astype(int)
    y_prob = predict_positive_probability(
        model=model,
        df=val_df,
        feature_groups=feature_groups,
    )

    best_row, threshold_df = search_best_f1_threshold(
        y_true=y_true,
        y_prob=y_prob,
    )
    return best_row, threshold_df