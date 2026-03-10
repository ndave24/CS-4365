from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.load_data import DatasetSchema
from src.preprocess import build_tabular_preprocessor


@dataclass(frozen=True)
class LogisticConfig:
    max_iter: int = 1000
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    class_weight: Optional[str | dict] = "balanced"
    random_state: int = 42


def validate_feature_groups(feature_groups: Dict) -> None:
    """
    Validate that feature_groups contains the keys expected from preprocess.py.
    """
    required_keys = {"feature_cols", "numeric_cols", "categorical_cols"}
    missing = required_keys - set(feature_groups.keys())

    if missing:
        raise ValueError(
            f"feature_groups is missing required keys: {sorted(missing)}"
        )

    if len(feature_groups["feature_cols"]) == 0:
        raise ValueError("feature_groups['feature_cols'] is empty.")


def build_logistic_pipeline(
    feature_groups: Dict,
    config: LogisticConfig = LogisticConfig(),
) -> Pipeline:
    """
    Build a preprocessing + logistic regression pipeline.
    """
    validate_feature_groups(feature_groups)

    preprocessor = build_tabular_preprocessor(
        numeric_cols=feature_groups["numeric_cols"],
        categorical_cols=feature_groups["categorical_cols"],
        scale_numeric=True,
    )

    classifier = LogisticRegression(
        max_iter=config.max_iter,
        C=config.C,
        penalty=config.penalty,
        solver=config.solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    return pipeline


def fit_logistic_pipeline(
    train_df: pd.DataFrame,
    feature_groups: Dict,
    schema: DatasetSchema = DatasetSchema(),
    config: LogisticConfig = LogisticConfig(),
) -> Pipeline:
    """
    Fit the logistic regression pipeline on the training dataframe.
    """
    validate_feature_groups(feature_groups)

    if schema.target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{schema.target_col}' not found in training dataframe."
        )

    X_train = train_df[feature_groups["feature_cols"]].copy()
    y_train = train_df[schema.target_col].copy()

    pipeline = build_logistic_pipeline(feature_groups=feature_groups, config=config)
    pipeline.fit(X_train, y_train)

    return pipeline


def predict_default_probability(
    model: Pipeline,
    df: pd.DataFrame,
    feature_groups: Dict,
) -> np.ndarray:
    """
    Predict default probabilities for the positive class.
    """
    validate_feature_groups(feature_groups)

    X = df[feature_groups["feature_cols"]].copy()
    probs = model.predict_proba(X)[:, 1]

    return probs


def predict_default_label(
    model: Pipeline,
    df: pd.DataFrame,
    feature_groups: Dict,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict binary default labels using a probability threshold.
    """
    probs = predict_default_probability(
        model=model,
        df=df,
        feature_groups=feature_groups,
    )

    return (probs >= threshold).astype(int)


def get_logistic_model(model: Pipeline) -> LogisticRegression:
    """
    Extract the fitted LogisticRegression object from the pipeline.
    """
    if "model" not in model.named_steps:
        raise ValueError("Pipeline does not contain a 'model' step.")

    logistic_model = model.named_steps["model"]

    if not isinstance(logistic_model, LogisticRegression):
        raise ValueError("Pipeline 'model' step is not a LogisticRegression instance.")

    return logistic_model
