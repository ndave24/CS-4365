from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.load_data import DatasetSchema
from src.preprocess import build_tabular_preprocessor


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    eval_metric: str = "logloss"


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


def build_xgboost_pipeline(
    feature_groups: Dict,
    config: XGBoostConfig = XGBoostConfig(),
) -> Pipeline:
    """
    Build a preprocessing + XGBoost pipeline.
    """
    validate_feature_groups(feature_groups)

    preprocessor = build_tabular_preprocessor(
        numeric_cols=feature_groups["numeric_cols"],
        categorical_cols=feature_groups["categorical_cols"],
        scale_numeric=False,
    )

    classifier = XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        tree_method=config.tree_method,
        eval_metric=config.eval_metric,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    return pipeline


def fit_xgboost_pipeline(
    train_df: pd.DataFrame,
    feature_groups: Dict,
    schema: DatasetSchema = DatasetSchema(),
    config: XGBoostConfig = XGBoostConfig(),
) -> Pipeline:
    """
    Fit the XGBoost pipeline on the training dataframe.
    """
    validate_feature_groups(feature_groups)

    if schema.target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{schema.target_col}' not found in training dataframe."
        )

    X_train = train_df[feature_groups["feature_cols"]].copy()
    y_train = train_df[schema.target_col].copy().astype(int)

    pipeline = build_xgboost_pipeline(
        feature_groups=feature_groups,
        config=config,
    )
    pipeline.fit(X_train, y_train)

    return pipeline