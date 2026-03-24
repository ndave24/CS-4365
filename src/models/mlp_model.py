from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.load_data import DatasetSchema
from src.preprocess import build_tabular_preprocessor


@dataclass(frozen=True)
class MLPConfig:
    svd_components: int = 64
    hidden_layer_sizes: Tuple[int, int] = (32, 16)
    activation: str = "relu"
    alpha: float = 0.0001
    batch_size: int = 1024
    learning_rate_init: float = 0.001
    max_iter: int = 20
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 3
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


def build_mlp_pipeline(
    feature_groups: Dict,
    config: MLPConfig = MLPConfig(),
) -> Pipeline:
    """
    Build a preprocessing + dimensionality reduction + MLP pipeline.
    """
    validate_feature_groups(feature_groups)

    preprocessor = build_tabular_preprocessor(
        numeric_cols=feature_groups["numeric_cols"],
        categorical_cols=feature_groups["categorical_cols"],
        scale_numeric=True,
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        alpha=config.alpha,
        batch_size=config.batch_size,
        learning_rate_init=config.learning_rate_init,
        max_iter=config.max_iter,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction,
        n_iter_no_change=config.n_iter_no_change,
        random_state=config.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("svd", TruncatedSVD(
                n_components=config.svd_components,
                random_state=config.random_state,
            )),
            ("scale_after_svd", StandardScaler()),
            ("model", classifier),
        ]
    )

    return pipeline


def fit_mlp_pipeline(
    train_df: pd.DataFrame,
    feature_groups: Dict,
    schema: DatasetSchema = DatasetSchema(),
    config: MLPConfig = MLPConfig(),
) -> Pipeline:
    """
    Fit the MLP pipeline on the training dataframe.
    """
    validate_feature_groups(feature_groups)

    if schema.target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{schema.target_col}' not found in training dataframe."
        )

    X_train = train_df[feature_groups["feature_cols"]].copy()
    y_train = train_df[schema.target_col].copy().astype(int)

    pipeline = build_mlp_pipeline(
        feature_groups=feature_groups,
        config=config,
    )
    pipeline.fit(X_train, y_train)

    return pipeline