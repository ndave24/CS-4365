from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.load_data import (
    DatasetSchema,
    get_categorical_columns,
    get_numeric_columns,
    get_structured_feature_columns,
    get_text_columns,
)


@dataclass(frozen=True)
class PreprocessConfig:
    include_text: bool = False
    include_id: bool = False
    drop_zip_code: bool = True
    max_categorical_cardinality: int = 50
    extra_drop_cols: tuple[str, ...] = ()


def select_feature_columns(
    df: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
    schema: DatasetSchema = DatasetSchema(),
) -> List[str]:
    """
    Select feature columns for a given experiment configuration.
    """
    feature_cols = get_structured_feature_columns(
        df=df,
        schema=schema,
        include_text=config.include_text,
        include_id=config.include_id,
        drop_cols=list(config.extra_drop_cols),
    )

    if config.drop_zip_code and "zip_code" in feature_cols:
        feature_cols.remove("zip_code")

    return feature_cols


def filter_high_cardinality_categoricals(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_cardinality: int,
) -> List[str]:
    """
    Remove categorical columns with too many unique values.
    Useful for logistic regression / one-hot pipelines.
    """
    kept = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            kept.append(col)
        else:
            nunique = df[col].nunique(dropna=True)
            if nunique <= max_cardinality:
                kept.append(col)
    return kept


def get_model_feature_groups(
    df: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
    schema: DatasetSchema = DatasetSchema(),
) -> dict:
    """
    Return structured, numeric, categorical, and text feature groups.
    """
    feature_cols = select_feature_columns(df, config=config, schema=schema)
    feature_cols = filter_high_cardinality_categoricals(
        df=df,
        feature_cols=feature_cols,
        max_cardinality=config.max_categorical_cardinality,
    )

    numeric_cols = get_numeric_columns(df, feature_cols)
    categorical_cols = get_categorical_columns(df, feature_cols)
    text_cols = get_text_columns(df, schema) if config.include_text else []

    return {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "text_cols": text_cols,
    }


def build_tabular_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline for tabular models.
    """
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def summarize_feature_groups(feature_groups: dict) -> pd.DataFrame:
    """
    Small summary for debugging/logging.
    """
    return pd.DataFrame(
        {
            "group": ["all_features", "numeric", "categorical", "text"],
            "count": [
                len(feature_groups["feature_cols"]),
                len(feature_groups["numeric_cols"]),
                len(feature_groups["categorical_cols"]),
                len(feature_groups["text_cols"]),
            ],
        }
    )