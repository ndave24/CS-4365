from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.load_data import DatasetSchema


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_end_year: int
    val_years: Optional[tuple[int, ...]] = None
    test_years: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class TemporalTrainValTestSplitConfig:
    train_end_year: int
    val_year: int
    test_years: Tuple[int, ...]


def validate_temporal_split_inputs(
    df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
) -> None:
    """
    Validate that the dataframe contains the columns needed
    for temporal splitting.
    """
    required_cols = {schema.target_col, schema.time_col, "year"}
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataframe is missing required columns for temporal splitting: {missing}"
        )

    if df["year"].isna().any():
        raise ValueError("Column 'year' contains missing values.")

    if not pd.api.types.is_integer_dtype(df["year"]):
        try:
            df["year"] = df["year"].astype(int)
        except Exception as exc:
            raise ValueError("Column 'year' must be integer-like.") from exc


def make_temporal_split(
    df: pd.DataFrame,
    config: TemporalSplitConfig,
    schema: DatasetSchema = DatasetSchema(),
) -> Dict[str, pd.DataFrame]:
    """
    Create a temporal split of the dataframe.

    Behavior:
    - train uses all rows with year <= train_end_year
    - validation uses explicitly provided val_years, if any
    - test uses explicitly provided test_years if given,
      otherwise all years > train_end_year not used by validation
    """
    validate_temporal_split_inputs(df, schema=schema)

    df = df.copy()

    val_years = tuple(sorted(config.val_years or ()))
    used_future_years = set(val_years)

    train_df = df[df["year"] <= config.train_end_year].copy()

    if val_years:
        val_df = df[df["year"].isin(val_years)].copy()
    else:
        val_df = df.iloc[0:0].copy()

    if config.test_years is not None:
        test_years = tuple(sorted(config.test_years))
    else:
        test_years = tuple(
            sorted(
                year
                for year in df["year"].unique().tolist()
                if year > config.train_end_year and year not in used_future_years
            )
        )

    test_df = df[df["year"].isin(test_years)].copy()

    if len(train_df) == 0:
        raise ValueError("Temporal split produced an empty training set.")

    if len(test_df) == 0:
        raise ValueError("Temporal split produced an empty test set.")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def make_temporal_train_val_test_split(
    df: pd.DataFrame,
    config: TemporalTrainValTestSplitConfig,
    schema: DatasetSchema = DatasetSchema(),
) -> Dict[str, pd.DataFrame]:
    """
    Create a temporal train/validation/test split.

    Example:
    - train_end_year=2013
    - val_year=2014
    - test_years=(2015, 2016, 2017, 2018)

    gives:
    - train_df: years <= 2013
    - val_df: year == 2014
    - test_df: years in test_years
    """
    validate_temporal_split_inputs(df, schema=schema)

    df = df.copy()

    train_df = df[df["year"] <= config.train_end_year].copy()
    val_df = df[df["year"] == config.val_year].copy()
    test_df = df[df["year"].isin(config.test_years)].copy()

    if len(train_df) == 0:
        raise ValueError("Temporal split produced an empty train_df.")
    if len(val_df) == 0:
        raise ValueError("Temporal split produced an empty val_df.")
    if len(test_df) == 0:
        raise ValueError("Temporal split produced an empty test_df.")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def get_test_subsets_by_year(test_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Return a dictionary mapping test year -> that year's dataframe.
    """
    if "year" not in test_df.columns:
        raise ValueError("test_df must contain a 'year' column.")

    year_to_df: Dict[int, pd.DataFrame] = {}
    for year in sorted(test_df["year"].unique().tolist()):
        year_to_df[int(year)] = test_df[test_df["year"] == year].copy()

    return year_to_df


def split_features_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    schema: DatasetSchema = DatasetSchema(),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into X and y using the provided feature columns.
    """
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if schema.target_col not in df.columns:
        raise ValueError(f"Target column '{schema.target_col}' not found in dataframe.")

    X = df[feature_cols].copy()
    y = df[schema.target_col].copy()

    return X, y


def describe_split(
    split_dict: Dict[str, pd.DataFrame],
    schema: DatasetSchema = DatasetSchema(),
) -> pd.DataFrame:
    """
    Create a compact summary of the temporal split.
    """
    rows = []

    for split_name in ["train_df", "val_df", "test_df"]:
        split_df = split_dict.get(split_name)

        if split_df is None or len(split_df) == 0:
            rows.append(
                {
                    "split": split_name,
                    "n_rows": 0,
                    "min_year": None,
                    "max_year": None,
                    "default_rate": None,
                }
            )
            continue

        rows.append(
            {
                "split": split_name,
                "n_rows": len(split_df),
                "min_year": int(split_df["year"].min()),
                "max_year": int(split_df["year"].max()),
                "default_rate": float(split_df[schema.target_col].mean()),
            }
        )

    return pd.DataFrame(rows)


def describe_test_years(
    test_df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
) -> pd.DataFrame:
    """
    Summarize the test set year-by-year.
    """
    if len(test_df) == 0:
        return pd.DataFrame(columns=["year", "n_rows", "default_rate"])

    rows = []
    for year, year_df in get_test_subsets_by_year(test_df).items():
        rows.append(
            {
                "year": year,
                "n_rows": len(year_df),
                "default_rate": float(year_df[schema.target_col].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)