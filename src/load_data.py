from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


REQUIRED_COLUMNS = {
    "issue_d",
    "Default",
}


@dataclass(frozen=True)
class DatasetSchema:
    target_col: str = "Default"
    time_col: str = "issue_d"
    id_col: str = "id"
    text_cols: tuple[str, ...] = ("title", "desc")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load Lending Club dataset from CSV or Parquet.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Use CSV or Parquet."
        )

    return df


def validate_schema(
    df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
) -> None:
    """
    Check that the dataframe contains the minimum columns required
    for this project.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if schema.target_col not in df.columns:
        raise ValueError(f"Target column '{schema.target_col}' not found.")

    if schema.time_col not in df.columns:
        raise ValueError(f"Time column '{schema.time_col}' not found.")


def basic_cleaning(
    df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
) -> pd.DataFrame:
    """
    Perform project-level cleaning that is safe for all downstream tasks.

    This function:
    - parses issue date
    - creates year/month columns
    - standardizes text columns
    - enforces binary numeric target
    - drops rows unusable for temporal evaluation
    """
    df = df.copy()

    # Parse time column
    df[schema.time_col] = pd.to_datetime(df[schema.time_col], errors="coerce")
    df = df.dropna(subset=[schema.time_col])

    # Time-derived columns for drift analysis / splitting
    df["year"] = df[schema.time_col].dt.year
    df["month"] = df[schema.time_col].dt.month

    # Clean target
    # Zenodo should already be binary, but normalize just in case.
    target = df[schema.target_col]

    if not pd.api.types.is_numeric_dtype(target):
        target_map = {
            "Fully Paid": 0,
            "Default": 1,
            "Charged Off": 1,
        }
        df[schema.target_col] = target.astype(str).map(target_map)

    df = df.dropna(subset=[schema.target_col])
    df[schema.target_col] = df[schema.target_col].astype(int)

    invalid_target = ~df[schema.target_col].isin([0, 1])
    if invalid_target.any():
        bad_vals = sorted(df.loc[invalid_target, schema.target_col].unique().tolist())
        raise ValueError(f"Target column must be binary 0/1. Found: {bad_vals}")

    # Standardize text columns but keep them for later text-embedding experiments
    for col in schema.text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    return df


def load_and_clean_dataset(
    path: str | Path,
    schema: DatasetSchema = DatasetSchema(),
) -> pd.DataFrame:
    """
    Convenience wrapper: load raw data, validate schema, clean it.
    """
    df = load_dataset(path)
    validate_schema(df, schema)
    df = basic_cleaning(df, schema)
    return df


def get_structured_feature_columns(
    df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
    drop_cols: Optional[List[str]] = None,
    include_text: bool = False,
    include_id: bool = False,
) -> List[str]:
    """
    Return feature columns for structured modeling.

    By default:
    - excludes target, raw time columns, derived time columns
    - excludes text columns
    - excludes ID column
    """
    exclude = {
        schema.target_col,
        schema.time_col,
        "year",
        "month",
    }

    if not include_id and schema.id_col in df.columns:
        exclude.add(schema.id_col)

    if not include_text:
        for col in schema.text_cols:
            if col in df.columns:
                exclude.add(col)

    if drop_cols:
        exclude.update(drop_cols)

    return [col for col in df.columns if col not in exclude]


def get_numeric_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Return numeric feature columns from a provided column list.
    """
    return [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]


def get_categorical_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Return non-numeric feature columns from a provided column list.
    """
    return [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]


def get_text_columns(
    df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
) -> List[str]:
    """
    Return text columns present in the dataframe.
    """
    return [col for col in schema.text_cols if col in df.columns]


def summarize_dataset(
    df: pd.DataFrame,
    schema: DatasetSchema = DatasetSchema(),
) -> pd.DataFrame:
    """
    Small summary table useful for debugging and logging.
    """
    summary = {
        "n_rows": [len(df)],
        "n_cols": [df.shape[1]],
        "target_mean": [df[schema.target_col].mean()],
        "min_year": [df["year"].min() if "year" in df.columns else None],
        "max_year": [df["year"].max() if "year" in df.columns else None],
    }
    return pd.DataFrame(summary)
