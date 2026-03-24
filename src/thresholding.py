from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import json

import pandas as pd

from src.load_data import DatasetSchema


@dataclass(frozen=True)
class LLMPrepConfig:
    """
    Configuration for preparing future LLM-based evaluation data.
    """

    text_cols: Sequence[str] = ("title_clean", "desc_clean")
    structured_context_cols: Sequence[str] = (
        "loan_amnt",
        "term",
        "int_rate",
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "purpose",
        "home_ownership",
        "emp_length",
    )
    id_col: str = "id"
    year_col: str = "year"
    max_text_chars_per_field: int = 600
    include_structured_context: bool = True
    include_target: bool = True


def _existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """
    Return only the columns from cols that actually exist in df.
    """
    return [col for col in cols if col in df.columns]


def _clean_text(value) -> str:
    """
    Convert a value to a stripped string, returning an empty string for missing values.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to max_chars with an ellipsis if needed.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_text_block(
    row: pd.Series,
    text_cols: Sequence[str],
    max_text_chars_per_field: int,
) -> str:
    """
    Build a multi-line text block from available text columns in a row.
    """
    parts: List[str] = []

    for col in text_cols:
        value = _clean_text(row.get(col, ""))
        if value:
            value = _truncate_text(value, max_text_chars_per_field)
            parts.append(f"{col}: {value}")

    return "\n".join(parts).strip()


def build_structured_context_block(
    row: pd.Series,
    structured_cols: Sequence[str],
) -> str:
    """
    Build a multi-line structured context block from selected structured columns.
    """
    parts: List[str] = []

    for col in structured_cols:
        value = row.get(col, None)
        if pd.isna(value):
            continue
        parts.append(f"{col}: {value}")

    return "\n".join(parts).strip()


def make_default_risk_prompt(row: pd.Series) -> str:
    """
    Build a future-facing prompt for an LLM-based default-risk estimate.
    This function only prepares the prompt; it does not call an LLM.
    """
    structured_context = row.get("structured_context", "")
    text_block = row.get("text_block", "")

    prompt = f"""You are evaluating the risk that a consumer loan will default.

Use the borrower-provided text and optional structured loan context below.
Estimate the probability of default as a number between 0 and 1.
Then briefly explain the reasoning using only the provided information.

Structured context:
{structured_context if structured_context else "(none provided)"}

Borrower text:
{text_block if text_block else "(no text provided)"}

Respond in JSON with keys:
- default_probability
- rationale
"""
    return prompt.strip()


def build_llm_eval_dataframe(
    df: pd.DataFrame,
    config: LLMPrepConfig = LLMPrepConfig(),
    schema: DatasetSchema = DatasetSchema(),
    years: Optional[Iterable[int]] = None,
    sample_per_year: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a clean dataframe for future LLM-based evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    config : LLMPrepConfig
        Configuration for text/context preparation.
    schema : DatasetSchema
        Dataset schema used to locate the target column.
    years : Optional[Iterable[int]]
        If provided, filter rows to these years.
    sample_per_year : Optional[int]
        If provided, sample up to this many rows per year.
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        A dataframe containing prompt-ready fields for future LLM evaluation.
    """
    work_df = df.copy()

    if years is not None:
        years = list(years)
        if config.year_col not in work_df.columns:
            raise ValueError(f"'{config.year_col}' column not found in dataframe.")
        work_df = work_df[work_df[config.year_col].isin(years)].copy()

    if sample_per_year is not None:
        if config.year_col not in work_df.columns:
            raise ValueError(f"'{config.year_col}' column not found in dataframe.")

        sampled_parts = []
        for _, year_df in work_df.groupby(config.year_col):
            n = min(sample_per_year, len(year_df))
            sampled_parts.append(year_df.sample(n=n, random_state=random_state))
        work_df = pd.concat(sampled_parts, ignore_index=True)

    text_cols = _existing_cols(work_df, config.text_cols)
    structured_cols = _existing_cols(work_df, config.structured_context_cols)

    if len(text_cols) == 0:
        raise ValueError(
            "No configured text columns were found in the dataframe. "
            f"Configured text_cols={list(config.text_cols)}"
        )

    work_df = work_df.copy()

    work_df["text_block"] = work_df.apply(
        lambda row: build_text_block(
            row=row,
            text_cols=text_cols,
            max_text_chars_per_field=config.max_text_chars_per_field,
        ),
        axis=1,
    )

    if config.include_structured_context:
        work_df["structured_context"] = work_df.apply(
            lambda row: build_structured_context_block(
                row=row,
                structured_cols=structured_cols,
            ),
            axis=1,
        )
    else:
        work_df["structured_context"] = ""

    work_df["llm_prompt"] = work_df.apply(make_default_risk_prompt, axis=1)

    output_cols: List[str] = []

    if config.id_col in work_df.columns:
        output_cols.append(config.id_col)

    if config.year_col in work_df.columns:
        output_cols.append(config.year_col)

    output_cols.extend(["text_block", "structured_context", "llm_prompt"])

    if config.include_target and schema.target_col in work_df.columns:
        output_cols.append(schema.target_col)

    return work_df[output_cols].reset_index(drop=True)


def export_llm_eval_csv(
    llm_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save the LLM evaluation dataframe as a CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llm_df.to_csv(output_path, index=False)
    return output_path


def export_llm_eval_jsonl(
    llm_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save the LLM evaluation dataframe as a JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in llm_df.iterrows():
            record = row.to_dict()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path