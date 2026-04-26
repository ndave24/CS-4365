from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from tqdm.auto import tqdm

from src.llm_prep import LLMPrepConfig, build_llm_eval_dataframe
from src.thresholding import search_best_f1_threshold


CREDIT_RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "default_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Estimated probability that the borrower defaults or has a bad loan outcome.",
        },
        "predicted_label": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 means default/bad outcome; 0 means non-default/good outcome.",
        },
    },
    "required": ["default_probability", "predicted_label"],
    "additionalProperties": False,
}


def safe_auc(y_true, y_prob) -> float:
    y_true = pd.Series(y_true).astype(int)
    if y_true.nunique(dropna=True) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if mask.sum() == 0:
            continue

        bin_confidence = float(y_prob[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        bin_weight = float(mask.mean())
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def random_sample_by_year_excluding_ids(
    df: pd.DataFrame,
    years: Iterable[int],
    year_col: str = "year",
    id_col: str = "id",
    sample_per_year: int = 500,
    random_state: int = 123,
    exclude_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Randomly sample rows by year while excluding already-used loan IDs.
    This preserves the real class distribution instead of forcing 50/50 balance.
    """
    if id_col not in df.columns:
        raise ValueError(f"Expected id column '{id_col}' in dataframe.")

    exclude_ids = set(str(x) for x in (exclude_ids or []))
    sampled_parts = []

    for year in years:
        year_df = df[df[year_col] == year].copy()

        if exclude_ids:
            year_df = year_df[~year_df[id_col].astype(str).isin(exclude_ids)].copy()

        n = min(sample_per_year, len(year_df))

        if n < sample_per_year:
            print(
                f"Warning: requested {sample_per_year} rows for {year}, "
                f"but only {n} available after exclusions."
            )

        sampled = year_df.sample(
            n=n,
            random_state=random_state + int(year),
        ).copy()

        sampled_parts.append(sampled)

    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def build_llm_eval_input_batch(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    llm_prep_config: LLMPrepConfig,
    results_dir: str | Path,
    batch_num: int,
    val_year: int,
    test_years: Iterable[int],
    sample_per_year: int = 500,
    random_state: int = 123,
    exclude_ids: Optional[set[str]] = None,
    starting_row_id: int = 0,
    target_col: str = "Default",
    year_col: str = "year",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Build one LLM evaluation batch:
    - sample_per_year rows from validation year
    - sample_per_year rows from each future test year
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    val_source_df = random_sample_by_year_excluding_ids(
        df=val_df,
        years=(val_year,),
        year_col=year_col,
        id_col=id_col,
        sample_per_year=sample_per_year,
        random_state=random_state,
        exclude_ids=exclude_ids,
    )

    test_source_df = random_sample_by_year_excluding_ids(
        df=test_df,
        years=test_years,
        year_col=year_col,
        id_col=id_col,
        sample_per_year=sample_per_year,
        random_state=random_state,
        exclude_ids=exclude_ids,
    )

    val_prompt_df = build_llm_eval_dataframe(
        df=val_source_df,
        config=llm_prep_config,
        years=(val_year,),
        sample_per_year=None,
        random_state=random_state,
    )

    test_prompt_df = build_llm_eval_dataframe(
        df=test_source_df,
        config=llm_prep_config,
        years=test_years,
        sample_per_year=None,
        random_state=random_state,
    )

    val_prompt_df.insert(0, "split", "validation")
    test_prompt_df.insert(0, "split", "test")

    input_df = pd.concat([val_prompt_df, test_prompt_df], ignore_index=True).reset_index(drop=True)

    input_df.insert(
        0,
        "llm_row_id",
        range(starting_row_id, starting_row_id + len(input_df)),
    )

    input_df["llm_batch"] = batch_num

    output_path = results_dir / f"llm_temporal_eval_batch{batch_num}_input_template.csv"
    input_df.to_csv(output_path, index=False)

    print(f"Saved batch {batch_num} input template:", output_path)
    print(f"Batch {batch_num} shape:", input_df.shape)
    print("Duplicate loan IDs inside batch:", input_df[id_col].astype(str).duplicated().sum())

    return input_df


def call_openai_credit_risk(
    client,
    row: pd.Series,
    model: str = "gpt-4.1-nano",
    max_retries: int = 3,
    max_output_tokens: int = 80,
) -> dict:
    """
    Call an OpenAI model on one prompt row and return parseable prediction fields.
    """
    prompt = str(row["llm_prompt"])
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful credit-risk evaluator. "
                            "Return only the requested structured output."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "credit_risk_prediction",
                        "schema": CREDIT_RISK_SCHEMA,
                        "strict": True,
                    }
                },
                temperature=0,
                max_output_tokens=max_output_tokens,
            )

            raw_text = response.output_text
            parsed = json.loads(raw_text)

            p = float(parsed["default_probability"])
            p = max(0.0, min(1.0, p))
            raw_label = int(parsed["predicted_label"])

            return {
                "llm_model": model,
                "llm_default_probability": p,
                "llm_predicted_label_raw": raw_label,
                "llm_raw_response": raw_text,
                "llm_error": pd.NA,
            }

        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))

    return {
        "llm_model": model,
        "llm_default_probability": pd.NA,
        "llm_predicted_label_raw": pd.NA,
        "llm_raw_response": pd.NA,
        "llm_error": last_error,
    }


def run_llm_inference_with_resume(
    client,
    input_df: pd.DataFrame,
    output_path: str | Path,
    model: str = "gpt-4.1-nano",
    batch_name: str = "batch",
    sleep_seconds: float = 0.5,
    checkpoint_every: int = 25,
    retry_error_rows: bool = True,
) -> pd.DataFrame:
    """
    Run LLM inference with:
    - tqdm progress bar / ETA
    - checkpointing to CSV
    - resume support if the CSV already exists
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)

        if retry_error_rows and "llm_error" in existing_df.columns:
            completed_df = existing_df[
                existing_df["llm_error"].isna()
                & existing_df["llm_default_probability"].notna()
            ].copy()
            print(
                f"Resuming {batch_name}: found {len(existing_df)} saved rows, "
                f"keeping {len(completed_df)} completed rows and retrying errors/missing rows."
            )
        else:
            completed_df = existing_df.copy()
            print(f"Resuming {batch_name}: found {len(completed_df)} saved rows.")

        rows = completed_df.to_dict("records")
        completed_ids = set(completed_df["llm_row_id"].astype(int))
    else:
        rows = []
        completed_ids = set()
        print(f"Starting {batch_name} from scratch.")

    pbar = tqdm(
        total=len(input_df),
        initial=len(completed_ids),
        desc=f"LLM inference {batch_name}",
    )

    for _, row in input_df.iterrows():
        row_id = int(row["llm_row_id"])

        if row_id in completed_ids:
            continue

        result = call_openai_credit_risk(
            client=client,
            row=row,
            model=model,
        )

        record = row.to_dict()
        record.update(result)
        rows.append(record)
        completed_ids.add(row_id)

        current_errors = sum(pd.notna(r.get("llm_error", pd.NA)) for r in rows)

        pbar.update(1)
        pbar.set_postfix(errors=current_errors, last_id=row_id)

        if len(completed_ids) % checkpoint_every == 0:
            temp_df = (
                pd.DataFrame(rows)
                .drop_duplicates(subset=["llm_row_id"], keep="last")
                .sort_values("llm_row_id")
            )
            temp_df.to_csv(output_path, index=False)

        time.sleep(sleep_seconds)

    pbar.close()

    output_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["llm_row_id"], keep="last")
        .sort_values("llm_row_id")
        .reset_index(drop=True)
    )

    output_df.to_csv(output_path, index=False)

    print(f"Done {batch_name}.")
    print("Saved:", output_path)
    print("Rows:", len(output_df))
    print("Errors:", output_df["llm_error"].notna().sum())

    return output_df


def evaluate_llm_temporal_predictions(
    predictions_df: pd.DataFrame,
    results_dir: str | Path,
    output_prefix: str = "llm_temporal_eval",
    model_name: Optional[str] = None,
    target_col: str = "Default",
    year_col: str = "year",
    split_col: str = "split",
    prob_col: str = "llm_default_probability",
    validation_split: str = "validation",
    test_split: str = "test",
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tune LLM threshold on validation rows and evaluate by future test year.
    Saves:
    - {output_prefix}_threshold_search.csv
    - {output_prefix}_metrics.csv
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = predictions_df.dropna(subset=[prob_col, target_col]).copy()
    df[prob_col] = df[prob_col].astype(float)
    df[target_col] = df[target_col].astype(int)

    val_pred_df = df[df[split_col] == validation_split].copy()
    test_pred_df = df[df[split_col] == test_split].copy()

    if len(val_pred_df) == 0:
        raise ValueError("No validation rows found for LLM threshold tuning.")
    if val_pred_df[target_col].nunique() < 2:
        print("Warning: validation split contains only one class; threshold tuning may be unstable.")

    best_row, threshold_df = search_best_f1_threshold(
        y_true=val_pred_df[target_col].to_numpy(),
        y_prob=val_pred_df[prob_col].to_numpy(),
    )

    best_threshold = float(best_row["threshold"])

    metric_rows = []
    for year, year_df in test_pred_df.groupby(year_col):
        y_true = year_df[target_col].to_numpy().astype(int)
        y_prob = year_df[prob_col].to_numpy().astype(float)
        y_pred = (y_prob >= best_threshold).astype(int)

        metric_rows.append(
            {
                "model": model_name or str(year_df["llm_model"].dropna().iloc[0]),
                "year": int(year),
                "n_rows": int(len(year_df)),
                "threshold": best_threshold,
                "auc": safe_auc(y_true, y_prob),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "brier_score": float(brier_score_loss(y_true, y_prob)),
                "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
                "default_rate": float(np.mean(y_true)),
                "mean_pred": float(np.mean(y_prob)),
                "pred_positive_rate": float(np.mean(y_pred)),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("year").reset_index(drop=True)

    threshold_df.to_csv(results_dir / f"{output_prefix}_threshold_search.csv", index=False)
    metrics_df.to_csv(results_dir / f"{output_prefix}_metrics.csv", index=False)

    return metrics_df, threshold_df


def combine_llm_prediction_files(
    prediction_paths: list[str | Path],
    output_path: str | Path,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Combine one or more LLM prediction CSVs and remove duplicate loan IDs.
    """
    frames = [pd.read_csv(path) for path in prediction_paths]
    combined_df = pd.concat(frames, ignore_index=True)

    combined_df = (
        combined_df
        .drop_duplicates(subset=[id_col], keep="first")
        .sort_values("llm_row_id")
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    return combined_df