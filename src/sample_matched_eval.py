from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def safe_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
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

        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_weight = mask.mean()
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def tune_threshold_for_f1(y_true, y_prob) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        ["f1", "threshold"],
        ascending=[False, True],
    ).iloc[0]

    return float(best_row["threshold"]), threshold_df


def build_original_sample_frame(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    llm_predictions_df: pd.DataFrame,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Recover the original feature rows used in the LLM sample by joining on loan ID.
    """
    original_df = pd.concat([val_df, test_df], ignore_index=True).copy()

    sample_keys = llm_predictions_df[
        [
            id_col,
            "llm_row_id",
            "llm_batch",
            "split",
            "llm_model",
            "llm_default_probability",
            "llm_predicted_label_raw",
            "llm_error",
        ]
    ].copy()

    sample_keys[id_col] = sample_keys[id_col].astype(str)
    original_df[id_col] = original_df[id_col].astype(str)

    sample_df = sample_keys.merge(
        original_df,
        on=id_col,
        how="left",
        validate="many_to_one",
    )

    missing_original_rows = sample_df["year"].isna().sum()
    if missing_original_rows > 0:
        raise ValueError(f"{missing_original_rows} LLM sample rows could not be matched back to original data.")

    return sample_df


def predict_structured_models_on_sample(
    sample_df: pd.DataFrame,
    models: Dict[str, object],
    feature_cols: list[str],
    target_col: str = "Default",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Generate long-format prediction rows for structured models.
    """
    rows = []

    X = sample_df[feature_cols]
    y = sample_df[target_col].astype(int).to_numpy()

    base_cols = sample_df[["id", "llm_row_id", "llm_batch", "split", year_col, target_col]].copy()

    for model_name, model in models.items():
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Model {model_name} does not expose predict_proba.")

        y_prob = model.predict_proba(X)[:, 1]

        model_df = base_cols.copy()
        model_df["model"] = model_name
        model_df["y_true"] = y
        model_df["y_prob"] = y_prob
        rows.append(model_df)

    return pd.concat(rows, ignore_index=True)


def build_llm_prediction_frame(
    llm_predictions_df: pd.DataFrame,
    target_col: str = "Default",
    year_col: str = "year",
    prob_col: str = "llm_default_probability",
    model_name: str | None = None,
) -> pd.DataFrame:
    """
    Convert LLM predictions into the same long format as structured model predictions.
    """
    df = llm_predictions_df.dropna(subset=[prob_col, target_col]).copy()

    if model_name is None:
        non_null_models = df["llm_model"].dropna().unique()
        model_name = str(non_null_models[0]) if len(non_null_models) else "llm"

    out = df[["id", "llm_row_id", "llm_batch", "split", year_col, target_col]].copy()
    out["model"] = model_name
    out["y_true"] = df[target_col].astype(int)
    out["y_prob"] = df[prob_col].astype(float)

    return out


def evaluate_sample_matched_predictions(
    prediction_df: pd.DataFrame,
    results_dir: str | Path,
    output_prefix: str = "sample_matched",
    target_col: str = "y_true",
    prob_col: str = "y_prob",
    split_col: str = "split",
    year_col: str = "year",
    validation_split: str = "validation",
    test_split: str = "test",
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tune one threshold per model on the validation sample and evaluate on future test years.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    threshold_rows = []

    for model_name, model_df in prediction_df.groupby("model"):
        val_df = model_df[model_df[split_col] == validation_split].copy()
        test_df = model_df[model_df[split_col] == test_split].copy()

        if len(val_df) == 0:
            raise ValueError(f"No validation rows found for model {model_name}.")

        best_threshold, threshold_df = tune_threshold_for_f1(
            y_true=val_df[target_col].to_numpy(),
            y_prob=val_df[prob_col].to_numpy(),
        )

        threshold_df["model"] = model_name
        threshold_rows.append(threshold_df)

        for year, year_df in test_df.groupby(year_col):
            y_true = year_df[target_col].to_numpy().astype(int)
            y_prob = year_df[prob_col].to_numpy().astype(float)
            y_pred = (y_prob >= best_threshold).astype(int)

            metric_rows.append(
                {
                    "model": model_name,
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

    metrics_df = pd.DataFrame(metric_rows).sort_values(["model", "year"]).reset_index(drop=True)
    thresholds_df = pd.concat(threshold_rows, ignore_index=True)

    metrics_df.to_csv(results_dir / f"{output_prefix}_yearly_metrics.csv", index=False)
    thresholds_df.to_csv(results_dir / f"{output_prefix}_threshold_search.csv", index=False)

    return metrics_df, thresholds_df


def summarize_sample_matched_metrics(metrics_df: pd.DataFrame, results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)

    summary_df = (
        metrics_df.groupby("model", as_index=False)
        .agg(
            n_years=("year", "nunique"),
            total_rows=("n_rows", "sum"),
            mean_auc=("auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            mean_pred_positive_rate=("pred_positive_rate", "mean"),
        )
        .sort_values("mean_auc", ascending=False)
        .reset_index(drop=True)
    )

    summary_df.to_csv(results_dir / "sample_matched_model_summary.csv", index=False)
    return summary_df


def plot_metric_by_year(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model_name, model_df in metrics_df.groupby("model"):
        model_df = model_df.sort_values("year")
        plt.plot(model_df["year"], model_df[metric], marker="o", label=model_name)

    plt.xlabel("Future test year")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path


def save_sample_matched_plots(metrics_df: pd.DataFrame, results_dir: str | Path) -> None:
    results_dir = Path(results_dir)

    plot_metric_by_year(
        metrics_df,
        metric="auc",
        output_path=results_dir / "sample_matched_auc_by_year.png",
        title="Sample-Matched AUC by Year",
    )

    plot_metric_by_year(
        metrics_df,
        metric="f1",
        output_path=results_dir / "sample_matched_f1_by_year.png",
        title="Sample-Matched F1 by Year",
    )

    plot_metric_by_year(
        metrics_df,
        metric="brier_score",
        output_path=results_dir / "sample_matched_brier_by_year.png",
        title="Sample-Matched Brier Score by Year",
    )

    plot_metric_by_year(
        metrics_df,
        metric="ece",
        output_path=results_dir / "sample_matched_ece_by_year.png",
        title="Sample-Matched ECE by Year",
    )


def run_sample_matched_comparison(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    llm_predictions_df: pd.DataFrame,
    models: Dict[str, object],
    feature_cols: list[str],
    results_dir: str | Path,
    llm_model_name: str | None = None,
    id_col: str = "id",
) -> dict[str, pd.DataFrame]:
    """
    Full sample-matched comparison:
    - recover original rows used by LLM
    - predict structured models on those rows
    - combine structured and LLM predictions
    - tune thresholds on validation sample
    - evaluate by future year
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_df = build_original_sample_frame(
        val_df=val_df,
        test_df=test_df,
        llm_predictions_df=llm_predictions_df,
        id_col=id_col,
    )

    structured_pred_df = predict_structured_models_on_sample(
        sample_df=sample_df,
        models=models,
        feature_cols=feature_cols,
    )

    llm_pred_df = build_llm_prediction_frame(
        llm_predictions_df=llm_predictions_df,
        model_name=llm_model_name,
    )

    prediction_df = pd.concat(
        [structured_pred_df, llm_pred_df],
        ignore_index=True,
    ).sort_values(["model", "split", "year", "llm_row_id"])

    prediction_df.to_csv(results_dir / "sample_matched_predictions.csv", index=False)

    metrics_df, thresholds_df = evaluate_sample_matched_predictions(
        prediction_df=prediction_df,
        results_dir=results_dir,
        output_prefix="sample_matched",
    )

    summary_df = summarize_sample_matched_metrics(
        metrics_df=metrics_df,
        results_dir=results_dir,
    )

    save_sample_matched_plots(
        metrics_df=metrics_df,
        results_dir=results_dir,
    )

    return {
        "sample_matched_predictions": prediction_df,
        "sample_matched_yearly_metrics": metrics_df,
        "sample_matched_threshold_search": thresholds_df,
        "sample_matched_model_summary": summary_df,
    }