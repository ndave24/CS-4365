from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import predict_positive_probability


def _as_binary_1d(y_true, name: str = "y_true") -> np.ndarray:
    """
    Convert a binary target array-like to a 1D integer numpy array.
    """
    arr = np.asarray(y_true).reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    if pd.isna(arr).any():
        raise ValueError(f"{name} contains missing values.")

    unique_values = set(np.unique(arr).tolist())
    if not unique_values.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError(
            f"{name} must be binary with values in {{0, 1}}; got {sorted(unique_values)}"
        )

    return arr.astype(int)


def _as_probability_1d(y_prob, name: str = "y_prob") -> np.ndarray:
    """
    Convert a probability array-like to a 1D float numpy array and validate [0, 1].
    """
    arr = np.asarray(y_prob, dtype=float).reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values.")

    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must lie in [0, 1].")

    return arr


def compute_brier_score(y_true, y_prob) -> float:
    """
    Compute the Brier score for binary probabilities.
    Lower is better.
    """
    y_true_arr = _as_binary_1d(y_true, name="y_true")
    y_prob_arr = _as_probability_1d(y_prob, name="y_prob")

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("y_true and y_prob must have the same length.")

    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def _build_bin_edges(
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> np.ndarray:
    """
    Build bin edges for reliability / ECE computations.

    strategy:
        - 'uniform': fixed-width bins on [0, 1]
        - 'quantile': bins based on y_prob quantiles
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_prob, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)

        if len(edges) < 3:
            raise ValueError(
                "Quantile binning collapsed too aggressively; probabilities do not "
                "have enough spread. Try strategy='uniform'."
            )
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    return edges


def build_reliability_table(
    y_true,
    y_prob,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """
    Build a reliability table with one row per probability bin.

    Output columns:
        bin_id, bin_lower, bin_upper, n_bin,
        mean_pred, frac_positive, abs_gap, weighted_gap
    """
    y_true_arr = _as_binary_1d(y_true, name="y_true")
    y_prob_arr = _as_probability_1d(y_prob, name="y_prob")

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError("y_true and y_prob must have the same length.")

    edges = _build_bin_edges(y_prob_arr, n_bins=n_bins, strategy=strategy)

    # Bin assignment:
    # searchsorted over internal edges gives integers from 0 to n_bins-1
    internal_edges = edges[1:-1]
    bin_ids = np.searchsorted(internal_edges, y_prob_arr, side="right")

    n_total = len(y_true_arr)
    records = []

    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        n_bin = int(mask.sum())

        if n_bin == 0:
            mean_pred = np.nan
            frac_positive = np.nan
            abs_gap = np.nan
            weighted_gap = 0.0
        else:
            mean_pred = float(np.mean(y_prob_arr[mask]))
            frac_positive = float(np.mean(y_true_arr[mask]))
            abs_gap = float(abs(mean_pred - frac_positive))
            weighted_gap = float(abs_gap * (n_bin / n_total))

        records.append(
            {
                "bin_id": int(bin_id),
                "bin_lower": float(edges[bin_id]),
                "bin_upper": float(edges[bin_id + 1]),
                "n_bin": n_bin,
                "mean_pred": mean_pred,
                "frac_positive": frac_positive,
                "abs_gap": abs_gap,
                "weighted_gap": weighted_gap,
            }
        )

    return pd.DataFrame(records)


def compute_ece(
    y_true,
    y_prob,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Lower is better.
    """
    reliability_df = build_reliability_table(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    return float(reliability_df["weighted_gap"].sum())


def evaluate_calibration_for_dataframe(
    model,
    df: pd.DataFrame,
    feature_groups: Dict,
    target_col: str = "Default",
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate probability calibration on a single dataframe.

    Returns:
        metrics_dict, reliability_table_df
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe.")

    y_true = _as_binary_1d(df[target_col], name=target_col)
    y_prob = predict_positive_probability(model=model, df=df, feature_groups=feature_groups)

    reliability_df = build_reliability_table(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )

    metrics = {
        "n_rows": int(len(df)),
        "default_rate": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_prob)),
        "brier_score": compute_brier_score(y_true, y_prob),
        "ece": float(reliability_df["weighted_gap"].sum()),
        "abs_mean_prob_minus_default_rate": float(abs(np.mean(y_prob) - np.mean(y_true))),
    }

    return metrics, reliability_df


def evaluate_calibration_by_year(
    model,
    test_df: pd.DataFrame,
    feature_groups: Dict,
    target_col: str = "Default",
    year_col: str = "year",
    n_bins: int = 10,
    strategy: str = "uniform",
    reference_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate calibration year-by-year on a future test dataframe.

    Returns:
        calibration_metrics_df:
            one row per year with summary metrics
        reliability_bins_df:
            one row per bin per year
    """
    if year_col not in test_df.columns:
        raise ValueError(f"year_col '{year_col}' not found in dataframe.")

    metrics_records = []
    reliability_frames = []

    years = sorted(pd.Series(test_df[year_col]).dropna().unique().tolist())

    for year in years:
        year_df = test_df.loc[test_df[year_col] == year].copy()

        metrics, reliability_df = evaluate_calibration_for_dataframe(
            model=model,
            df=year_df,
            feature_groups=feature_groups,
            target_col=target_col,
            n_bins=n_bins,
            strategy=strategy,
        )

        metrics["year"] = int(year)
        if reference_year is not None:
            metrics["time_gap"] = int(year) - int(reference_year)

        reliability_df = reliability_df.copy()
        reliability_df["year"] = int(year)
        if reference_year is not None:
            reliability_df["time_gap"] = int(year) - int(reference_year)

        metrics_records.append(metrics)
        reliability_frames.append(reliability_df)

    calibration_metrics_df = pd.DataFrame(metrics_records).sort_values("year").reset_index(drop=True)

    if reliability_frames:
        reliability_bins_df = (
            pd.concat(reliability_frames, ignore_index=True)
            .sort_values(["year", "bin_id"])
            .reset_index(drop=True)
        )
    else:
        reliability_bins_df = pd.DataFrame(
            columns=[
                "bin_id",
                "bin_lower",
                "bin_upper",
                "n_bin",
                "mean_pred",
                "frac_positive",
                "abs_gap",
                "weighted_gap",
                "year",
            ]
        )

    return calibration_metrics_df, reliability_bins_df


def plot_reliability_by_year(
    reliability_bins_df: pd.DataFrame,
    title: str = "Reliability Diagrams by Year",
    output_path: Optional[Path] = None,
    n_cols: int = 2,
    figsize_per_panel: Tuple[float, float] = (5.0, 4.0),
) -> None:
    """
    Plot one reliability diagram panel per year.
    """
    required_cols = {"year", "mean_pred", "frac_positive", "n_bin"}
    missing = required_cols - set(reliability_bins_df.columns)
    if missing:
        raise ValueError(f"reliability_bins_df is missing required columns: {sorted(missing)}")

    years = sorted(reliability_bins_df["year"].dropna().unique().tolist())
    if not years:
        raise ValueError("No years found in reliability_bins_df.")

    n_panels = len(years)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, year in zip(axes, years):
        sub = reliability_bins_df.loc[reliability_bins_df["year"] == year].copy()
        sub = sub.loc[sub["n_bin"] > 0].copy()

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.plot(sub["mean_pred"], sub["frac_positive"], marker="o")
        ax.set_title(f"Year {year}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed default rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_calibration_metric_trend(
    calibration_df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    x_col: str = "year",
    model_col: str = "model",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot a calibration metric (e.g. 'ece' or 'brier_score') over time,
    optionally with one line per model.
    """
    if metric not in calibration_df.columns:
        raise ValueError(f"metric '{metric}' not found in calibration_df.")
    if x_col not in calibration_df.columns:
        raise ValueError(f"x_col '{x_col}' not found in calibration_df.")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if model_col in calibration_df.columns:
        for model_name, sub in calibration_df.groupby(model_col):
            sub = sub.sort_values(x_col)
            ax.plot(sub[x_col], sub[metric], marker="o", label=str(model_name))
        ax.legend()
    else:
        sub = calibration_df.sort_values(x_col)
        ax.plot(sub[x_col], sub[metric], marker="o")

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} over {x_col}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.show()