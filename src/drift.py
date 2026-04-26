from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-6
MISSING = "__MISSING__"
OTHER = "__OTHER__"


def _safe_proportions(counts: pd.Series, categories: list[str]) -> np.ndarray:
    counts = counts.reindex(categories, fill_value=0).astype(float)
    total = counts.sum()
    if total <= 0:
        return np.full(len(categories), 1.0 / len(categories))
    props = counts / total
    return np.clip(props.to_numpy(), EPS, None)


def _psi_from_counts(reference_counts: pd.Series, actual_counts: pd.Series) -> float:
    categories = sorted(set(reference_counts.index.astype(str)) | set(actual_counts.index.astype(str)))
    expected = _safe_proportions(reference_counts, categories)
    actual = _safe_proportions(actual_counts, categories)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_edges(reference: pd.Series, n_bins: int = 10) -> Optional[np.ndarray]:
    ref = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if ref.nunique() < 2:
        return None

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    if len(edges) < 3:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _numeric_bin_counts(series: pd.Series, edges: np.ndarray) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bins = pd.cut(values, bins=edges, include_lowest=True).astype(str)
    bins = bins.where(values.notna(), MISSING)
    return bins.value_counts(dropna=False)


def _categorical_counts(
    series: pd.Series,
    reference_top_categories: Optional[set[str]] = None,
) -> pd.Series:
    values = series.astype("object").where(series.notna(), MISSING).astype(str)

    if reference_top_categories is not None:
        values = values.where(values.isin(reference_top_categories), OTHER)

    return values.value_counts(dropna=False)


def compute_feature_psi(
    train_df: pd.DataFrame,
    year_df: pd.DataFrame,
    feature: str,
    n_bins: int = 10,
    max_categories: int = 25,
) -> tuple[float, str]:
    """
    Compute PSI for one feature comparing train_df to one future-year dataframe.

    Returns
    -------
    psi, feature_type
    """
    if feature not in train_df.columns or feature not in year_df.columns:
        return np.nan, "missing"

    ref = train_df[feature]
    actual = year_df[feature]

    if pd.api.types.is_numeric_dtype(ref):
        edges = _numeric_edges(ref, n_bins=n_bins)
        if edges is None:
            return np.nan, "numeric_constant"

        ref_counts = _numeric_bin_counts(ref, edges)
        actual_counts = _numeric_bin_counts(actual, edges)
        return _psi_from_counts(ref_counts, actual_counts), "numeric"

    ref_clean = ref.astype("object").where(ref.notna(), MISSING).astype(str)
    top_categories = set(ref_clean.value_counts().head(max_categories).index.astype(str))

    ref_counts = _categorical_counts(ref, reference_top_categories=top_categories)
    actual_counts = _categorical_counts(actual, reference_top_categories=top_categories)

    return _psi_from_counts(ref_counts, actual_counts), "categorical"


def compute_psi_by_feature_year(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    year_col: str = "year",
    n_bins: int = 10,
    max_categories: int = 25,
) -> pd.DataFrame:
    """
    Compute PSI for each original feature in each future test year.
    """
    if year_col not in train_df.columns or year_col not in test_df.columns:
        raise ValueError(f"Both train_df and test_df must contain '{year_col}'.")

    records = []
    years = sorted(test_df[year_col].dropna().unique().tolist())

    for feature in feature_cols:
        if feature not in train_df.columns or feature not in test_df.columns:
            continue

        for year in years:
            year_df = test_df[test_df[year_col] == year].copy()
            psi, feature_type = compute_feature_psi(
                train_df=train_df,
                year_df=year_df,
                feature=feature,
                n_bins=n_bins,
                max_categories=max_categories,
            )

            records.append(
                {
                    "year": int(year),
                    "feature": feature,
                    "feature_type": feature_type,
                    "psi": psi,
                    "train_missing_rate": float(train_df[feature].isna().mean()),
                    "year_missing_rate": float(year_df[feature].isna().mean()),
                    "train_nunique": int(train_df[feature].nunique(dropna=True)),
                    "year_nunique": int(year_df[feature].nunique(dropna=True)),
                }
            )

    return pd.DataFrame(records).sort_values(["year", "feature"]).reset_index(drop=True)


def compute_base_rate_by_year(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Default",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute default-rate shift relative to the training period and first future test year.
    """
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Both train_df and test_df must contain '{target_col}'.")

    train_default_rate = float(train_df[target_col].mean())

    rows = []
    for year, year_df in test_df.groupby(year_col):
        rows.append(
            {
                "year": int(year),
                "n_rows": int(len(year_df)),
                "default_rate": float(year_df[target_col].mean()),
                "train_default_rate": train_default_rate,
                "default_rate_shift_from_train": float(year_df[target_col].mean() - train_default_rate),
            }
        )

    base_rate_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    first_future_rate = float(base_rate_df.loc[0, "default_rate"])
    base_rate_df["first_test_year_default_rate"] = first_future_rate
    base_rate_df["default_rate_shift_from_first_test_year"] = (
        base_rate_df["default_rate"] - first_future_rate
    )

    return base_rate_df


def summarize_drift_by_year(
    psi_df: pd.DataFrame,
    base_rate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact year-level drift summary.
    """
    valid_psi = psi_df.dropna(subset=["psi"]).copy()

    summary = (
        valid_psi.groupby("year")
        .agg(
            mean_psi=("psi", "mean"),
            median_psi=("psi", "median"),
            max_psi=("psi", "max"),
            num_features=("feature", "nunique"),
            num_features_psi_gt_0_10=("psi", lambda x: int((x > 0.10).sum())),
            num_features_psi_gt_0_25=("psi", lambda x: int((x > 0.25).sum())),
        )
        .reset_index()
    )

    return base_rate_df.merge(summary, on="year", how="left").sort_values("year").reset_index(drop=True)


def top_drift_features(psi_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Return features with largest observed PSI across future years.
    """
    return (
        psi_df.dropna(subset=["psi"])
        .groupby(["feature", "feature_type"], as_index=False)
        .agg(
            max_psi=("psi", "max"),
            mean_psi=("psi", "mean"),
        )
        .sort_values("max_psi", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def plot_base_rate_by_year(
    base_rate_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(base_rate_df["year"], base_rate_df["default_rate"], marker="o", label="Future-year default rate")
    plt.axhline(
        base_rate_df["train_default_rate"].iloc[0],
        linestyle="--",
        label="Training-period default rate",
    )
    plt.xlabel("Year")
    plt.ylabel("Default rate")
    plt.title("Default Base Rate Shift by Year")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path


def plot_psi_heatmap_top_features(
    psi_df: pd.DataFrame,
    output_path: str | Path,
    n_features: int = 20,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_features = top_drift_features(psi_df, n=n_features)["feature"].tolist()
    plot_df = psi_df[psi_df["feature"].isin(top_features)].copy()

    pivot = plot_df.pivot_table(
        index="feature",
        columns="year",
        values="psi",
        aggfunc="mean",
    )

    # Order rows by maximum PSI so the most shifted features appear first.
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Future test year")
    ax.set_ylabel("Feature")
    ax.set_title("PSI Heatmap for Top-Drifting Features")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Population Stability Index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path


def save_drift_analysis_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str],
    results_dir: str | Path,
    target_col: str = "Default",
    year_col: str = "year",
    n_bins: int = 10,
    max_categories: int = 25,
    top_n_features: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Compute and save final dataset-drift artifacts.

    Returns a dictionary of the main dataframes for notebook display.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_rate_df = compute_base_rate_by_year(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        year_col=year_col,
    )

    psi_df = compute_psi_by_feature_year(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        year_col=year_col,
        n_bins=n_bins,
        max_categories=max_categories,
    )

    top_features_df = top_drift_features(psi_df, n=top_n_features)

    drift_summary_df = summarize_drift_by_year(
        psi_df=psi_df,
        base_rate_df=base_rate_df,
    )

    base_rate_df.to_csv(results_dir / "base_rate_by_year.csv", index=False)
    psi_df.to_csv(results_dir / "psi_by_feature_year.csv", index=False)
    top_features_df.to_csv(results_dir / "psi_top_features.csv", index=False)
    drift_summary_df.to_csv(results_dir / "drift_summary_by_year.csv", index=False)

    plot_base_rate_by_year(
        base_rate_df=base_rate_df,
        output_path=results_dir / "base_rate_by_year.png",
    )

    plot_psi_heatmap_top_features(
        psi_df=psi_df,
        output_path=results_dir / "psi_heatmap_top_features.png",
        n_features=top_n_features,
    )

    return {
        "base_rate_by_year": base_rate_df,
        "psi_by_feature_year": psi_df,
        "psi_top_features": top_features_df,
        "drift_summary_by_year": drift_summary_df,
    }