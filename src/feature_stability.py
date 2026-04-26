from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_transformed_feature_names(pipeline) -> np.ndarray:
    """
    Extract feature names after the fitted ColumnTransformer preprocessing step.
    """
    if "preprocess" not in pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'preprocess' step.")

    preprocessor = pipeline.named_steps["preprocess"]

    try:
        return preprocessor.get_feature_names_out()
    except Exception as exc:
        raise RuntimeError(
            "Could not extract transformed feature names from the preprocessing step. "
            "Make sure the pipeline is fitted before calling this function."
        ) from exc


def _clean_transformed_name(name: str) -> str:
    """
    Remove sklearn ColumnTransformer prefixes like num__ and cat__.
    """
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        return name.replace("cat__", "", 1)
    return name


def _base_feature_from_transformed_name(
    transformed_name: str,
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
) -> str:
    """
    Map transformed feature names back to their original feature column.

    Examples:
    - num__loan_amnt -> loan_amnt
    - cat__purpose_debt_consolidation -> purpose
    - cat__home_ownership_n_RENT -> home_ownership_n
    """
    clean_name = _clean_transformed_name(transformed_name)

    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols)

    if clean_name in numeric_cols:
        return clean_name

    # Match longest categorical column first in case one name is a prefix of another.
    for col in sorted(categorical_cols, key=len, reverse=True):
        if clean_name == col or clean_name.startswith(col + "_"):
            return col

    return clean_name


def extract_logreg_feature_importance(
    logreg_pipeline,
    feature_groups: Dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract transformed-level and original-feature-level logistic regression importances.
    """
    feature_names = _get_transformed_feature_names(logreg_pipeline)

    if "model" not in logreg_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'model' step.")

    model = logreg_pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        raise ValueError("The logistic model does not expose coef_.")

    coefficients = model.coef_.ravel()

    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Feature-name length ({len(feature_names)}) does not match coefficient length "
            f"({len(coefficients)})."
        )

    transformed_df = pd.DataFrame(
        {
            "model": "logreg",
            "transformed_feature": feature_names,
            "clean_feature": [_clean_transformed_name(name) for name in feature_names],
            "coefficient": coefficients,
            "importance_abs": np.abs(coefficients),
        }
    )

    transformed_df["base_feature"] = transformed_df["transformed_feature"].apply(
        lambda name: _base_feature_from_transformed_name(
            name,
            numeric_cols=feature_groups["numeric_cols"],
            categorical_cols=feature_groups["categorical_cols"],
        )
    )

    base_df = (
        transformed_df.groupby("base_feature", as_index=False)
        .agg(
            importance_abs=("importance_abs", "sum"),
            mean_abs_coefficient=("importance_abs", "mean"),
            max_abs_coefficient=("importance_abs", "max"),
            n_transformed_features=("transformed_feature", "count"),
        )
        .sort_values("importance_abs", ascending=False)
        .reset_index(drop=True)
    )

    base_df.insert(0, "model", "logreg")

    return transformed_df.sort_values("importance_abs", ascending=False), base_df


def extract_xgboost_feature_importance(
    xgboost_pipeline,
    feature_groups: Dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract transformed-level and original-feature-level XGBoost importances.
    """
    feature_names = _get_transformed_feature_names(xgboost_pipeline)

    if "model" not in xgboost_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'model' step.")

    model = xgboost_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("The XGBoost model does not expose feature_importances_.")

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature-name length ({len(feature_names)}) does not match importance length "
            f"({len(importances)})."
        )

    transformed_df = pd.DataFrame(
        {
            "model": "xgboost",
            "transformed_feature": feature_names,
            "clean_feature": [_clean_transformed_name(name) for name in feature_names],
            "importance": importances,
            "importance_abs": np.abs(importances),
        }
    )

    transformed_df["base_feature"] = transformed_df["transformed_feature"].apply(
        lambda name: _base_feature_from_transformed_name(
            name,
            numeric_cols=feature_groups["numeric_cols"],
            categorical_cols=feature_groups["categorical_cols"],
        )
    )

    base_df = (
        transformed_df.groupby("base_feature", as_index=False)
        .agg(
            importance=("importance", "sum"),
            importance_abs=("importance_abs", "sum"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            n_transformed_features=("transformed_feature", "count"),
        )
        .sort_values("importance_abs", ascending=False)
        .reset_index(drop=True)
    )

    base_df.insert(0, "model", "xgboost")

    return transformed_df.sort_values("importance_abs", ascending=False), base_df


def compute_top_feature_overlap(
    logreg_base_df: pd.DataFrame,
    xgboost_base_df: pd.DataFrame,
    ks: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Compare whether logistic regression and XGBoost rely on similar top features.
    """
    records = []

    for k in ks:
        logreg_top = set(logreg_base_df.head(k)["base_feature"])
        xgboost_top = set(xgboost_base_df.head(k)["base_feature"])

        intersection = logreg_top & xgboost_top
        union = logreg_top | xgboost_top

        records.append(
            {
                "top_k": k,
                "overlap_count": len(intersection),
                "jaccard_overlap": len(intersection) / len(union) if union else np.nan,
                "overlapping_features": ", ".join(sorted(intersection)),
            }
        )

    return pd.DataFrame(records)


def plot_top_features(
    base_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    top_n: int = 15,
) -> Path:
    """
    Save a horizontal bar plot of top original-feature importances.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = base_df.sort_values("importance_abs", ascending=False).head(top_n)
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["base_feature"], plot_df["importance_abs"])
    plt.xlabel("Aggregated absolute importance")
    plt.ylabel("Original feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    return output_path


def save_feature_reliance_outputs(
    logreg_model,
    xgboost_model,
    feature_groups: Dict,
    results_dir: str | Path,
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    """
    Extract and save feature-reliance artifacts for logistic regression and XGBoost.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logreg_transformed_df, logreg_base_df = extract_logreg_feature_importance(
        logreg_pipeline=logreg_model,
        feature_groups=feature_groups,
    )

    xgboost_transformed_df, xgboost_base_df = extract_xgboost_feature_importance(
        xgboost_pipeline=xgboost_model,
        feature_groups=feature_groups,
    )

    overlap_df = compute_top_feature_overlap(
        logreg_base_df=logreg_base_df,
        xgboost_base_df=xgboost_base_df,
    )

    logreg_transformed_df.to_csv(
        results_dir / "logreg_transformed_feature_importance.csv",
        index=False,
    )
    xgboost_transformed_df.to_csv(
        results_dir / "xgboost_transformed_feature_importance.csv",
        index=False,
    )
    logreg_base_df.to_csv(
        results_dir / "logreg_feature_importance.csv",
        index=False,
    )
    xgboost_base_df.to_csv(
        results_dir / "xgboost_feature_importance.csv",
        index=False,
    )
    overlap_df.to_csv(
        results_dir / "feature_importance_overlap.csv",
        index=False,
    )

    plot_top_features(
        base_df=logreg_base_df,
        output_path=results_dir / "top_logreg_features.png",
        title="Top Logistic Regression Feature Reliance",
        top_n=top_n,
    )

    plot_top_features(
        base_df=xgboost_base_df,
        output_path=results_dir / "top_xgboost_features.png",
        title="Top XGBoost Feature Reliance",
        top_n=top_n,
    )

    return {
        "logreg_transformed_feature_importance": logreg_transformed_df,
        "xgboost_transformed_feature_importance": xgboost_transformed_df,
        "logreg_feature_importance": logreg_base_df,
        "xgboost_feature_importance": xgboost_base_df,
        "feature_importance_overlap": overlap_df,
    }