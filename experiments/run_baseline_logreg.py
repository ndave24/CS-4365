from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


from src.load_data import load_and_clean_dataset, summarize_dataset
from src.preprocess import (
    PreprocessConfig,
    get_model_feature_groups,
    summarize_feature_groups,
)
from src.temporal_split import (
    TemporalSplitConfig,
    describe_split,
    describe_test_years,
    make_temporal_split,
)
from src.models.logistic import LogisticConfig, fit_logistic_pipeline
from src.evaluate import (
    add_time_gap_column,
    evaluate_temporal_by_year,
    plot_temporal_metrics,
    save_temporal_metrics,
    summarize_temporal_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline temporal logistic regression experiment."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Lending Club dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2014,
        help="Final year included in the training set.",
    )
    parser.add_argument(
        "--val-years",
        type=int,
        nargs="*",
        default=None,
        help="Optional validation years.",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit list of test years. If omitted, all years after train_end_year are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where metrics and plots will be saved.",
    )
    parser.add_argument(
        "--max-categorical-cardinality",
        type=int,
        default=50,
        help="Maximum allowed cardinality for categorical features.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include text columns as raw features. For the checkpoint baseline, leave this off.",
    )
    parser.add_argument(
        "--include-id",
        action="store_true",
        help="Include ID column as a feature. Normally this should remain off.",
    )
    parser.add_argument(
        "--keep-zip-code",
        action="store_true",
        help="Keep zip_code as a feature. By default it is dropped for the baseline.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold used for F1.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("1. Loading and cleaning dataset")
    print("=" * 80)

    df = load_and_clean_dataset(args.data_path)
    print(summarize_dataset(df).to_string(index=False))

    print("\n" + "=" * 80)
    print("2. Selecting baseline features")
    print("=" * 80)

    preprocess_config = PreprocessConfig(
        include_text=args.include_text,
        include_id=args.include_id,
        drop_zip_code=not args.keep_zip_code,
        max_categorical_cardinality=args.max_categorical_cardinality,
    )

    feature_groups = get_model_feature_groups(df, config=preprocess_config)
    print(summarize_feature_groups(feature_groups).to_string(index=False))
    print("\nFeature columns:")
    print(feature_groups["feature_cols"])

    print("\n" + "=" * 80)
    print("3. Creating temporal split")
    print("=" * 80)

    split_config = TemporalSplitConfig(
        train_end_year=args.train_end_year,
        val_years=tuple(args.val_years) if args.val_years else None,
        test_years=tuple(args.test_years) if args.test_years else None,
    )

    split_dict = make_temporal_split(df, config=split_config)

    split_summary_df = describe_split(split_dict)
    print(split_summary_df.to_string(index=False))

    test_year_summary_df = describe_test_years(split_dict["test_df"])
    print("\nTest years:")
    print(test_year_summary_df.to_string(index=False))

    split_summary_path = output_dir / "split_summary.csv"
    test_year_summary_path = output_dir / "test_year_summary.csv"
    split_summary_df.to_csv(split_summary_path, index=False)
    test_year_summary_df.to_csv(test_year_summary_path, index=False)

    print("\n" + "=" * 80)
    print("4. Fitting logistic regression baseline")
    print("=" * 80)

    logistic_config = LogisticConfig()
    model = fit_logistic_pipeline(
        train_df=split_dict["train_df"],
        feature_groups=feature_groups,
        config=logistic_config,
    )

    print("Model fit complete.")

    print("\n" + "=" * 80)
    print("5. Evaluating temporal performance by test year")
    print("=" * 80)

    results_df = evaluate_temporal_by_year(
        model=model,
        test_df=split_dict["test_df"],
        feature_groups=feature_groups,
        threshold=args.threshold,
    )
    results_df = add_time_gap_column(
        results_df,
        train_end_year=args.train_end_year,
    )

    print(summarize_temporal_results(results_df).to_string(index=False))

    metrics_path = output_dir / "temporal_metrics.csv"
    save_temporal_metrics(results_df, metrics_path)

    plot_path = output_dir / "temporal_auc_f1.png"
    plot_temporal_metrics(
        results_df=results_df,
        output_path=plot_path,
        use_time_gap=False,
        title="Temporal Performance of Logistic Regression",
    )

    time_gap_plot_path = output_dir / "temporal_auc_f1_time_gap.png"
    plot_temporal_metrics(
        results_df=results_df,
        output_path=time_gap_plot_path,
        use_time_gap=True,
        title="Temporal Performance vs Time Gap",
    )

    print("\n" + "=" * 80)
    print("6. Saved outputs")
    print("=" * 80)
    print(f"Split summary:      {split_summary_path}")
    print(f"Test-year summary:  {test_year_summary_path}")
    print(f"Metrics CSV:        {metrics_path}")
    print(f"Year plot:          {plot_path}")
    print(f"Time-gap plot:      {time_gap_plot_path}")


if __name__ == "__main__":
    main()
