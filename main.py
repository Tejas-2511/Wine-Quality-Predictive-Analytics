"""
main.py - Executes the Wine Quality Predictive Analytics Pipeline.

Structure:
  Phase 1: Data Preparation & Cleaning
  Phase 2: Exploratory Data Analysis (EDA)
  Phase 3: Regression Analysis (Simple, Multiple, Ridge)
  Phase 4: Classification Analysis (Logistic Regression, LDA, QDA)

All figures are saved to outputs/graphs/, and all tabular results to outputs/results/.
"""
import json
import time
import warnings
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend to save figures to disk

import pandas as pd
from src import classification_models as clf
from src import eda
from src import regression_models as reg
from src.data_preprocessing import (
    FEATURES,
    GRAPH_DIR,
    RESULT_DIR,
    TARGET,
    add_binary_target,
    clean_data,
    ensure_output_dirs,
    get_data_overview,
    iqr_outlier_summary,
    load_wine_data,
    quality_check,
    split_classification,
    split_regression,
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 15)


def print_section(title: str):
    sep = "=" * 78
    print(f"\n{sep}\n {title}\n{sep}")


def save_csv(df: pd.DataFrame, filename: str):
    import matplotlib.pyplot as plt
    plt.close("all")
    df.to_csv(RESULT_DIR / f"{filename}.csv")


def main():
    start_time = time.time()
    ensure_output_dirs()
    eda.set_style()
    summary = {}

    # -------------------------------------------------------------------------
    # PHASE 1: Data Preparation
    # -------------------------------------------------------------------------
    print_section("PHASE 1: Data Loading & Preparation")
    raw_df = load_wine_data()
    print("Raw Data Preview:\n", raw_df.head(3))
    
    overview = get_data_overview(raw_df)
    qc = quality_check(raw_df)
    print("\nData Health Check:", qc)
    
    # Save descriptive stats and outlier summary
    desc_stats = raw_df.describe().T.round(4)
    save_csv(desc_stats, "01_descriptive_statistics")
    
    outliers = iqr_outlier_summary(raw_df)
    save_csv(outliers, "02_iqr_outliers")
    print("\nTop Outlier Features (IQR Method):\n", outliers[["IQR", "Outlier Count", "Outlier Pct (%)"]])

    # Clean data (remove duplicates) and add binary target
    clean_df, clean_log = clean_data(raw_df)
    clean_df = add_binary_target(clean_df)
    print("\nData Cleaning Log:", clean_log)

    summary["health_check"] = qc
    summary["cleaning_log"] = clean_log

    # -------------------------------------------------------------------------
    # PHASE 2: Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------------------
    print_section("PHASE 2: Exploratory Data Analysis (EDA)")
    eda.plot_target_distribution(clean_df)
    eda.plot_feature_distributions(clean_df)
    eda.plot_feature_boxplots(clean_df)
    eda.plot_correlation_heatmap(clean_df)
    eda.plot_top_features_vs_quality(clean_df)

    corr_matrix = clean_df[FEATURES + [TARGET]].corr().round(4)
    save_csv(corr_matrix, "03_correlation_matrix")
    
    quality_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("Correlation of Features with Wine Quality:\n", quality_corr)
    summary["top_correlations"] = quality_corr.to_dict()

    # -------------------------------------------------------------------------
    # PHASE 3: Regression Analysis (Quality as a Numeric Target)
    # -------------------------------------------------------------------------
    print_section("PHASE 3: Regression Analysis (Continuous Quality 3-8)")
    X_train_r, X_test_r, y_train_r, y_test_r = split_regression(clean_df)
    print(f"Regression Train Samples: {len(X_train_r)}, Test Samples: {len(X_test_r)}")

    reg_models = reg.build_regressors()
    reg_models, reg_results, reg_preds = reg.fit_and_evaluate(
        reg_models, X_train_r, X_test_r, y_train_r, y_test_r
    )
    reg_results = reg_results.round(4)
    save_csv(reg_results, "04_regression_model_comparison")
    print("\nRegression Model Performance:\n", reg_results)

    # Standardized coefficients from Multiple Linear Regression
    coefs = reg.get_linear_coefficients(reg_models["Multiple Linear Regression"])
    save_csv(coefs.round(4), "05_linear_regression_coefficients")
    print("\nMultiple Linear Regression Standardized Coefficients:\n", coefs)

    # Diagnostic plots
    reg.plot_actual_vs_predicted(y_test_r, reg_preds)
    reg.plot_residuals(y_test_r, reg_preds)
    reg.plot_model_comparison(reg_results)

    summary["regression_results"] = reg_results.to_dict(orient="index")

    # -------------------------------------------------------------------------
    # PHASE 4: Classification Analysis (LDA & QDA & Logistic Regression)
    # -------------------------------------------------------------------------
    print_section("PHASE 4: Classification Analysis (High Quality >= 7)")
    X_train_c, X_test_c, y_train_c, y_test_c = split_classification(clean_df)
    print(f"Classification Train Samples: {len(X_train_c)}, Test Samples: {len(X_test_c)}")

    class_dist = clf.class_balance(clean_df["high_quality"])
    save_csv(class_dist, "06_classification_class_balance")
    print("\nTarget Class Distribution:\n", class_dist)

    clf_models = clf.build_classifiers()
    clf_models, clf_results, clf_scores, clf_cms = clf.fit_and_evaluate(
        clf_models, X_train_c, X_test_c, y_train_c, y_test_c
    )
    clf_results = clf_results.round(4)
    save_csv(clf_results, "07_classification_model_comparison")
    print("\nClassification Model Performance:\n", clf_results)

    cm_df = pd.DataFrame(
        {name: cm.ravel() for name, cm in clf_cms.items()},
        index=["TN", "FP", "FN", "TP"],
    ).T
    save_csv(cm_df, "08_confusion_matrices")

    clf.plot_confusion_matrices(clf_cms)
    clf.plot_roc_curves(y_test_c, clf_scores)

    summary["classification_results"] = clf_results.to_dict(orient="index")

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------
    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    elapsed = time.time() - start_time
    print_section(f"Pipeline Completed Successfully in {elapsed:.2f} seconds!")
    print(f"Graphs saved to:   {GRAPH_DIR}")
    print(f"Results saved to:  {RESULT_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
