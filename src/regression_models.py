"""
regression_models.py
--------------------
Regression models for predicting wine quality as a numeric score (3 to 8).
Implements Baseline, Simple Linear Regression, Multiple Linear Regression,
and Ridge Regression (L2 Regularization).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import FEATURES, RANDOM_STATE
from .eda import ACCENT_ORANGE, PRIMARY_BLUE, save
from .model_evaluation import regression_metrics


def build_regressors() -> dict:
    """
    Construct regression models.
    Preprocessing (StandardScaler) is placed inside Pipeline to ensure
    it is fitted solely on training data without leakage.
    """
    return {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Simple Linear Regression (Alcohol)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Multiple Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression (L2 Regularized)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
    }


def fit_and_evaluate(models: dict, X_train, X_test, y_train, y_test):
    """
    Fit each model, generate predictions on train and test sets,
    and compute evaluation metrics (MAE, RMSE, R2).
    """
    rows = []
    preds = {}

    for name, model in models.items():
        if "Alcohol" in name:
            model.fit(X_train[["alcohol"]], y_train)
            p_tr = model.predict(X_train[["alcohol"]])
            p_te = model.predict(X_test[["alcohol"]])
        else:
            model.fit(X_train, y_train)
            p_tr = model.predict(X_train)
            p_te = model.predict(X_test)

        preds[name] = p_te
        tr = regression_metrics(y_train, p_tr)
        te = regression_metrics(y_test, p_te)

        rows.append({
            "Model": name,
            "Train_MAE": tr["MAE"],
            "Train_RMSE": tr["RMSE"],
            "Train_R2": tr["R2"],
            "Test_MAE": te["MAE"],
            "Test_RMSE": te["RMSE"],
            "Test_R2": te["R2"],
        })

    table = pd.DataFrame(rows).set_index("Model")
    return models, table, preds


def get_linear_coefficients(mlr_model, feature_names=FEATURES) -> pd.DataFrame:
    """
    Extract standardized coefficients from fitted Multiple Linear Regression.
    Because features are scaled, coefficients indicate relative feature importance.
    """
    model = mlr_model.named_steps["model"]
    coefs = pd.DataFrame({
        "Feature": feature_names,
        "Standardized_Coefficient": model.coef_,
        "Abs_Coefficient": np.abs(model.coef_),
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    return coefs


def plot_actual_vs_predicted(y_test, preds: dict, name="06_regression_actual_vs_predicted"):
    """Scatter plot of actual vs predicted quality scores with ideal 45-degree line."""
    eval_names = [n for n in preds if not n.startswith("Baseline")]
    fig, axes = plt.subplots(1, len(eval_names), figsize=(5 * len(eval_names), 4.5), sharey=True)

    rng = np.random.default_rng(RANDOM_STATE)
    jitter = rng.uniform(-0.12, 0.12, len(y_test))

    for ax, n in zip(np.atleast_1d(axes), eval_names):
        m = regression_metrics(y_test, preds[n])
        ax.scatter(np.asarray(y_test) + jitter, preds[n], s=16, alpha=0.5, color=PRIMARY_BLUE, edgecolor="none")
        ax.plot([3, 8], [3, 8], ls="--", color=ACCENT_ORANGE, lw=1.8, label="Ideal Line (y = x)")
        ax.set_title(f"{n}\n(RMSE: {m['RMSE']:.3f}, R2: {m['R2']:.3f})", fontsize=10)
        ax.set_xlabel("Actual Quality (Jittered for Visibility)")
        ax.set_ylabel("Predicted Quality")
        ax.set_xlim(2.5, 8.5)
        ax.set_ylim(2.5, 8.5)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Actual vs Predicted Wine Quality", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, name)
    return fig


def plot_residuals(y_test, preds: dict, name="07_regression_residuals"):
    """Residual plots (Actual - Predicted vs Predicted) to check regression assumptions."""
    eval_names = [n for n in preds if not n.startswith("Baseline")]
    fig, axes = plt.subplots(1, len(eval_names), figsize=(5 * len(eval_names), 4.2), sharey=True)

    for ax, n in zip(np.atleast_1d(axes), eval_names):
        resid = np.asarray(y_test) - preds[n]
        ax.scatter(preds[n], resid, s=16, alpha=0.5, color=PRIMARY_BLUE, edgecolor="none")
        ax.axhline(0, color=ACCENT_ORANGE, ls="--", lw=1.8)
        ax.set_title(f"{n}: Residuals", fontsize=10)
        ax.set_xlabel("Predicted Quality")
        ax.set_ylabel("Residual (Actual - Predicted)")

    fig.suptitle("Residual Plots (Checking Linearity & Homoscedasticity)", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, name)
    return fig


def plot_model_comparison(table: pd.DataFrame, name="08_regression_model_comparison"):
    """Bar charts comparing Test RMSE and Test R2 across regression models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t = table.sort_values("Test_RMSE")

    bars1 = axes[0].barh(t.index, t["Test_RMSE"], color=PRIMARY_BLUE)
    axes[0].bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set_title("Test RMSE (Lower is Better)", fontweight="bold")
    axes[0].set_xlabel("RMSE")

    bars2 = axes[1].barh(t.index, t["Test_R2"], color=ACCENT_ORANGE)
    axes[1].bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    axes[1].set_title("Test R2 (Higher is Better)", fontweight="bold")
    axes[1].set_xlabel("R2 Score")

    fig.tight_layout()
    save(fig, name)
    return fig
