"""
classification_models.py
------------------------
Classification models for predicting high wine quality (quality >= 7).
Implements Logistic Regression, Linear Discriminant Analysis (LDA),
and Quadratic Discriminant Analysis (QDA).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import RANDOM_STATE
from .eda import save
from .model_evaluation import classification_metrics

SKF5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_classifiers() -> dict:
    """
    Build classification models inside Pipelines with standard scaling.
    Scaling ensures numeric stability and consistency across models.
    """
    scale = lambda m: Pipeline([("scaler", StandardScaler()), ("model", m)])
    return {
        "Baseline (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": scale(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "Linear Discriminant Analysis (LDA)": scale(LinearDiscriminantAnalysis()),
        "Quadratic Discriminant Analysis (QDA)": scale(QuadraticDiscriminantAnalysis(reg_param=0.0)),
    }


def class_balance(y) -> pd.DataFrame:
    """Return count and proportion of high quality vs normal/low quality wines."""
    c = pd.Series(y).value_counts().sort_index()
    names = {0: "Normal/Low (< 7)", 1: "High (>= 7)"}
    return pd.DataFrame({
        "Class": [names.get(i, str(i)) for i in c.index],
        "Count": c.values,
        "Share (%)": (100 * c / c.sum()).round(2).values,
    }).set_index("Class")


def fit_and_evaluate(models: dict, X_train, X_test, y_train, y_test):
    """
    Fit classifiers, predict test instances, perform 5-fold CV,
    and calculate classification metrics and confusion matrices.
    """
    rows = []
    scores = {}
    cms = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = None

        scores[name] = y_score
        cms[name] = confusion_matrix(y_test, y_pred)

        m = classification_metrics(y_test, y_pred, y_score)

        if not name.startswith("Baseline"):
            cv_f1 = cross_val_score(model, X_train, y_train, cv=SKF5, scoring="f1").mean()
            cv_auc = cross_val_score(model, X_train, y_train, cv=SKF5, scoring="roc_auc").mean()
        else:
            cv_f1 = 0.0
            cv_auc = 0.5

        m["CV_F1 (Train)"] = cv_f1
        m["CV_ROC_AUC (Train)"] = cv_auc

        rows.append({"Model": name, **m})

    table = pd.DataFrame(rows).set_index("Model")
    return models, table, scores, cms


def plot_confusion_matrices(cms: dict, name="09_classification_confusion_matrices"):
    """Plot side-by-side confusion matrices on test set."""
    names = [n for n in cms if not n.startswith("Baseline")]
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4))

    for ax, n in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay(cms[n], display_labels=["< 7", ">= 7"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(n, fontsize=10, fontweight="bold")
        ax.grid(False)

    fig.suptitle("Test Set Confusion Matrices (Rows: Actual, Cols: Predicted)", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, name)
    return fig


def plot_roc_curves(y_test, scores: dict, name="10_classification_roc_curves"):
    """Plot ROC curves for Logistic Regression, LDA, and QDA."""
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    active_scores = [(n, s) for n, s in scores.items() if not n.startswith("Baseline")]
    for (m_name, s), c in zip(active_scores, colors):
        fpr, tpr, _ = roc_curve(y_test, s)
        auc_val = roc_auc_score(y_test, s)
        ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{m_name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", label="Random Chance (AUC = 0.500)")
    ax.set_title("ROC Curves - High Quality Classification", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    save(fig, name)
    return fig
