"""
model_evaluation.py
-------------------
Standard evaluation metrics for regression and classification models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE, MSE, RMSE, and R2 on raw unrounded predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """Compute Accuracy, Precision, Recall, F1, and ROC-AUC."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan
    return metrics
