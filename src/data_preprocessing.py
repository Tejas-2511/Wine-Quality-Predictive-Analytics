"""
data_preprocessing.py
---------------------
Data loading, inspection, cleaning, and preparation for Wine Quality Predictive Analytics.

Dataset: Cortez et al. (2009), UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/186/wine+quality
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "graphs"
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"

RANDOM_STATE = 42          # Fixed seed for reproducibility
TEST_SIZE = 0.20           # 80% train / 20% test split
HIGH_QUALITY_THRESHOLD = 7 # Quality >= 7 is defined as "high quality"

TARGET = "quality"
BINARY_TARGET = "high_quality"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def ensure_output_dirs():
    """Ensure output directories exist."""
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_wine_data() -> pd.DataFrame:
    """Load the official UCI Red Wine CSV (semicolon-separated)."""
    path = DATA_DIR / "winequality-red.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Ensure winequality-red.csv is in data/"
        )
    df = pd.read_csv(path, sep=";")
    expected = FEATURES + [TARGET]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns in {path.name}: {list(df.columns)}")
    return df


def get_data_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return data types, missing value count, and unique values for each feature."""
    return pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Unique Values": df.nunique(),
    })


def quality_check(df: pd.DataFrame) -> dict:
    """Return high-level dataset health metrics."""
    return {
        "Total Rows": int(df.shape[0]),
        "Total Columns": int(df.shape[1]),
        "Missing Values Total": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
    }


def iqr_outlier_summary(df: pd.DataFrame, columns=FEATURES, k: float = 1.5) -> pd.DataFrame:
    """
    Compute 1.5 * IQR outlier fences and outlier counts for each feature.
    Outliers are flagged for inspection rather than deleted, as extreme
    physicochemical readings represent valid natural wine variations.
    """
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "Feature": col,
            "Q1 (25%)": q1,
            "Q3 (75%)": q3,
            "IQR": iqr,
            "Lower Fence": low,
            "Upper Fence": high,
            "Outlier Count": n_out,
            "Outlier Pct (%)": 100 * n_out / len(df),
            "Min": df[col].min(),
            "Max": df[col].max(),
        })
    return pd.DataFrame(rows).set_index("Feature").round(4)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean the dataset:
    1. Check missing values (UCI dataset has 0 missing).
    2. Remove exact duplicate rows to prevent train-test data leakage.
    3. Retain genuine IQR outliers.
    """
    log = {
        "rows_before": len(df),
        "missing_before": int(df.isna().sum().sum()),
    }
    cleaned = df.copy()
    if log["missing_before"] > 0:
        cleaned = cleaned.fillna(cleaned.median(numeric_only=True))

    log["duplicates_removed"] = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    log["rows_after"] = len(cleaned)
    return cleaned, log


def add_binary_target(df: pd.DataFrame, threshold: int = HIGH_QUALITY_THRESHOLD) -> pd.DataFrame:
    """Create binary target high_quality: 1 if quality >= threshold else 0."""
    out = df.copy()
    out[BINARY_TARGET] = (out[TARGET] >= threshold).astype(int)
    return out


def split_regression(df: pd.DataFrame):
    """80/20 train-test split for numeric regression target (quality)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def split_classification(df: pd.DataFrame):
    """80/20 stratified train-test split for binary classification target (high_quality)."""
    X = df[FEATURES]
    y = df[BINARY_TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
