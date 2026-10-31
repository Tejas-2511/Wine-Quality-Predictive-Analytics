"""
eda.py
------
Exploratory Data Analysis visualizations for Wine Quality Predictive Analytics.
Every plot function saves a high-resolution PNG to outputs/graphs/ and returns
the matplotlib Figure object.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_preprocessing import (
    BINARY_TARGET,
    FEATURES,
    GRAPH_DIR,
    TARGET,
    ensure_output_dirs,
)

PRIMARY_BLUE = "#2a78d6"
ACCENT_ORANGE = "#eb6834"
LIGHT_BLUE = "#cfe0f6"
BORDER_GREY = "#8a8984"


def set_style():
    """Configure clean and professional matplotlib / seaborn plot styling."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e5e1",
        "axes.edgecolor": BORDER_GREY,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    """Save figure to outputs/graphs/<name>.png and return path."""
    ensure_output_dirs()
    path = GRAPH_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def plot_target_distribution(df: pd.DataFrame, name="01_target_distribution"):
    """
    Plot the distribution of the original quality score (3-8)
    alongside the binary classification target (Normal/Low vs High).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 1. Discrete quality score counts
    counts = df[TARGET].value_counts().sort_index()
    bars = axes[0].bar(counts.index.astype(str), counts.values, color=PRIMARY_BLUE, width=0.6)
    axes[0].bar_label(bars, labels=[f"{v} ({v / counts.sum():.1%})" for v in counts.values], fontsize=8)
    axes[0].set_title("Distribution of Wine Quality Scores (3-8)", fontweight="bold")
    axes[0].set_xlabel("Quality Rating (Median of Expert Evaluations)")
    axes[0].set_ylabel("Number of Wines")
    axes[0].set_ylim(0, counts.max() * 1.2)

    # 2. Binary target distribution
    bin_counts = df[BINARY_TARGET].value_counts().sort_index()
    labels = ["Normal/Low (< 7)", "High (>= 7)"]
    bars2 = axes[1].bar(labels, bin_counts.values, color=[PRIMARY_BLUE, ACCENT_ORANGE], width=0.5)
    axes[1].bar_label(bars2, labels=[f"{v} ({v / bin_counts.sum():.1%})" for v in bin_counts.values], fontsize=9)
    axes[1].set_title("Binary Target Distribution (High vs Normal/Low)", fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Number of Wines")
    axes[1].set_ylim(0, bin_counts.max() * 1.2)

    fig.tight_layout()
    save(fig, name)
    return fig


def plot_feature_distributions(df: pd.DataFrame, name="02_feature_distributions"):
    """Histograms with KDE overlay and skewness values for all 11 features."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, col in zip(axes.flat, FEATURES):
        skew_val = df[col].skew()
        sns.histplot(df[col], bins=30, kde=True, color=PRIMARY_BLUE, edgecolor="white", ax=ax)
        ax.set_title(f"{col}\n(skew = {skew_val:.2f})", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes.flat[-1].axis("off")
    fig.suptitle("Distributions of Physicochemical Features (Histograms + KDE)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, name)
    return fig


def plot_feature_boxplots(df: pd.DataFrame, name="03_feature_boxplots"):
    """Boxplots of all 11 features showing median, IQR spread, and outliers."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, col in zip(axes.flat, FEATURES):
        sns.boxplot(y=df[col], color=LIGHT_BLUE, fliersize=2, linewidth=1, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("")

    axes.flat[-1].axis("off")
    fig.suptitle("Feature Boxplots (Points Represent 1.5 * IQR Outliers)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, name)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, name="04_correlation_heatmap"):
    """Lower-triangle Pearson correlation matrix heatmap."""
    corr = df[FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.7, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.grid(False)
    ax.set_title("Correlation Heatmap (Physicochemical Features & Quality)", fontweight="bold")
    save(fig, name)
    return fig


def plot_top_features_vs_quality(df: pd.DataFrame, name="05_top_features_vs_quality"):
    """
    Boxplots and mean trends for the top two correlated features:
    Alcohol (positive driver) and Volatile Acidity (negative driver).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, feature, desc in zip(
        axes,
        ["alcohol", "volatile acidity"],
        ["Strongest Positive Driver (r = +0.48)", "Strongest Negative Driver (r = -0.39)"],
    ):
        sns.boxplot(
            data=df,
            x=TARGET,
            y=feature,
            color=LIGHT_BLUE,
            width=0.55,
            fliersize=2,
            linewidth=1,
            ax=ax,
        )
        means = df.groupby(TARGET)[feature].mean()
        ax.plot(range(len(means)), means.values, color=ACCENT_ORANGE, marker="o", lw=2, label="Mean Trend")
        ax.set_title(f"{feature.title()} vs Quality\n({desc})", fontweight="bold", fontsize=11)
        ax.set_xlabel("Quality Score")
        ax.set_ylabel(feature)
        ax.legend(loc="best")

    fig.tight_layout()
    save(fig, name)
    return fig
