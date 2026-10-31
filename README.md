# Wine Quality Predictive Analytics (Mini-Project)

A focused, textbook-aligned **Predictive Analytics** mini-project implementing the core syllabus topics:
1. **Data Preparation & Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Regression Analysis** (Simple Linear, Multiple Linear, Ridge Regression)
4. **Discriminant Analysis & Classification** (Linear Discriminant Analysis - LDA, Quadratic Discriminant Analysis - QDA, and Logistic Regression)

Using the benchmark **UCI Wine Quality Dataset** (Cortez et al., 2009), this project predicts red wine quality from 11 physicochemical laboratory measurements.

---

## Syllabus Alignment

| Syllabus Topic | Mini-Project Implementation |
|---|---|
| **Data Preparation** | Data loading, health check, duplicate row removal (240 duplicates removed to prevent train-test data leakage), IQR outlier auditing, 80/20 train-test split, standard scaling via `Pipeline` |
| **EDA** | Univariate distributions (histograms + KDE + skewness), boxplots, correlation heatmap (Pearson $r$), and bivariate driver analysis (Alcohol vs Quality, Volatile Acidity vs Quality) |
| **Regression** | Predicting numeric quality (3?8) using **Baseline (Mean)**, **Simple Linear Regression** (Alcohol), **Multiple Linear Regression** (all 11 features), and **Ridge Regression** (L2 penalty). Evaluated with MAE, RMSE, and $R^2$, standardized coefficients, and residual diagnostics |
| **Classification (LDA & QDA)** | Classifying high-quality wine (`quality >= 7`) using **Linear Discriminant Analysis (LDA)**, **Quadratic Discriminant Analysis (QDA)**, and **Logistic Regression**. Evaluated with Confusion Matrices, Accuracy, Precision, Recall, F1-Score, and ROC-AUC |

---

## Project Structure

```
Wine_Quality_Predictive_Analytics/
data/
?   ??? winequality-red.csv            # Official UCI dataset (semicolon-separated)
?   ??? winequality-white.csv          # Official UCI dataset
??? notebooks/
?   ??? wine_quality_analysis.ipynb    # Clean interactive step-by-step notebook
??? src/
?   ??? data_preprocessing.py          # Data loading, cleaning, outlier checks, train-test splits
?   ??? eda.py                         # Plotting functions for distributions, boxplots, heatmap
?   ??? regression_models.py           # Simple Linear, Multiple Linear, Ridge, and residual plots
?   ??? classification_models.py       # Logistic Regression, LDA, QDA, and ROC curves
?   ??? model_evaluation.py            # Evaluation metrics (MAE, RMSE, R2, Accuracy, F1, ROC-AUC)
??? outputs/
?   ??? graphs/                        # 10 generated PNG visualizations
?   ??? results/                       # 8 CSV result tables and summary.json
??? docs/
?   ??? PROJECT_REPORT.md              # Comprehensive academic project report
?   ??? VIVA_QA.md                     # Curated viva questions and answers
??? download_data.py                   # Script to verify and download data
??? main.py                            # End-to-end execution script (~8 seconds)
??? requirements.txt                   # Minimal dependencies
??? README.md
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
python main.py
```
This runs all 4 phases in ~8 seconds, printing performance tables to the console and saving all figures to `outputs/graphs/` and tables to `outputs/results/`.

### 3. Open the Interactive Notebook
Open `notebooks/wine_quality_analysis.ipynb` in VS Code, Cursor, or Jupyter Notebook:
```bash
jupyter notebook notebooks/wine_quality_analysis.ipynb
```

---

## Summary of Results

### 1. Regression (Continuous Quality Score 3 to 8)
| Model | Test MAE | Test RMSE | Test $R^2$ |
|---|:---:|:---:|:---:|
| Baseline (Mean) | 0.7049 | 0.8443 | -0.0063 |
| Simple Linear Regression (Alcohol) | 0.5675 | 0.7327 | 0.2421 |
| **Multiple Linear Regression** | **0.5041** | **0.6565** | **0.3915** |
| Ridge Regression (L2 Regularized) | 0.5041 | 0.6564 | 0.3918 |

- **Top Positive Drivers:** Alcohol ($\beta = +0.334$), Sulphates ($\beta = +0.148$).
- **Top Negative Drivers:** Volatile Acidity ($\beta = -0.178$), Total Sulfur Dioxide ($\beta = -0.124$), pH ($\beta = -0.122$).

### 2. Classification: LDA vs QDA vs Logistic Regression (`quality >= 7`)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline (Majority Class) | 86.40% | 0.00% | 0.00% | 0.0000 | 0.5000 |
| Logistic Regression | 87.50% | 60.00% | 24.32% | 0.3462 | **0.8860** |
| **Linear Discriminant Analysis (LDA)** | **88.97%** | **66.67%** | 37.84% | **0.4828** | 0.8855 |
| Quadratic Discriminant Analysis (QDA) | 85.29% | 46.15% | **48.65%** | 0.4737 | 0.8094 |

**Why LDA beats QDA here:** High-quality wines represent an imbalanced minority class (~13.5%). QDA estimates individual covariance matrices for each class ($K \times \frac{p(p+1)}{2}$ parameters), leading to higher estimation variance. LDA pools the covariance matrix across classes, providing robust regularization.

---

## Documentation
- Detailed report: [PROJECT_REPORT.md](PROJECT_REPORT.md)

