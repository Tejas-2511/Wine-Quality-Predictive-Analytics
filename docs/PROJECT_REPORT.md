# Project Report
## Wine Quality Predictive Analytics (Mini-Project)

**Dataset:** UCI Machine Learning Repository - Red Wine Quality (Cortez et al., 2009)  
**Primary Focus:** Exploratory Data Analysis, Data Preparation, Regression, and Discriminant Analysis (LDA & QDA)

---

### 1. Introduction and Problem Definition

Predictive analytics uses historical observations, statistical theory, and machine learning algorithms to model and forecast future outcomes. In the wine industry, sensory quality assessment is conducted by human expert panels. This process is time-consuming, expensive, and subject to human fatigue.

**Problem Statement:**  
Can physicochemical laboratory measurements (such as alcohol level, volatile acidity, sulphates, and residual sugar) be used to predict:
1. The numeric sensory quality score assigned by expert evaluators (Regression problem)?
2. Whether a wine is of premium / high quality ($\text{quality} \ge 7$) vs normal/low quality (Classification / Discriminant Analysis problem)?

---

### 2. Dataset Overview

- **Source:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis (2009), *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, 47(4):547-553.
- **Samples:** 1,599 initial red wine observations from the Portuguese *Vinho Verde* region.
- **Input Variables (11 physicochemical properties):**
  1. Fixed acidity ($g(\text{tartaric acid})/\text{dm}^3$)
  2. Volatile acidity ($g(\text{acetic acid})/\text{dm}^3$)
  3. Citric acid ($g/\text{dm}^3$)
  4. Residual sugar ($g/\text{dm}^3$)
  5. Chlorides ($g(\text{sodium chloride})/\text{dm}^3$)
  6. Free sulfur dioxide ($mg/\text{dm}^3$)
  7. Total sulfur dioxide ($mg/\text{dm}^3$)
  8. Density ($g/\text{cm}^3$)
  9. pH
  10. Sulphates ($g(\text{potassium sulphate})/\text{dm}^3$)
  11. Alcohol ($\% \text{vol}$)
- **Target Variable:** `quality` ? an integer sensory score between 0 and 10 (ratings in this dataset range between 3 and 8).

---

### 3. Phase 1: Data Preparation & Cleaning

1. **Missing Values Check:**  
   The dataset was verified to have zero missing values across all 12 columns, eliminating the need for imputation.

2. **Deduplication:**  
   240 exact duplicate rows were detected (15.0% of the raw data). These duplicates were removed to yield **1,359 unique observations**. If duplicates were retained, identical measurements could appear in both the training and testing sets, resulting in **data leakage** and artificially optimistic evaluation metrics.

3. **Outlier Detection:**  
   Outlier fences were computed using the Tukey $1.5 \times \text{IQR}$ rule:
   $$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
   Features such as residual sugar (9.69% outliers) and chlorides (7.00% outliers) exhibited positive skewness with high outliers. These outliers were **retained** because they correspond to genuine, valid wine chemistry (e.g., naturally sweet or salty wines), rather than data entry errors.

4. **Target Formulation & Train-Test Split:**
   - **Numeric Target:** Continuous quality score (range 3?8).
   - **Binary Target:** `high_quality = 1` if $\text{quality} \ge 7$ else $0$. High-quality wines comprise 184 samples (13.54%), making this an imbalanced classification problem.
   - **Splits:** 80% training set (1,087 samples) and 20% test set (272 samples), using `random_state=42` and stratification for classification.

5. **Feature Standardisation:**  
   To prevent feature scale disparity and ensure numerical stability for Ridge Regression, LDA, and QDA, standardisation was applied inside an isolated `Pipeline`:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   The scaler was strictly fitted on training folds only.

---

### 4. Phase 2: Exploratory Data Analysis (EDA)

Key observations from EDA:
1. **Target Distribution:** The quality scores are approximately bell-shaped and centered around 5 (49.9%) and 6 (39.6%). Extreme scores (3, 4, 8) are rare.
2. **Correlation Analysis:**
   - **Alcohol** has the strongest positive correlation with quality ($r = +0.48$). As alcohol content increases, wine ratings tend to increase.
   - **Volatile Acidity** has the strongest negative correlation ($r = -0.40$). Volatile acidity (acetic acid / vinegar smell) degrades wine aroma and taste.
   - **Sulphates** ($r = +0.25$) and **Citric Acid** ($r = +0.23$) positively contribute to sensory quality.
3. **Collinearity:** Strong physical relationships are visible between features, such as fixed acidity and density ($r = +0.67$), fixed acidity and citric acid ($r = +0.67$), and total vs free sulfur dioxide ($r = +0.67$).

---

### 5. Phase 3: Regression Analysis (Predicting Numeric Quality)

#### Model Formulations:
1. **Baseline Model:** Predicts the training mean ($\bar{y}_{\text{train}} = 5.63$).
2. **Simple Linear Regression (Alcohol):** Fits univariate model $y = \beta_0 + \beta_1 \cdot \text{alcohol}$.
3. **Multiple Linear Regression (MLR):** Fits an OLS hyperplane on all 11 standardized predictors:
   $$y = \beta_0 + \sum_{j=1}^{11} \beta_j X_j$$
4. **Ridge Regression:** Minimizes sum of squared errors plus an L2 penalty on coefficient magnitudes:
   $$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2$$

#### Experimental Results:

| Model | Train MAE | Train RMSE | Train $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (Mean) | 0.6909 | 0.8181 | 0.0000 | 0.7049 | 0.8443 | -0.0063 |
| Simple Linear Regression (Alcohol) | 0.5678 | 0.7196 | 0.2262 | 0.5675 | 0.7327 | 0.2421 |
| **Multiple Linear Regression** | **0.5107** | **0.6585** | **0.3520** | **0.5041** | **0.6565** | **0.3915** |
| Ridge Regression (L2 Regularized) | 0.5107 | 0.6585 | 0.3520 | 0.5041 | 0.6564 | **0.3918** |

#### Standardized Coefficients (Feature Importance):
- Alcohol: **$+0.334$** (Strongest positive driver)
- Volatile Acidity: **$-0.178$** (Strongest negative driver)
- Sulphates: **$+0.148$**
- Total Sulfur Dioxide: **$-0.124$**
- pH: **$-0.122$**
- Chlorides: **$-0.116$**

#### Residual Diagnostics:
Residual plots ($\hat{y}$ vs $y - \hat{y}$) show even vertical spread around zero across the central predictions, confirming that the linear model captures the core trend without extreme non-linear curvature.

---

### 6. Phase 4: Discriminant Analysis & Classification (LDA & QDA)

#### Mathematical Formulations:
1. **Linear Discriminant Analysis (LDA):**
   - Assumes predictors given class $k$ follow $\mathcal{N}(\mu_k, \Sigma)$, with a **shared covariance matrix** $\Sigma$.
   - Discriminant function:
     $$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \ln \pi_k$$
   - The boundary between class 1 and class 0 ($\delta_1(x) = \delta_0(x)$) is linear in $x$.

2. **Quadratic Discriminant Analysis (QDA):**
   - Assumes predictors follow $\mathcal{N}(\mu_k, \Sigma_k)$ with **class-specific covariance matrices** $\Sigma_k$.
   - Discriminant function:
     $$\delta_k(x) = -\frac{1}{2} \ln |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \ln \pi_k$$
   - The decision boundary is quadratic.

3. **Logistic Regression (Benchmark):**
   - Models the log-odds directly: $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$.

#### Test Set Performance:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (Majority Class) | 86.40% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.0000 |
| Logistic Regression | 87.50% | 60.00% | 24.32% | 0.3462 | **0.8860** | 0.3721 |
| **Linear Discriminant Analysis (LDA)** | **88.97%** | **66.67%** | 37.84% | **0.4828** | 0.8855 | **0.4259** |
| Quadratic Discriminant Analysis (QDA) | 85.29% | 46.15% | **48.65%** | 0.4737 | 0.8094 | 0.4456 |

#### Confusion Matrix Summary (Test Set, N = 272):
- **LDA:** TN = 228, FP = 7, FN = 23, TP = 14 (High precision of 66.7%, fewest false alarms).
- **QDA:** TN = 214, FP = 21, FN = 19, TP = 18 (Higher sensitivity of 48.6%, but 21 false positives).
- **Logistic Regression:** TN = 229, FP = 6, FN = 28, TP = 9 (Conservative predictions at 0.5 threshold).

---

### 7. Analytical Comparison: Why LDA Outperformed QDA

1. **Parameter Estimation & Sample Size:**  
   With $p = 11$ features, estimating an unconstrained covariance matrix requires $\frac{p(p+1)}{2} = \frac{11 \times 12}{2} = 66$ parameters.  
   - LDA pools both classes to estimate a single $11 \times 11$ covariance matrix (66 parameters).  
   - QDA must estimate separate covariance matrices for class 0 and class 1 ($2 \times 66 = 132$ covariance parameters).  
   Because high-quality wines represent only 13.5% of the data (approx. 147 samples in the training set), QDA suffers from higher variance in estimating its 66 parameters from a relatively small sample.

2. **Bias-Variance Trade-off:**  
   LDA introduces a slight bias (assuming $\Sigma_1 = \Sigma_0$), but drastically lowers estimation variance. This regularizing effect allows LDA to generalize better on the test set, achieving higher accuracy (88.97% vs 85.29%) and a higher test ROC-AUC (0.8855 vs 0.8094).

---

### 8. Conclusion

1. **Data Prep:** Removing 240 duplicates was essential to prevent data leakage, while keeping IQR outliers preserved valid physical variations.
2. **Key Chemical Predictors:** Alcohol content is the primary positive driver of wine quality, while volatile acidity is the primary negative driver.
3. **Regression:** Multiple Linear Regression explains ~39.2% of wine quality variance ($R^2 = 0.3915$, RMSE = 0.6565), outperforming the baseline mean model.
4. **Classification:** LDA proved to be the most effective classifier overall, achieving 88.97% accuracy and an F1-score of 0.4828, demonstrating the practical superiority of linear decision boundaries on moderate-sized imbalanced datasets.
