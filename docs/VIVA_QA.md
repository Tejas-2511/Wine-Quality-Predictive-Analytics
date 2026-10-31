# Viva Preparation: Questions and Answers
## Wine Quality Predictive Analytics Mini-Project

This guide contains frequently asked viva questions, clear intuitive explanations, and mathematical justification covering the 4 core syllabus areas: **EDA**, **Data Preparation**, **Regression**, and **LDA & QDA**.

---

### Section 1: Exploratory Data Analysis (EDA)

#### Q1. What is the goal of EDA in predictive analytics?
**Answer:**  
Exploratory Data Analysis (EDA) is the preliminary phase where we analyze data sets to summarize their main characteristics, uncover underlying distributions, detect outliers and anomalies, and identify relationships between predictors and the target variable. It informs model selection and data transformation strategies.

#### Q2. What were the main findings from the wine quality EDA?
**Answer:**  
1. **Target Distribution:** Quality scores range from 3 to 8, but are heavily concentrated at scores 5 and 6 (~82.5% of wines). Extreme qualities (3, 4, 8) are rare.
2. **Key Positive Driver:** **Alcohol** exhibits the strongest positive correlation with quality ($r = +0.48$).
3. **Key Negative Driver:** **Volatile Acidity** exhibits the strongest negative correlation ($r = -0.40$), as excess acetic acid imparts a harsh, vinegar-like taste.
4. **Multicollinearity:** Predictors like fixed acidity, citric acid, and density show moderate-to-strong pairwise correlation ($r \approx 0.67$).

---

### Section 2: Data Preparation

#### Q3. Why did you remove the 240 duplicate rows?
**Answer:**  
In predictive analytics, identical duplicate rows present a severe risk of **data leakage**. If duplicate rows are not removed prior to splitting, the same exact wine could end up in both the training set and the testing set. The model would effectively memorize this wine, causing test evaluation metrics to be artificially inflated and misleading.

#### Q4. Why did you keep the IQR outliers instead of deleting them?
**Answer:**  
The Tukey $1.5 \times \text{IQR}$ rule flagged outliers in several features (e.g., residual sugar has ~9.7% outliers, chlorides has ~7.0%). However:
1. These values represent valid chemical extremes of real, natural wines (e.g., naturally sweet late-harvest wines or wines with higher mineral content).
2. They are not data entry or measurement errors.
3. Linear models with standard scaling and discriminant analysis can accommodate these valid natural variations. Deleting nearly 10% of real observations would discard valuable information.

#### Q5. Why is feature standardisation important, and where should it be applied?
**Answer:**  
Standardisation transforms features to have zero mean and unit variance: $z = \frac{x - \mu}{\sigma}$.
- It is crucial for **Ridge Regression** because the L2 penalty ($\alpha \sum \beta_j^2$) penalizes coefficients equally; unscaled features with large units would unfairly have tiny coefficients and avoid regularization.
- Standardisation is applied inside a `Pipeline` fitted **only on the training folds** to prevent data leakage from the test or validation sets.

---

### Section 3: Regression Analysis

#### Q6. What is the difference between Simple and Multiple Linear Regression?
**Answer:**  
- **Simple Linear Regression** models the target using a single explanatory variable:
  $$y = \beta_0 + \beta_1 X_1 + \epsilon$$
  In our project, using `alcohol` alone explained ~24.2% of the variance in wine quality ($R^2 = 0.2421$).
- **Multiple Linear Regression (MLR)** models the target using multiple predictors simultaneously:
  $$y = \beta_0 + \sum_{j=1}^p \beta_j X_j + \epsilon$$
  Using all 11 features increased the explained variance to ~39.2% ($R^2 = 0.3915$, RMSE = 0.6565).

#### Q7. What do MAE, RMSE, and $R^2$ represent?
**Answer:**  
- **MAE (Mean Absolute Error):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$. The average magnitude of prediction errors in the original units (quality points). Our test MAE was $\approx 0.50$ points.
- **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$. Penalizes larger prediction errors more heavily than MAE. Our test RMSE was $\approx 0.656$.
- **$R^2$ (Coefficient of Determination):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. The proportion of variance in the target explained by the model compared to a naive mean predictor.

#### Q8. What is Ridge Regression and why is it used?
**Answer:**  
Ridge Regression adds an $L_2$ regularization penalty to the Ordinary Least Squares (OLS) objective function:
$$\min_{\beta} \left[ \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p \beta_j^2 \right]$$
When predictors have multicollinearity (like fixed acidity, citric acid, and density in our dataset), OLS coefficient estimates can become unstable with high variance. The Ridge penalty shrinks the coefficients toward zero, reducing estimation variance and preventing overfitting.

---

### Section 4: Discriminant Analysis & Classification (LDA & QDA)

#### Q9. What is Linear Discriminant Analysis (LDA) and what are its core assumptions?
**Answer:**  
LDA is a **generative** classification method based on Bayes' theorem. It models the distribution of inputs $X$ within each class $k$:
$$P(Y=k|X=x) = \frac{\pi_k f_k(x)}{\sum_{l} \pi_l f_l(x)}$$
**Key Assumptions:**
1. The predictors in each class follow a multivariate normal distribution: $X|Y=k \sim \mathcal{N}(\mu_k, \Sigma)$.
2. **Equal Covariance:** All classes share the exact same covariance matrix ($\Sigma_1 = \Sigma_0 = \Sigma$).
Because the covariance matrices are equal, the quadratic terms $x^T \Sigma^{-1} x$ cancel out in the log-ratio, resulting in a **strictly linear decision boundary**.

#### Q10. What is Quadratic Discriminant Analysis (QDA) and how does it differ from LDA?
**Answer:**  
QDA is also a generative Gaussian classifier, but it relaxes the equal covariance assumption.  
- In QDA, each class $k$ has its **own separate covariance matrix** ($\Sigma_k$).
- Because $\Sigma_1 \neq \Sigma_0$, the quadratic terms do not cancel out, resulting in a **quadratic (curved) decision boundary**.

#### Q11. What is the Bias-Variance trade-off between LDA and QDA?
**Answer:**  
- **Parameter Count:** Estimating a $p \times p$ covariance matrix requires $\frac{p(p+1)}{2}$ parameters.
  - With $p = 11$, a single covariance matrix has $\frac{11 \times 12}{2} = 66$ parameters.
  - **LDA** estimates 1 shared matrix = **66 covariance parameters**.
  - **QDA** estimates 2 class-specific matrices = **132 covariance parameters**.
- **Trade-off:**
  - LDA has **higher bias** (if the true covariance is unequal), but **much lower variance**.
  - QDA has **lower bias**, but **higher variance**, making it prone to overfitting when the sample size in any class is relatively small.

#### Q12. Why did LDA outperform QDA on our wine quality dataset?
**Answer:**  
In our dataset, high-quality wines ($\ge 7$) are a minority class (13.5%, only ~147 samples in the training set).  
Estimating 66 parameters of an $11 \times 11$ covariance matrix from only 147 samples introduces substantial estimation error and variance.  
LDA pools data across both classes to estimate a single, robust covariance matrix. This pooling acts as effective regularization, leading to better generalization on the test set:
- **LDA Test Accuracy:** **88.97%** (vs QDA: 85.29%)
- **LDA Test F1-Score:** **0.4828** (vs QDA: 0.4737)
- **LDA Test ROC-AUC:** **0.8855** (vs QDA: 0.8094)

#### Q13. How does LDA compare with Logistic Regression?
**Answer:**  
1. **Generative vs. Discriminative:** LDA models the joint distribution $P(X, Y) = P(X|Y)P(Y)$ (generative), whereas Logistic Regression models the conditional probability $P(Y|X)$ directly (discriminative).
2. **Assumptions:** LDA assumes multivariate normality; Logistic Regression makes no distribution assumption for $X$.
3. **When to choose which:**
   - If classes are well-separated, Logistic Regression coefficient estimates can become unstable (infinite), whereas LDA remains stable.
   - If predictors are approximately normal, LDA can be more efficient and require less training data.
   - If normality is severely violated or features are categorical, Logistic Regression is generally preferred.

#### Q14. Why is Accuracy alone misleading on this dataset, and which metrics should we use?
**Answer:**  
Because 86.46% of the wines belong to the majority class (quality < 7), a naive dummy classifier that always predicts class 0 achieves **86.40% accuracy** while having zero predictive utility for detecting high-quality wines.  
Therefore, we evaluate models using:
- **Precision:** $\frac{TP}{TP + FP}$ (proportion of predicted premium wines that are genuinely premium).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ (proportion of actual premium wines detected).
- **F1-Score:** Harmonic mean of precision and recall ($2 \cdot \frac{P \cdot R}{P + R}$).
- **ROC-AUC:** Area under the ROC curve across all classification thresholds, measuring the model's discriminative ability independent of a specific probability cut-off.
