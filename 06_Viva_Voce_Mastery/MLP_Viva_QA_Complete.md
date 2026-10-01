# MLP Viva Preparation — Complete Q&A Guide (V3)
### Student: 22f2001153 | Competition: Heavy Equipment Selling Price Prediction
### Metric: RMSLE | Best Score: **0.18732** | Models: LightGBM 1 (25%) + LightGBM 2 (45%) + CatBoost (30%)

---

> **HOW TO USE THIS GUIDE:** This document contains a comprehensive collection of viva questions compiled from senior feedback, official handouts, and our notebook iterations. Questions marked **[HIGH FREQ]** or **[SENIOR QA]** are the most common questions asked by examiners.

---

## SECTION 1: PROBLEM STATEMENT & PROJECT OVERVIEW

### Q1. [HIGH FREQ] What is the problem statement of your project?
The project is **Heavy Equipment Selling Price Prediction** on Kaggle.
- **Objective:** Predict the auction selling price (`TargetValue` in USD) of heavy construction equipment (bulldozers, excavators, skid steers) based on historical auction records.
- **Type:** Supervised Machine Learning — **Regression** (predicting continuous dollar value)
- **Training Data:** 138,701 records with 50 features (includes actual auction price)
- **Test Data:** 15,000 records without prices (we submit our predictions to Kaggle)
- **Evaluation Metric:** RMSLE (Root Mean Squared Log Error) — lower is better
- **Best Score:** **0.18732** (v18 notebook)

### Q2. [HIGH FREQ] What type of ML problem is this — Regression or Classification? Why?
This is a **Regression** problem because the target variable (`TargetValue`) is a continuous numerical value (selling price in USD), which can take any positive real value (e.g. $7,500 to $142,000). Classification is used for discrete categories (e.g. spam/no-spam). Bucketing prices would lose the exact dollar precision required.

### Q3. What is the evaluation metric and why is it used?
The metric is **RMSLE (Root Mean Squared Log Error)**:
$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_{\text{pred}}) - \log(1 + y_{\text{true}}))^2}$$
Why used instead of RMSE or MAE:
1. **Scale Independence:** Equipment prices range from $7,500 to $142,000 (19x range). RMSE penalizes errors on expensive machines disproportionately. RMSLE penalizes percentage errors equally: a 10% error on a $10,000 machine gets the same penalty as a 10% error on a $100,000 machine.
2. **Log-transformation safety:** The log1p function (log(1+x)) handles zeros safely, preventing undefined log(0) values.

### Q4. Describe the dataset features.
The dataset has 50 columns containing:
- **Transaction info:** `TransactionID`, `TransactionDate`, `AuctioneerID`
- **Machine characteristics:** `YearMade`, `MachineHoursCurrentMeter` (operational hours), `ProductConfigID`, `InventoryGroupDescription`, `UtilizationTier`, `AssetScaleFactor`
- **Specifications:** Over 35 features detailing cabin type, engine horsepower, drive system, etc. (mostly high-missing).

---

## SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)

### Q5. [HIGH FREQ] What EDA did you perform and what insights did you find?
1. **Shape check:** Train: 138,701 x 50, Test: 15,000 x 49.
2. **Missing value check:** Found that many technical specifications had 70%+ missing values. `col18` and `col19` had 99.95% missing, so they were dropped.
3. **Target distribution:** Highly right-skewed. Applied `log1p(TargetValue)` to normalize the distribution.
4. **Asset Overlap:** Discovered that **21.4% of machines in the test set** appeared in the training set (identified by identical `AssetID`). This meant we could leverage price history.
5. **Feature correlations:** Newer year made and enclosed cabin type correlated with higher prices.

### Q6. What is a Correlation Matrix and what did you infer?
A correlation matrix shows the Pearson correlation coefficient ($r$) between all pairs of numerical variables, ranging from -1 to +1. We found:
- `ManufactureYear` has a moderate positive correlation (+0.20) with price.
- `OperationalHoursMeter` has a near-zero linear correlation with price, but a highly non-linear relationship (trees can capture this, linear models cannot).

### Q7. [SENIOR QA] How can you find correlation between categorical features?
Pearson correlation is only for continuous variables. To find correlation between categorical features:
1. **Chi-Square Test of Independence:** Tests if there is a significant association between two categories.
2. **Cramer's V:** A metric derived from Chi-Square that ranges from 0 (no association) to 1 (perfect association).

### Q8. What is a Box Plot and what is IQR?
A Box Plot visualizes the distribution of data based on a five-number summary: minimum, first quartile (Q1), median (50th percentile), third quartile (Q3), and maximum. The **IQR (Interquartile Range)** is Q3 - Q1, which represents the middle 50% of the data. Outliers are plotted as individual points beyond $1.5 \times \text{IQR}$ from the quartiles.

### Q9. What are Outliers? How did you handle them?
Outliers are data points that lie far away from the majority of observations. In our project:
- `ManufactureYear` <= 1920: Set to NaN because modern heavy equipment did not exist then (data entry errors).
- `OperationalHoursMeter` == 0: Set to NaN because zero hours on an auctioned used machine implies a broken or reset meter.
- **Hours Outlier Clipping (v17):** Clipped `OperationalHoursMeter` at the 99th percentile to prevent extreme values from distorting splits.
- We did not drop outlier prices since auction prices can legitimately range from $7,500 to $142,000. Instead, we used log-transformation to compress their scale.

---

## SECTION 3: DATA PREPROCESSING

### Q10. [HIGH FREQ] What is Data Preprocessing? What steps did you take?
Data preprocessing prepares raw data for modeling. Our steps:
1. **Cleaning:** Handled bad entries (years <= 1920, hours == 0) by converting them to NaN.
2. **Missing value imputation:** Medians for numericals; 'Missing' string for categoricals.
3. **Encoding:** OrdinalEncoder for ordered categories; K-Fold OOF Target Encoding for high-cardinality columns.
4. **Log transformation:** `y = log1p(TargetValue)` to stabilize variance and normalize target.

### Q11. [HIGH FREQ] What are the types of missing value imputation? Why use median?
- **Numerical:** Mean, Median, Mode, constant value, or forward/backward fill. We used **Median Imputation** because median is robust to outliers, whereas mean gets pulled by extremely high prices or hours.
- **Categorical:** Mode (most frequent) or filling with a constant class like 'Missing'. We filled with 'Missing' to prevent replacing unique specification categories with the most common one.

### Q12. [HIGH FREQ] What is Feature Scaling? Why did or didn't you use it?
Feature scaling (e.g. `StandardScaler` or `MinMaxScaler`) normalizes the range of features. We used `StandardScaler` in our v1 Ridge baseline because linear regression coefficients are sensitive to scale mismatch. However, for our final ensemble (LightGBM & CatBoost), scaling is **not required**. Tree-based models split features using rank order (thresholds like 'Age > 15') and are scale-invariant.

### Q13. [SENIOR QA] What is the difference between StandardScaler, MinMaxScaler, and RobustScaler?
- **StandardScaler:** Centering and scaling using: $x_{\text{scaled}} = (x - \mu) / \sigma$. Assumes normally distributed data.
- **MinMaxScaler:** Scales data to a fixed range (usually [0, 1]) using: $x_{\text{scaled}} = (x - x_{\text{min}}) / (x_{\text{max}} - x_{\text{min}})$. Very sensitive to outliers.
- **RobustScaler:** Scales data using median and IQR: $x_{\text{scaled}} = (x - \text{median}) / \text{IQR}$. Highly robust to outliers because it does not use mean or variance.

### Q14. One-Hot Encoding vs. Ordinal Encoding vs. Target Encoding?
- **One-Hot Encoding (OHE):** Creates a binary column for each category. Bad for high cardinality (creates thousands of columns, causing sparsity and memory overload).
- **Ordinal Encoding:** Converts categories to integers (0, 1, 2...). Fine for tree models, but assumes order if used in linear models.
- **Target Encoding (TE):** Replaces category with the average target price. Excellent for high cardinality, but prone to leakage unless OOF is used.

### Q15. What is a Pipeline and ColumnTransformer in scikit-learn?
- **Pipeline:** Sequentially chains multiple transformer steps and an estimator (e.g., Imputer -> Scaler -> Model). Prevents data leakage since it ensures fit() only learns from training folds.
- **ColumnTransformer:** Applies different preprocessing pipelines to different columns (e.g. median imputer to numerical columns, and ordinal encoder to categorical columns).

### Q16. What is the difference between fit(), transform(), and fit_transform()?
- `fit()`: Learns parameters (e.g., mean and variance for scaler, category mapping for encoder) from training data.
- `transform()`: Applies the learned parameters to scale or encode data. Used on both train and test.
- `fit_transform()`: Learns parameters and applies them in a single step. ONLY used on training data, never on test data.

---

## SECTION 4: FEATURE ENGINEERING

### Q17. [HIGH FREQ] What Feature Engineering did you perform? Justify each.
1. **MachineAge:** `SaleYear - ManufactureYear`. Captures physical depreciation over time.
2. **UsagePerYear:** `OperationalHoursMeter / (MachineAge + 1)`. Normalizes usage intensity.
3. **Asset History (v10):** Prev mean price, prev last price, days since last sale, and prev sales count. Leverages historical resale values for the 21.4% overlapping machines.
4. **Cyclical Date Features (v16/v17):** Sine and cosine of SaleDayOfYear, SaleMonth, SaleDayOfWeek, and `SaleDay` (day of month, v17). Resolves the boundary jump (e.g., Day 31 vs. Day 1).
5. **Non-Linear Age (v16):** `MachineAge_sq` (rapid early depreciation) and `LogAge` (slow long-term price stabilization).
6. **LogHours x LogAge Interaction (v17):** Captures usage intensity relative to age (high hours on a young machine has a different meaning than on an old machine).
7. **Years_Since_2000 (v17):** Numeric trend capturing overall inflation/price movements since the year 2000.
8. **AgeBucket (v18):** Categorical grouping (New/Young/Mid/Old/Ancient) for high-density target encoding.
9. **Fold-Level Group Stats (v14):** Counts and price averages per category computed fold-by-fold to prevent scale mismatch.

### Q18. What is OOF Target Encoding and how does it prevent leakage?
Target encoding replaces category values with the average price of that category. Naive target encoding includes the current row's own price in the average, allowing the model to 'cheat' (leakage). Out-of-Fold (OOF) target encoding computes the category means using only the other 4 folds in a 5-fold CV loop. The current row's price is never included in its own encoding.

### Q19. What is target smoothing (alpha) and why is it useful?
Formula: `encoded = (n * category_mean + alpha * global_mean) / (n + alpha)`
Smoothing pulls category means toward the global average price. If a machine configuration appears only twice (n=2), its mean is unreliable. With `alpha=15`, the global mean dominates, preventing the model from overfitting to small sample sizes.

### Q20. What is your Density-Check philosophy for interaction target encodings? (v18 Innovation)
In v17, we created combinations like `ProductConfigID` x `SaleYear` that had 16,316 potential combination cells. Since we only have 138,701 training rows, this meant an average of only **8 samples per cell**. Most cells were empty or had 1 sample, making their target encoding pure noise. In v18, we implemented a **density check**: only use interaction target encodings where the average cell density is **greater than 30 (preferably >100) samples per cell** (such as `AgeBucket` x `SaleYear` with 1,981 samples/cell). This prevents the model from training on noise.

---

## SECTION 5: MODEL BUILDING & ALGORITHMS

### Q21. [HIGH FREQ] What models did you use and why?
We compared Ridge Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM, and CatBoost. Our final ensemble consists of:
- **LightGBM 1 (25% weight):** Tuned with `num_leaves=1023` to handle high-dimensional interaction features.
- **LightGBM 2 (45% weight):** Tuned with `num_leaves=511` and different random seed (123) for prediction diversity.
- **CatBoost (30% weight):** Provides excellent categorical feature handling and Ordered Boosting diversity.

### Q22. [HIGH FREQ] How does LightGBM work? Key hyperparameters?
LightGBM is a gradient boosting framework that grows trees **leaf-wise** (splitting the leaf that reduces loss most) rather than level-wise. It uses Histogram-based decision tree learning and GOSS (Gradient-based One-Side Sampling) for speed.
Key hyperparameters tuned:
- `num_leaves=1023`: Max leaves per tree in LGB1 (controls model capacity).
- `learning_rate=0.010`: Slower rate for precise optimization.
- `reg_lambda=3.0`: L2 regularization to penalize large weights.

### Q23. Why did you drop XGBoost and use two LightGBMs instead?
XGBoost consistently received 0% blend weight because its predictions were highly correlated with LGB1 — it did not add diversity. Instead of wasting resources on it, we replaced it with a second LightGBM (LGB2) with a different seed (123) and different hyperparameters. LGB2 captured slightly different patterns, helping to lower our OOF score.

### Q24. [SENIOR QA] How does a Decision Tree split? Explain Entropy, Information Gain, Gini.
A decision tree splits nodes to maximize purity:
1. **Entropy:** Measures impurity/randomness: $H(X) = -\sum p_i \log_2(p_i)$. Range [0, 1].
2. **Information Gain:** The reduction in entropy after a split: $IG = H(\text{parent}) - H(\text{children})$. The tree splits on the feature maximizing $IG$.
3. **Gini Impurity:** Probability of misclassifying a random element: $G = 1 - \sum p_i^2$. Range [0, 0.5] for binary class. Faster to compute than entropy because it doesn't require logs.

### Q25. [HIGH FREQ] Bagging vs. Boosting?
- **Bagging (e.g. Random Forest):** Builds multiple independent trees in parallel using bootstrap samples. Reduces **variance** (prevents overfitting).
- **Boosting (e.g. LightGBM):** Builds trees sequentially, where each new tree corrects the errors (residuals) of the previous trees. Reduces **bias** (improves learning).

### Q26. What is the Bias-Variance tradeoff?
- **Bias:** Error from simple assumptions (underfitting). Ridge Regression has high bias (RMSLE=0.3800).
- **Variance:** Error from sensitivity to small fluctuations in training data (overfitting). A decision tree with infinite depth has high variance.
- **Ensembles:** Combining models balances both, providing low bias and low variance.

---

## SECTION 6: CROSS VALIDATION & HYPERPARAMETER TUNING

### Q27. [HIGH FREQ] What is K-Fold Cross Validation? How did you use it?
K-Fold splits the dataset into K equal folds. For each fold: train on K-1 folds, validate on the remaining fold, and repeat K times. We used **5-Fold CV** with shuffling and `random_state=42`. All predictions on test data were averaged across the 5 trained folds (fold-averaging), which makes predictions highly stable.

### Q28. [SENIOR QA] What is the difference between train_test_split stratify=True and False?
- **stratify=None:** Splits randomly. The class proportions in train and test sets may differ from the original data.
- **stratify=y:** Ensures train and test sets have the **exact same proportion** of classes as the original target variable `y` (essential for classification, not used in regression).

### Q29. GridSearchCV vs. RandomizedSearchCV?
- **GridSearchCV:** Evaluates every possible combination of hyperparameters. Guaranteed to find the best parameters, but extremely slow.
- **RandomizedSearchCV:** Evaluates a fixed number of random parameter combinations from defined distributions. Much faster and finds near-optimal results with far fewer fits.

### Q30. What is Overfitting and Underfitting? How did you handle them?
- **Underfitting:** Model is too simple. Handled by switching from Ridge Regression to gradient boosted trees.
- **Overfitting:** Model memorizes training data. Handled by K-Fold CV, early stopping (stopping training if validation RMSLE doesn't improve for 300 rounds), L1/L2 tree regularization, and density-checking target encodings.

---

## SECTION 7: MODEL EVALUATION METRICS

### Q31. What is R-squared ($R^2$)? Can it be negative?
$R^2$ (Coefficient of Determination) measures the proportion of target variance explained by the model. Range is $(-\infty, 1]$. Yes, $R^2$ can be **negative** if the model's predictions are worse than simply predicting the global average value for all rows.

### Q32. [SENIOR QA] Explain Precision, Recall, F1-Score, and Support.
Used in classification reports:
- **Precision:** Out of all predicted positives, how many were actually positive? $TP / (TP + FP)$.
- **Recall:** Out of all actual positives, how many did we predict positive? $TP / (TP + FN)$.
- **F1-Score:** Harmonic mean of precision and recall: $2 \times (\text{Precision} \times \text{Recall}) / (\text{Precision} + \text{Recall})$.
- **Support:** The actual number of occurrences of each class in the specified dataset.

### Q33. [SENIOR QA] What is the difference between Macro Average and Weighted Average?
- **Macro Average:** Computes the metric independently for each class and takes the unweighted mean (treats all classes equally).
- **Weighted Average:** Computes the metric for each class and takes the average weighted by class support (gives more weight to common classes).

### Q34. [SENIOR QA] How do you handle highly imbalanced datasets?
1. **Resampling:** Under-sampling majority class, or over-sampling minority class (using **SMOTE** - Synthetic Minority Over-sampling Technique).
2. **Loss Weighting:** Adjusting class weights in model fit (e.g. `class_weight='balanced'`).
3. **Metrics:** Use Precision-Recall AUC (PR-AUC) or F1-score instead of Accuracy (which is misleading for imbalanced data).

---

## SECTION 8: NEURAL NETWORKS & DEEP LEARNING

### Q35. [SENIOR QA] What is an Activation Function? Sigmoid vs. Tanh vs. ReLU?
Activation functions introduce non-linearity into neural network nodes, allowing them to learn complex non-linear mappings:
- **Sigmoid:** Maps values to range **(0, 1)**. Used in binary classification output layer.
- **Tanh (Hyperbolic Tangent):** Maps values to range **(-1, 1)**. Centered around zero, making optimization faster than sigmoid.
- **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. Extremely fast to compute, prevents vanishing gradients, but can suffer from 'dying ReLU'.

### Q36. What is the vanishing gradient problem?
During backpropagation, gradients are multiplied chain-wise back through layers. For activation functions like Sigmoid or Tanh, the derivative is small (< 0.25). Multiplying many small values causes gradients to shrink exponentially toward zero in early layers, stopping the weights from updating. ReLU solves this because its derivative is 1 for positive inputs.

---

## SECTION 9: LIVE CODING QUESTIONS

### Q37. [HIGH FREQ] Code the MNIST Classification Pipeline with and without PCA.
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# 1. Load MNIST (subset for speed)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data[:10000], mnist.target[:10000]

# 2. Train-test split (50:50, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 3. Without PCA
pipe_no_pca = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
pipe_no_pca.fit(X_train, y_train)
print('--- WITHOUT PCA ---')
print(classification_report(y_test, pipe_no_pca.predict(X_test)))

# 4. With PCA
pipe_with_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),  # Reduce dimensionality to 100 components
    ('svm', SVC())
])
pipe_with_pca.fit(X_train, y_train)
print('--- WITH PCA ---')
print(classification_report(y_test, pipe_with_pca.predict(X_test)))
```

### Q38. [SENIOR QA] Code ColumnTransformer with different preprocessing pipelines.
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = ['MachineAge', 'UsagePerYear']
cat_cols = ['InventoryGroupDescription', 'UtilizationTier']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
```

### Q39. [SENIOR QA] Create a DataFrame with only object columns, filter by unique values <= 10, and OneHotEncode them.
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

cat_df = df.select_dtypes(include=['object']).copy()
cols_to_encode = [col for col in cat_df.columns if cat_df[col].nunique() <= 10]
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_arr = encoder.fit_transform(cat_df[cols_to_encode])
encoded_cols = encoder.get_feature_names_out(cols_to_encode)
encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=cat_df.index)
```

### Q40. [SENIOR QA] Copy training data and split into numerical and categorical lists.
```python
X_train_copy = X_train.copy()
num_cols = X_train_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train_copy.select_dtypes(include=['object', 'category']).columns.tolist()
```

### Q41. [SENIOR QA] Code Recursive Feature Elimination (RFE).
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(solver='liblinear', random_state=42)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_].tolist()
```

---

## SECTION 10: PROJECT-SPECIFIC Q&A

### Q42. [HIGH FREQ] What is your best score and how did you achieve it?
Our best score is **0.18732** (v18 notebook). We achieved it through:
1. **5-Fold Cross Validation** with fold-level feature generation (preventing data leakage).
2. **Leak-free history features** to map past resales (21.4% test overlap).
3. **Density-checked target encoding** to prevent training on noise.
4. **Cyclical encoding** of date boundaries.
5. **Ensembling** LightGBM 1 (25%, 1023 leaves) + LightGBM 2 (45%, 511 leaves) + CatBoost (30%).

### Q43. Explain the v12/v13 leakage bug and how it was fixed.
In v12/v13, we computed category frequency encoding globally across all training rows. This caused a **scale mismatch** because the training set (138K rows) has a 9x higher absolute scale than the test set (15K). The model trained expecting large frequency numbers but saw small ones in test. We fixed it in v14 by computing frequency maps only on the active training fold and mapping them fold-by-fold.

### Q44. Why is your Kaggle score (0.18732) better than your cross-validation OOF score (0.19326)?
1. **Ensemble Smoothing:** The final Kaggle predictions are generated by averaging predictions across 5 folds for each model. This acts as a variance-reducing bagging ensemble, yielding a cleaner prediction than OOF.
2. **Asset History Power:** Asset history features work extremely well on test data due to the high resale overlap.

---

*Document End — Version 3 Complete*
