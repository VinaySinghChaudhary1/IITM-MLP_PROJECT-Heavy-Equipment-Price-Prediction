# IIT-M BS Degree | Machine Learning Practice (MLP) Course
# 🎓 COMPREHENSIVE VIVA PREPARATION MASTER GUIDE
### Grounded in Project: Heavy Equipment Selling Price Prediction (Kaggle Regression)
### Student ID: 22f2001153 | Best Kaggle Score: **0.18732** (RMSLE)

---

> [!NOTE]
> This guide contains the **278 meaningful, deduplicated questions** extracted from the three senior PDFs. Each entry has the PDF's general answer followed by a **Project-Specific Viva Answer**. The project-specific text was checked against the notebook journey from the baseline through v20. It deliberately distinguishes between (a) methods actually used, (b) experiments that were rejected, and (c) concepts that were not used but may still be asked in the viva.

> **Use this in the viva:** answer the project-specific part first. Use the general PDF answer only if the inspector asks for the definition or mathematics. Similar questions are intentionally retained because inspectors may phrase the same concept differently; the answers cross-reference the same verified project facts instead of inventing separate experiments.


## Verified Project Facts - Learn These First

| Topic | Verified answer |
|---|---|
| Problem | Supervised regression: predict heavy-equipment auction selling price (`TargetValue`). |
| Data | Train: 138,701 rows x 50 columns; test: 15,000 rows x 49 columns. |
| Metric | RMSLE; the target is trained as `log1p(TargetValue)` and predictions are converted with `expm1()`. Lower is better. |
| Best submitted score | **0.18732 RMSLE, v18**. Later v19 (0.18777) and v20 (0.18770) did not improve it. |
| Final best architecture | 5-fold OOF blend of LightGBM-1, LightGBM-2 (different seed/feature sampling), and CatBoost. The exact blend weights are selected from OOF predictions for the version. |
| Key validated improvements | v1 baseline ~0.2050; v6 0.19086 (3-model ensemble + 5-fold CV); v10 0.19020 (leak-safe asset history + OOF TE); v14 0.18964 (fold-level encodings); v15 0.18847 (expanded TE + second LGB); v16 0.18763 (cyclical dates/nonlinear age); v18 0.18732 (dense interaction TEs). |
| Most important caution | All target encodings, frequency mappings and target-based group statistics are built using training-fold data only. Naive/global versions caused leakage or poor generalisation in earlier experiments. |

---

## 📋 Table of Contents
1. [Introduction & Project Overview](#introduction--project-overview)
2. [Exploratory Data Analysis (EDA) & Visualization](#exploratory-data-analysis-eda--visualization)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering & Leakage Prevention](#feature-engineering--leakage-prevention)
5. [Machine Learning Models & Algorithms](#machine-learning-models--algorithms)
6. [Hyperparameter Tuning & Cross-Validation](#hyperparameter-tuning--cross-validation)
7. [Model Evaluation Metrics & Statistics](#model-evaluation-metrics--statistics)
8. [Neural Networks & Deep Learning](#neural-networks--deep-learning)
9. [Scenario & Business Case Studies](#scenario--business-case-studies)
10. [Live Coding Challenges](#live-coding-challenges)
11. [Miscellaneous & Tricky Questions](#miscellaneous--tricky-questions)

---

## 📁 Introduction & Project Overview
This section covers 7 questions related to introduction & project overview.

### Q1. What is the objective/problem statement of your project?

**Original Answer (from PDF):**
Clearly state what you are predicting (target variable), whether it is a classification or regression problem, and why it matters. E.g., 'We predict X using Y features; this is a binary classification problem because the target has two classes.'

**Project-Specific Viva Answer:**
- I predict `TargetValue`, the auction selling price of heavy construction equipment. It is supervised regression because price is continuous. I trained on 138,701 records (50 columns) and generated predictions for 15,000 test records; Kaggle evaluates them with RMSLE.

---

### Q2. How should you introduce yourself in a viva?

**Original Answer (from PDF):**
Keep it concise (1–2 min): your name, academic background (degree, year), why you chose MLP, and one sentence on what interested you most in the project. End with readiness to walk through your notebook.

**Project-Specific Viva Answer:**
- Introduce yourself: State your name, degree, and explain that your MLP project predicted heavy construction equipment auction prices on Kaggle (regression task). Ground it by stating that your final solution achieved a score of **0.18732** using an ensemble of dual LightGBMs (seeds 42 & 123) and CatBoost, trained via 5-Fold cross-validation.

---

### Q3. What would you do if you had more time on the project?

**Original Answer (from PDF):**
Mention concrete improvements: try more feature engineering (e.g., interaction terms, domain-specific features), experiment with additional models (StackingClassifier, Neural Networks), perform more thorough EDA, use automated HPT tools (Optuna/Bayesian optimisation), perform error analysis on misclassified samples, or tune TF-IDF parameters more rigorously.

**Project-Specific Viva Answer:**
- I would first do disciplined error analysis and only retain features that improve OOF and Kaggle performance. I would test more leak-safe, sufficiently dense interactions, run a constrained hyperparameter search for the existing allowed models, and compare a time-aware split as a robustness check. I would not claim that I used Optuna, TabNet, or an AutoML system: they were not part of this submitted project and may be outside the course's allowed-library rules.

---

### Q4. What resources did you use for this project?

**Original Answer (from PDF):**
Acceptable answers: sklearn documentation, Kaggle notebooks/discussions, YouTube tutorials, MLT/MLF course material, research papers. Avoid saying you just used AI/LLM — mention you used it as a helper to understand concepts, not to copy code blindly.

**Project-Specific Viva Answer:**
- The implementation uses Python, NumPy, pandas, scikit-learn, LightGBM and CatBoost, together with the course/Kaggle material. In a viva I would emphasize that I used documentation and guidance to understand ideas, then verified each change through notebook outputs and OOF/Kaggle scores.

---

### Q5. What other approaches did you try that did not work?

**Original Answer (from PDF):**
Be honest. E.g., 'I tried adding polynomial features but it caused overfitting. I tried CountVectorizer before TF-IDF but TF-IDF gave better results because it down-weights common words. I tried StandardScaler on skewed data and it did not help, so I switched to RobustScaler.'

**Project-Specific Viva Answer:**
- Several experiments were useful failures: naive/global target encoding leaked target information (v8); globally calculated frequency/group statistics created inconsistent train/test behavior (v12/v13); sparse time interactions in v17 were not reliable; and v19 SaleMonth interaction encodings worsened both OOF (0.19326 to 0.19354) and Kaggle (0.18732 to 0.18777), indicating seasonal overfitting. v20 also did not beat v18. The lesson is to use fold-safe features and keep only features that generalise.

---

### Q6. What is your Kaggle rank/leaderboard score?

**Original Answer (from PDF):**
Know your exact public and private leaderboard score, your rank among participants, and which version of your notebook achieved the best score. Also know what specific change (model, feature, HPT) gave the biggest improvement.

**Project-Specific Viva Answer:**
- The verified best submitted Kaggle score is **0.18732 RMSLE from v18**; lower is better. I would not invent a rank if it is not shown on the leaderboard. The score progression is more important: baseline ~0.2050, then 0.19086, 0.19020, 0.18964, 0.18847, 0.18763, and finally 0.18732.

---

### Q7. What is the XGBoost objective/metric parameter ('xgboost matrix')?

**Original Answer (from PDF):**
XGBoost's objective parameter specifies the learning task and loss function: 'binary:logistic': binary classification, outputs probability. 'multi:softmax': multi-class, outputs class index. 'multi:softprob': multi-class, outputs probabilities. 'reg:squarederror': regression with MSE. 'reg:logistic': regression with log loss. eval_metric specifies the evaluation metric used for validation: 'logloss', 'mlogloss', 'error', 'rmse', 'auc'. These are set in XGBClassifier(objective=..., eval_metric=...).

**Project-Specific Viva Answer:**
- Our objective is predicting the auction selling price (`TargetValue` in USD) of heavy construction machinery. This is a supervised regression problem, evaluated on Kaggle using RMSLE. Our training dataset has 138,701 rows and 50 columns; the test dataset has 15,000 rows.

---

## 📁 Exploratory Data Analysis (EDA) & Visualization
This section covers 19 questions related to exploratory data analysis (eda) & visualization.

### Q8. What is your best model and why does it perform better?

**Original Answer (from PDF):**
Name your best model, give its score, and compare it against baseline/other models. Explain why it performs better — e.g., tree-based models handle non-linearity well, boosting corrects previous errors, etc. Reference train vs. validation scores to show it is not overfitting.

**Project-Specific Viva Answer:**
- The best submitted system is v18: a 5-fold OOF blend of two LightGBM models and CatBoost, with Kaggle RMSLE **0.18732** and blended OOF RMSLE **0.19326**. Gradient boosting captures nonlinear effects such as age, configuration and region; the two LightGBMs have different randomness/feature sampling and CatBoost handles categoricals differently, so averaging reduces correlated errors. The blend weights are searched on OOF predictions, not guessed.

---

### Q9. What is Exploratory Data Analysis (EDA) and why do we do it?

**Original Answer (from PDF):**
EDA is the process of visually and statistically summarising a dataset before modelling. We do it to understand distributions, detect missing values and outliers, find correlations between features, check class balance, and form hypotheses about which features matter most.

**Project-Specific Viva Answer:**
- In our project's EDA: (1) we checked the shape (138,701 train, 15,000 test), (2) checked missing values (specs had 70%+ missing, so we dropped col18 and col19 and imputed others), (3) analyzed target price skewness (log-transformed to normal), and (4) discovered a 21.4% machine resale overlap (AssetID) which we utilized for resale history mapping.

---

### Q10. What is a correlation matrix and how do you read it?

**Original Answer (from PDF):**
A correlation matrix shows pairwise Pearson (or Spearman) correlation coefficients between numeric features, ranging from –1 (perfect negative) to +1 (perfect positive). Values near 0 mean no linear relationship. You use it to drop highly correlated redundant features and to identify predictors strongly related to the target.

**Project-Specific Viva Answer:**
- We computed a Pearson correlation matrix for our numerical features during EDA. We found that `ManufactureYear` has a moderate linear correlation (+0.20) with price. However, `OperationalHoursMeter` showed a near-zero linear correlation, but a highly non-linear relationship (depreciation plateaus after a certain number of hours), which linear models like Ridge could not capture but tree models split effectively.

---

### Q11. What is a box plot (box-and-whisker plot)?

**Original Answer (from PDF):**
A box plot displays a feature's quartiles: the box spans Q1 to Q3 (IQR), the line inside is the median, and the whiskers extend to 1.5×IQR. Points beyond the whiskers are plotted as outliers. It quickly shows spread, skew, and outlier presence.

**Project-Specific Viva Answer:**
- We used Seaborn box plots to analyze categorical variables like `InventoryGroupDescription` and `UtilizationTier` against log-price. For instance, Wheel Loaders showed a median log-price around 10.7 ($44K), while Skid Steer Loaders had a median around 9.6 ($15K). This confirmed distinct price distributions that tree splits could leverage.

---

### Q12. What is univariate and bivariate analysis?

**Original Answer (from PDF):**
Univariate analysis studies one variable at a time (histograms, box plots, descriptive stats). Bivariate analysis studies the relationship between two variables (scatter plots, correlation, cross-tabulations). Together they reveal individual distributions and inter-feature relationships.

**Project-Specific Viva Answer:**
- Univariate analysis in our project included plotting histograms of price (log scale) and machine age. Bivariate analysis included plotting scatter plots of `MachineAge` vs. price, which revealed a clear non-linear depreciation curve (prices drop rapidly in the first 3 years and then stabilize).

---

### Q13. What does a histogram tell you?

**Original Answer (from PDF):**
A histogram shows the frequency distribution of a numeric variable. You can infer whether the distribution is normal, skewed (left/right), bimodal, or uniform, and spot outliers or data-entry errors as isolated bars far from the main mass.

**Project-Specific Viva Answer:**
- We plotted histograms of `TargetValue` and `OperationalHoursMeter`. The price histogram showed a severe right skew. The log-transformed price histogram, however, showed a clean, near-normal bell curve, validating our target log-transformation strategy.

---

### Q14. What are outliers and how do you detect and handle them?

**Original Answer (from PDF):**
Outliers are data points far from the bulk of the distribution. Detection methods: IQR method (flag values beyond Q1 − 1.5×IQR or Q3 + 1.5×IQR), Z-score (|z| > 3), and visual inspection (box plots, scatter plots). Handling: remove them (if erroneous), cap/clip (Winsorising), transform (log), use robust models (tree-based), or use RobustScaler.

**Project-Specific Viva Answer:**
- We identified and treated outliers during EDA: (1) ManufactureYear <= 1920 (data entry errors) and OperationalHoursMeter == 0 (reset/broken meters) were set to NaN. (2) Equipment prices range naturally up to $142,000, so instead of deleting high-price rows, we used target log-transformation and price prediction clipping to [Q0.1%, Q99.9%] ($7,500 to $137,500) to protect against extreme predictions.

---

### Q15. What is skewness and how does it affect your model?

**Original Answer (from PDF):**
Skewness measures the asymmetry of a distribution. Right-skewed (positive): long tail to the right, mean > median. Left-skewed (negative): long tail to the left. Skewed distributions can hurt linear models and distance-based models. Solutions: log transform, square-root transform, Box-Cox transform to make it more normal.

**Project-Specific Viva Answer:**
- In our dataset, the target variable `TargetValue` (selling price) is highly right-skewed, ranging from $7,500 to $142,000. Right-skewness causes tree models to over-prioritize rare expensive machines. We resolved this by applying a `log1p(TargetValue)` transformation to normalize the distribution, and reversed predictions with `expm1()` at inference time.

---

### Q16. What is a word cloud and where can it be used?

**Original Answer (from PDF):**
A word cloud is a visual where word size is proportional to its frequency in a text corpus. Used in NLP EDA to quickly identify the most common words in each class or the overall dataset. It helps spot domain-specific terms, stop words that should be removed, and any data quality issues.

**Project-Specific Viva Answer:**
- In our project's EDA: (1) we checked the shape (138,701 train, 15,000 test), (2) checked missing values (specs had 70%+ missing, so we dropped col18 and col19 and imputed others), (3) analyzed target price skewness (log-transformed to normal), and (4) discovered a 21.4% machine resale overlap (AssetID) which we utilized for resale history mapping.

---

### Q17. Why use both seaborn and matplotlib? What is the difference?

**Original Answer (from PDF):**
Matplotlib is the foundational plotting library — low-level, fully customisable, but verbose. Seaborn is built on top of matplotlib — higher-level, produces statistically-oriented plots (heatmaps, violin, pair plots) with less code and better default aesthetics. Use seaborn for quick statistical plots, matplotlib for fine-grained customisation.

**Project-Specific Viva Answer:**
- In our project's EDA: (1) we checked the shape (138,701 train, 15,000 test), (2) checked missing values (specs had 70%+ missing, so we dropped col18 and col19 and imputed others), (3) analyzed target price skewness (log-transformed to normal), and (4) discovered a 21.4% machine resale overlap (AssetID) which we utilized for resale history mapping.

---

### Q18. What does .describe() tell you?

**Original Answer (from PDF):**
df.describe() returns count, mean, std, min, 25th percentile (Q1), 50th percentile (median), 75th percentile (Q3), and max for each numeric column. It gives a quick statistical summary to spot outliers (compare min/max to Q1/Q3), check for missing values (count < total rows), and understand the scale of each feature.

**Project-Specific Viva Answer:**
- We used `df.describe()` to inspect our numeric columns. It revealed that `OperationalHoursMeter` had a max of over 20,000 hours, a median of 3,200 hours, and many 0 values (which we converted to NaN as outliers representing broken meters). It also confirmed the 19x range in equipment prices.

---

### Q19. What is the IQR method for outlier detection?

**Original Answer (from PDF):**
IQR = Q3 − Q1. Lower bound = Q1 − 1.5 × IQR. Upper bound = Q3 + 1.5 × IQR. Values outside these bounds are considered outliers. The 1.5 factor is standard (Tukey's rule); using 3.0 detects only extreme outliers.

**Project-Specific Viva Answer:**
- We identified and treated outliers during EDA: (1) ManufactureYear <= 1920 (data entry errors) and OperationalHoursMeter == 0 (reset/broken meters) were set to NaN. (2) Equipment prices range naturally up to $142,000, so instead of deleting high-price rows, we used target log-transformation and price prediction clipping to [Q0.1%, Q99.9%] ($7,500 to $137,500) to protect against extreme predictions.

---

### Q20. What is mean and standard deviation?

**Original Answer (from PDF):**
Mean = (sum of all values) / n — the average, sensitive to outliers. Standard deviation σ = sqrt(mean of squared deviations from mean) — measures spread. A small σ means data is clustered around the mean; large σ means data is spread out. Used in StandardScaler: z = (x − mean) / σ.

**Project-Specific Viva Answer:**
- In our project's EDA: (1) we checked the shape (138,701 train, 15,000 test), (2) checked missing values (specs had 70%+ missing, so we dropped col18 and col19 and imputed others), (3) analyzed target price skewness (log-transformed to normal), and (4) discovered a 21.4% machine resale overlap (AssetID) which we utilized for resale history mapping.

---

### Q21. What is Pearson vs. Spearman correlation?

**Original Answer (from PDF):**
Pearson measures linear correlation between two continuous variables, assuming normality. Range: [−1, 1]. Spearman measures monotonic (rank-based) correlation — does not assume normality, works for ordinal data and non-linear monotonic relationships. Use Spearman when data is skewed or ordinal.

**Project-Specific Viva Answer:**
- In our project's EDA: (1) we checked the shape (138,701 train, 15,000 test), (2) checked missing values (specs had 70%+ missing, so we dropped col18 and col19 and imputed others), (3) analyzed target price skewness (log-transformed to normal), and (4) discovered a 21.4% machine resale overlap (AssetID) which we utilized for resale history mapping.

---

### Q22. Which model is better for outliers — Logistic Regression or SVM?

**Original Answer (from PDF):**
SVM (with RBF kernel) is generally more robust to outliers because the decision boundary is determined only by support vectors (points near the margin). Logistic Regression is sensitive to outliers as they can heavily influence the decision boundary. However, with a small C (strong regularisation), SVM can also be affected.

**Project-Specific Viva Answer:**
- We identified and treated outliers during EDA: (1) ManufactureYear <= 1920 (data entry errors) and OperationalHoursMeter == 0 (reset/broken meters) were set to NaN. (2) Equipment prices range naturally up to $142,000, so instead of deleting high-price rows, we used target log-transformation and price prediction clipping to [Q0.1%, Q99.9%] ($7,500 to $137,500) to protect against extreme predictions.

---

### Q23. patterns did EDA reveal (seasonality, demographic trends, outlier groups)? 3) What is the confidence of?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- We identified and treated outliers during EDA: (1) ManufactureYear <= 1920 (data entry errors) and OperationalHoursMeter == 0 (reset/broken meters) were set to NaN. (2) Equipment prices range naturally up to $142,000, so instead of deleting high-price rows, we used target log-transformation and price prediction clipping to [Q0.1%, Q99.9%] ($7,500 to $137,500) to protect against extreme predictions.

---

### Q24. What is the real name of a box plot?

**Original Answer (from PDF):**
Box-and-whisker plot (coined by statistician John Tukey). The 'box' shows the IQR (Q1 to Q3), the line inside is the median, and the 'whiskers' extend to 1.5×IQR. Points beyond whiskers are plotted individually as outliers.

**Project-Specific Viva Answer:**
- We used Seaborn box plots to analyze categorical variables like `InventoryGroupDescription` and `UtilizationTier` against log-price. For instance, Wheel Loaders showed a median log-price around 10.7 ($44K), while Skid Steer Loaders had a median around 9.6 ($15K). This confirmed distinct price distributions that tree splits could leverage.

---

### Q25. Is removing outliers always good practice? Can we do something else?

**Original Answer (from PDF):**
No — removing outliers is NOT always good practice. Outliers may be: 1) Valid extreme values (e.g., very high income earners — important for the model). 2) Informative patterns (fraud transactions are by definition 'outliers'). 3) Rare but real events in the tail distribution. Alternatives to removal: Capping/Winsorising (clip to 1st/99th percentile). Robust scaling (RobustScaler uses IQR). Log transformation (compresses extreme values). Robust models (tree-based models are naturally robust to outliers). Separate model for outlier segment. Only remove if the outlier is a data entry error (e.g., age = 999).

**Project-Specific Viva Answer:**
- We identified and treated outliers during EDA: (1) ManufactureYear <= 1920 (data entry errors) and OperationalHoursMeter == 0 (reset/broken meters) were set to NaN. (2) Equipment prices range naturally up to $142,000, so instead of deleting high-price rows, we used target log-transformation and price prediction clipping to [Q0.1%, Q99.9%] ($7,500 to $137,500) to protect against extreme predictions.

---

### Q26. What does '.head()' vs '.describe()' vs '.info()' show?

**Original Answer (from PDF):**
df.head(n): displays first n rows (default 5). Quick sanity check on data format. df.tail(n): last n rows. df.describe(): statistical summary of numeric columns: count, mean, std, min, Q1, median, Q3, max. df.info(): column names, non-null counts, dtypes, memory usage. Best for spotting missing values and wrong dtypes. df.shape: (rows, columns) tuple. df.dtypes: data type of each column.

**Project-Specific Viva Answer:**
- We used `df.describe()` to inspect our numeric columns. It revealed that `OperationalHoursMeter` had a max of over 20,000 hours, a median of 3,200 hours, and many 0 values (which we converted to NaN as outliers representing broken meters). It also confirmed the 19x range in equipment prices.

---

## 📁 Data Preprocessing
This section covers 25 questions related to data preprocessing.

### Q27. How can you find correlation between categorical features?

**Original Answer (from PDF):**
Use Chi-Square test of independence, Cramér's V, or convert categories to dummy variables and compute point-biserial correlation with the target. For ordinal categories, Spearman correlation is appropriate.

**Project-Specific Viva Answer:**
- General methods are a chi-square test of independence and Cramer's V; for category-to-continuous-target association, compare grouped target distributions or use ANOVA/eta-squared where appropriate. I did not use a categorical-correlation matrix as a final modelling step. Instead, I assessed category usefulness through EDA, target-encoding design, fold-safe validation, and feature importance/OOF results.

---

### Q28. What is data preprocessing and why is it needed?

**Original Answer (from PDF):**
Preprocessing transforms raw data into a form suitable for ML models. It includes handling missing values, removing/treating outliers, encoding categoricals, scaling numerics, and splitting data. Without it, models may produce biased or inaccurate results because most algorithms cannot handle missing data, mixed types, or vastly different feature scales.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q29. What is imputation and what are its techniques?

**Original Answer (from PDF):**
Imputation fills missing values so models can use the full dataset. Common strategies: mean (for normally distributed numeric), median (for skewed numeric or when outliers exist), mode/most_frequent (for categorical or skewed numeric), KNN imputer (uses similar rows), and constant (fills with a fixed value). sklearn's SimpleImputer covers the first three.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q30. When would you use mean vs. median vs. mode imputation?

**Original Answer (from PDF):**
Use mean when the feature is roughly normally distributed and has no significant outliers. Use median when the feature is skewed or has outliers (it is robust). Use mode (most_frequent) for categorical columns or highly discrete numerics.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q31. What is StandardScaler and how does it work?

**Original Answer (from PDF):**
StandardScaler (z-score normalisation) subtracts the mean and divides by the standard deviation, producing features with mean=0 and std=1. It is appropriate when the feature roughly follows a normal distribution. Formula: z = (x − µ) / σ.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q32. What is MinMaxScaler and how does it differ from StandardScaler?

**Original Answer (from PDF):**
MinMaxScaler rescales values to a fixed range [0, 1] using (x − min)/(max − min). It is sensitive to outliers. StandardScaler is not bounded but is more robust to outliers. Use MinMaxScaler when you need a bounded output; use StandardScaler for normally distributed data.

**Project-Specific Viva Answer:**
- We did not use `MinMaxScaler` in our final ensemble because tree models are scale-invariant. `MinMaxScaler` is highly sensitive to outliers (it compresses standard values if a huge outlier exists), which would be problematic for features like `OperationalHoursMeter` with extreme max values.

---

### Q33. What is RobustScaler and when should you use it?

**Original Answer (from PDF):**
RobustScaler uses the median and IQR instead of mean and std, making it robust to outliers. Use it when your numeric features have significant outliers that you do not want to remove but do not want to skew scaling.

**Project-Specific Viva Answer:**
- If we were using a distance-based or linear model, `RobustScaler` (scaling by median and IQR) would be preferred over `StandardScaler` because features like `OperationalHoursMeter` have extreme outliers that distort mean and variance. However, since our final models are tree-based (LightGBM/CatBoost), they are scale-invariant and neither scaler was needed.

---

### Q34. What is One-Hot Encoding (OHE) and when is it used?

**Original Answer (from PDF):**
OHE creates a binary column for each category. E.g., 'Color' with [Red, Blue, Green] becomes three columns. Use OHE for nominal (unordered) categorical features. The handle_unknown='ignore' parameter prevents errors on unseen categories at inference time.

**Project-Specific Viva Answer:**
- One-Hot Encoding was not used for high-cardinality features like `ProductConfigID` (3,594 unique classes) or `Spec_FullDescriptor` (3,505 unique classes) because it would create thousands of sparse binary columns, overloading memory. Instead, we used OrdinalEncoder for tree splits and Out-of-Fold (OOF) Target Encoding with Bayesian smoothing.

---

### Q35. What is Label Encoding and when should you use it?

**Original Answer (from PDF):**
Label Encoding assigns an integer to each category (e.g., Red=0, Blue=1, Green=2). It is appropriate only for ordinal features (natural order exists) or tree-based models that are not affected by magnitude. For nominal features in linear models, OHE is preferred because Label Encoding implies a false ordinal relationship.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q36. What is Ordinal Encoding and how does it differ from Label Encoding?

**Original Answer (from PDF):**
Ordinal Encoding also maps categories to integers but allows you to specify the order explicitly (e.g., Low=0, Medium=1, High=2). Label Encoding assigns arbitrary integers. Use Ordinal Encoding when the category order matters.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q37. What is a Pipeline in sklearn and why use it?

**Original Answer (from PDF):**
A Pipeline chains preprocessing steps and a model into a single estimator. It ensures the same transformations are applied consistently to training and test data, prevents data leakage (fit only on train), makes code cleaner, and enables hyperparameter tuning of the entire workflow with GridSearchCV.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q38. What is a ColumnTransformer?

**Original Answer (from PDF):**
ColumnTransformer applies different transformations to different columns simultaneously and concatenates the results. E.g., apply StandardScaler to numeric columns and OHE to categorical columns in one step before passing to a model.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q39. What is the difference between fit(), transform(), and fit_transform()?

**Original Answer (from PDF):**
fit() learns parameters (e.g., mean, std) from training data. transform() applies the learned transformation. fit_transform() does both in one step. IMPORTANT: always fit only on training data, then transform both train and test to prevent data leakage.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q40. What is SMOTE and when is it used?

**Original Answer (from PDF):**
SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples for the minority class by interpolating between existing minority samples. It is used to handle class imbalance. Apply SMOTE only on training data to avoid leakage.

**Project-Specific Viva Answer:**
- SMOTE synthesizes minority-class samples and is used for imbalanced **classification**, not ordinary regression. I did not use it because my target is continuous selling price, so there is no minority class to oversample. Using SMOTE here would be conceptually wrong.

---

### Q41. How do you find correlation between a continuous and a categorical feature?

**Original Answer (from PDF):**
Use point-biserial correlation (for binary categorical), ANOVA F-test (for multi-class categorical vs. continuous), or Eta-squared. For two categorical features: Chi-Square test or Cramér's V. These tell you whether the categorical grouping significantly differs in the numeric feature's distribution.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q42. What is the difference between fillna() and SimpleImputer?

**Original Answer (from PDF):**
fillna() is a pandas method for quick filling — it does not fit/transform and does not integrate with sklearn Pipelines. SimpleImputer is a sklearn transformer that fits (learns mean/median/mode from training data) and transforms — it integrates seamlessly with Pipeline and ColumnTransformer, preventing data leakage.

**Project-Specific Viva Answer:**
- `fillna()` is a pandas function that replaces NaNs globally without fitting, which can introduce leakage if applied before splitting. We used Scikit-learn's `SimpleImputer` in our pipelines: `strategy='median'` for numeric features (like hours) to remain robust to outliers, and `strategy='constant', fill_value='Missing'` for categoricals (like `CabinType` which has 91% missing values) to preserve the missing state as a distinct category.

---

### Q43. Why should you avoid manual preprocessing (fillna, manual encoding) in favour of?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- `fillna()` is a pandas function that replaces NaNs globally without fitting, which can introduce leakage if applied before splitting. We used Scikit-learn's `SimpleImputer` in our pipelines: `strategy='median'` for numeric features (like hours) to remain robust to outliers, and `strategy='constant', fill_value='Missing'` for categoricals (like `CabinType` which has 91% missing values) to preserve the missing state as a distinct category.

---

### Q44. Pipelines?

**Original Answer (from PDF):**
Manual preprocessing applied to the full dataset before splitting causes data leakage (test data info leaks into training). Pipelines enforce that fitting happens only on training data and transformation is consistently applied to both train and test. They also make the code reproducible, deployable, and easier to tune with GridSearchCV.

**Project-Specific Viva Answer:**
- We avoided manual preprocessing (like global `fillna` or global encoders) because it causes validation leakage. By wrapping our imputers and encoders inside Scikit-learn Pipelines and ColumnTransformers, we ensure that parameters (like category mappings and medians) are learned only from the active training fold and applied cleanly to validation/test sets.

---

### Q45. What is handle_unknown='ignore' in OneHotEncoder?

**Original Answer (from PDF):**
When the encoder encounters a category at inference time that it did not see during training, handle_unknown='ignore' outputs a zero vector for that category (all OHE columns = 0) instead of raising an error. This is essential for production systems where new categories may appear.

**Project-Specific Viva Answer:**
- One-Hot Encoding was not used for high-cardinality features like `ProductConfigID` (3,594 unique classes) or `Spec_FullDescriptor` (3,505 unique classes) because it would create thousands of sparse binary columns, overloading memory. Instead, we used OrdinalEncoder for tree splits and Out-of-Fold (OOF) Target Encoding with Bayesian smoothing.

---

### Q46. What is sparse_output=False in OneHotEncoder?

**Original Answer (from PDF):**
By default, OHE returns a sparse matrix (efficient memory storage with mostly zeros). sparse_output=False returns a dense numpy array, which is required by some downstream estimators or when you want to inspect the data as a regular array. In newer sklearn versions this parameter is set_output(transform='pandas').

**Project-Specific Viva Answer:**
- One-Hot Encoding was not used for high-cardinality features like `ProductConfigID` (3,594 unique classes) or `Spec_FullDescriptor` (3,505 unique classes) because it would create thousands of sparse binary columns, overloading memory. Instead, we used OrdinalEncoder for tree splits and Out-of-Fold (OOF) Target Encoding with Bayesian smoothing.

---

### Q47. What is forward fill (ffill) and backward fill (bfill) in imputation?

**Original Answer (from PDF):**
Used for ordered/time-series data. Forward fill propagates the last known value forward to fill NaN. Backward fill uses the next known value to fill backward. Appropriate when data has temporal order and nearby values are likely similar. Not suitable for randomly missing tabular data.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q48. What does class_weight='balanced' do in sklearn models?

**Original Answer (from PDF):**
It automatically adjusts class weights inversely proportional to class frequencies: weight for class i = n_samples / (n_classes × n_samples_in_class_i). This penalises misclassification of the minority class more heavily, effectively oversampling the minority class in the loss function — useful for imbalanced datasets.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q49. What does stratify=y do in train_test_split?

**Original Answer (from PDF):**
It ensures the train and test splits preserve the same class distribution as the original dataset. Without stratify, random splits may produce very different class ratios (especially with small or imbalanced datasets), leading to biased evaluation.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q50. What is the difference between Pipeline and ColumnTransformer?

**Original Answer (from PDF):**
Pipeline: chains steps sequentially — each step's output becomes the next step's input. Used for: imputer → scaler → model (linear sequence). ColumnTransformer: applies different transformers to different columns IN PARALLEL and concatenates the results. Used for: apply num_pipeline to numeric cols and cat_pipeline to categorical cols simultaneously. In practice: ColumnTransformer is nested inside a Pipeline as the preprocessing step.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

### Q51. What is encoding for datetime/temporal features in time series?

**Original Answer (from PDF):**
Datetime objects cannot be fed directly to ML models. Extract: year, month, day, hour, minute, day_of_week (0=Monday), day_of_year, week_of_year, is_weekend, is_holiday, quarter. For cyclical features (hour, month, day_of_week) use sin/cos encoding to preserve cyclical nature: sin(2π × hour/24) and cos(2π × hour/24) — so that hour 23 is close to hour 0. Also add lag features and rolling statistics.

**Project-Specific Viva Answer:**
- For preprocessing in our project: (1) bad entries (years <= 1920, hours == 0) were converted to NaN, (2) numeric NaNs were imputed with the median (robust to outliers), (3) categorical NaNs were filled with 'Missing', and (4) features were Ordinal Encoded. Scaling (StandardScaler) was not used in our final tree ensemble as tree models are scale-invariant.

---

## 📁 Feature Engineering & Leakage Prevention
This section covers 22 questions related to feature engineering & leakage prevention.

### Q52. What is Target Encoding?

**Original Answer (from PDF):**
Target Encoding replaces each category with the mean of the target variable for that category. It is compact and can be powerful for high-cardinality features but risks data leakage if not done carefully (use cross-validated target encoding on training data only).

**Project-Specific Viva Answer:**
- Target encoding replaces category values with the average target price. Naive target encoding includes the current row's price in the average, causing validation leakage. We implemented Out-of-Fold (OOF) target encoding inside each fold, computing means only on other inner folds and applying Bayesian smoothing (`alpha=15` to `22`) to pull rare configuration averages toward the global mean.

---

### Q53. What is data leakage?

**Original Answer (from PDF):**
Data leakage occurs when information from outside the training set (test/future data) influences the model during training, causing overly optimistic evaluation. Common causes: fitting scalers/encoders on the full dataset before splitting, or including target-derived features computed on the test set.

**Project-Specific Viva Answer:**
- We prevented data leakage by: (1) calculating all category frequencies, group means, and target encodings fold-by-fold strictly inside the cross-validation loop on training folds only, (2) using Out-of-Fold (OOF) target encoding, and (3) using temporal shifting (cumcount/shift) for asset history features so they do not see future transactions.

---

### Q54. What are methods to handle class imbalance?

**Original Answer (from PDF):**
1) Oversampling minority class (SMOTE, random oversampling). 2) Undersampling majority class (random undersampling, Tomek links, cluster centroids). 3) Class weights: set class_weight='balanced' in sklearn models. 4) Use appropriate metrics (F1, AUC-ROC) instead of accuracy. 5) Ensemble methods like BalancedRandomForest.

**Project-Specific Viva Answer:**
- Class imbalance is a classification issue. Typical approaches include class weights, undersampling, oversampling/SMOTE, threshold tuning and precision-recall metrics. My project is regression, so I did not apply any of them; I handled the skewed continuous target with a log transform and evaluated with RMSLE.

---

### Q55. What is feature engineering?

**Original Answer (from PDF):**
Feature engineering is creating new informative features from existing ones to improve model performance. Examples: extracting hour/day from a datetime, combining two features (ratio, product), binning continuous values, extracting text statistics (word count, uppercase ratio), TF-IDF vectorisation.

**Project-Specific Viva Answer:**
- We engineered: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) `MachineAge_sq` and `LogAge` to model non-linear depreciation curves, (3) `UsagePerYear` (usage intensity), (4) cyclical sine/cos date encodings for seasons, (5) binary `Has_Asset_History` flag, and (6) resale price history features (`Asset_Prev_Mean_Price`, etc.).

---

### Q56. What is dimensionality reduction? Does it discard data points?

**Original Answer (from PDF):**
Dimensionality reduction reduces the number of features while retaining as much information as possible. It does NOT discard data points (rows); it reduces columns. Methods: PCA (unsupervised, linear), LDA (supervised), t-SNE/UMAP (non-linear, for visualisation), and feature selection (RFE, SelectKBest).

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q57. What is PCA and how does it work?

**Original Answer (from PDF):**
Principal Component Analysis is an unsupervised linear dimensionality reduction technique. It finds orthogonal directions (principal components) of maximum variance in the data using eigenvectors of the covariance matrix. Each component is a linear combination of original features. PCA is linear; use Kernel PCA for non-linear data.

**Project-Specific Viva Answer:**
- PCA is an unsupervised linear dimensionality-reduction method: it projects features onto orthogonal directions of maximum variance. I did not use PCA in the final solution because tree boosting can work directly with the engineered tabular features, and PCA would reduce interpretability and can discard useful category-specific split structure.

---

### Q58. What are eigenvectors and how do you calculate PCA manually?

**Original Answer (from PDF):**
Eigenvectors of the covariance matrix define the principal components (directions of maximum variance). Steps: 1) Standardise data. 2) Compute covariance matrix. 3) Compute eigenvalues and eigenvectors. 4) Sort eigenvectors by descending eigenvalue. 5) Project data onto top-k eigenvectors. PCA is based on SVD (Singular Value Decomposition).

**Project-Specific Viva Answer:**
- PCA was not used in our final ensemble. Tree models perform binary splits on individual features and are robust to multi-collinearity; applying PCA would discard spatial feature alignment and make tree splits less interpretable. We only utilize PCA in our MNIST baseline coding challenge.

---

### Q59. What is RFE (Recursive Feature Elimination)?

**Original Answer (from PDF):**
RFE recursively trains a model, ranks features by importance, and removes the least important feature(s) in each iteration until the desired number remains. It requires a model that provides feature importances or coefficients (e.g., LinearSVC, RandomForestClassifier). sklearn: from sklearn.feature_selection import RFE.

**Project-Specific Viva Answer:**
- RFE repeatedly fits an estimator, removes the least important feature(s), and stops at the desired number. I did not run RFE in the final model. With high-cardinality categoricals and boosted trees, I used domain features plus OOF validation/feature importance to decide whether an engineered feature helped.

---

### Q60. What is TF-IDF and how does it work?

**Original Answer (from PDF):**
TF-IDF (Term Frequency–Inverse Document Frequency) converts text to numeric vectors. TF(t,d) = (count of term t in document d) / (total terms in d). IDF(t) = log(N / df(t)), where N = total documents, df(t) = documents containing t. TF-IDF = TF × IDF. Common words (high df) get low IDF; rare informative words get high IDF.

**Project-Specific Viva Answer:**
- Our project is tabular and doesn't contain text features, so NLP techniques like TF-IDF or CountVectorizer were not used. Instead, we used target encoding and frequency encoding for categorical columns.

---

### Q61. What are n-grams (unigrams, bigrams, trigrams)?

**Original Answer (from PDF):**
An n-gram is a contiguous sequence of n words. Unigram (n=1): single words. Bigram (n=2): two consecutive words (captures context). Trigram (n=3): three consecutive words. Larger n captures more context but increases sparsity and computation.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q62. What is tokenisation?

**Original Answer (from PDF):**
Tokenisation splits text into tokens (words, subwords, or characters). It is the first step in NLP: convert a string into a list of tokens that can then be numerically represented.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q63. What is the validation size in train_test_split?

**Original Answer (from PDF):**
test_size parameter: e.g., test_size=0.2 means 20% goes to test (validation), 80% to training. Common splits: 80/20 or 70/30. A larger test set gives more reliable evaluation but less data for training. With stratify, both splits have the same class proportions.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q64. What are lag features and rolling features in time series?

**Original Answer (from PDF):**
Lag features: the value of a variable at a previous time step (e.g., sales_lag1 = previous day's sales). They capture autocorrelation (today's value depends on yesterday's). Rolling features: statistics (mean, std, sum) computed over a sliding window (e.g., 7-day rolling average). They smooth noise and capture trends. Both help models detect temporal patterns without needing explicit time-series models.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q65. What is feature selection vs. feature extraction?

**Original Answer (from PDF):**
Feature selection chooses a subset of original features (RFE, SelectKBest, mutual information, correlation-based) — features remain interpretable. Feature extraction creates new features from original ones (PCA, LDA, TF-IDF) — original features are transformed into a new space, often losing interpretability.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q66. What is SelectKBest and how does it work?

**Original Answer (from PDF):**
SelectKBest selects the top k features based on a scoring function. For classification: chi2 (chi-square, for non-negative features), f_classif (ANOVA F-test), or mutual_info_classif. For regression: f_regression or mutual_info_regression. It ranks features by their statistical relationship with the target and keeps the best k.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q67. What is mutual information (MI)?

**Original Answer (from PDF):**
Mutual information measures how much knowing one variable reduces uncertainty about another. MI=0: independent variables. Higher MI: stronger dependency. Unlike correlation, MI captures non-linear relationships. Used in SelectKBest with mutual_info_classif or mutual_info_regression.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q68. What is TruncatedSVD and how does it differ from PCA?

**Original Answer (from PDF):**
TruncatedSVD is a linear dimensionality reduction using Singular Value Decomposition, but unlike PCA it does NOT center the data first. This makes it efficient for sparse matrices (like TF-IDF output) where centering would destroy sparsity. PCA = SVD on centered data. TruncatedSVD is also called Latent Semantic Analysis (LSA) in NLP.

**Project-Specific Viva Answer:**
- PCA was not used in our final ensemble. Tree models perform binary splits on individual features and are robust to multi-collinearity; applying PCA would discard spatial feature alignment and make tree splits less interpretable. We only utilize PCA in our MNIST baseline coding challenge.

---

### Q69. Did PCA improve your model score? Why or why not?

**Original Answer (from PDF):**
PCA may not improve score when: the dataset is small (PCA reduces information), the features are already low-dimensional, or tree-based models are used (they are not affected by scale/correlation). PCA helps when: there are many correlated features, the model benefits from reduced dimensionality (linear models, KNN, SVM), or training speed is a concern. Always compare performance with and without PCA using cross-validation.

**Project-Specific Viva Answer:**
- PCA was not used in our final ensemble. Tree models perform binary splits on individual features and are robust to multi-collinearity; applying PCA would discard spatial feature alignment and make tree splits less interpretable. We only utilize PCA in our MNIST baseline coding challenge.

---

### Q70. What is SelectKBest vs. RFE?

**Original Answer (from PDF):**
SelectKBest: scores each feature independently using a statistical test (chi2, f_classif, mutual information) — fast, ignores feature interactions. RFE: iteratively trains a model and removes the least important feature — captures feature interactions via model importance but is slower. Use SelectKBest for quick filtering; RFE for model-aware selection.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q71. Is PCA linear or non-linear? What about Kernel PCA?

**Original Answer (from PDF):**
Standard PCA is LINEAR — it finds linear combinations of features (principal components) that maximise variance. It cannot capture non-linear structure. Kernel PCA uses the kernel trick to implicitly map data into a high-dimensional feature space and applies PCA there, effectively performing non-linear dimensionality reduction. Common kernels: RBF, polynomial, sigmoid. Use Kernel PCA when data has non-linear structure that linear PCA cannot capture.

**Project-Specific Viva Answer:**
- PCA was not used in our final ensemble. Tree models perform binary splits on individual features and are robust to multi-collinearity; applying PCA would discard spatial feature alignment and make tree splits less interpretable. We only utilize PCA in our MNIST baseline coding challenge.

---

### Q72. What is a rolling mean and why use it?

**Original Answer (from PDF):**
A rolling mean (moving average) computes the average of a fixed-size window sliding over time. E.g., 7-day rolling mean smooths daily noise to reveal weekly trends. Used as a feature in time series ML to capture recent trend without using future data (no leakage). Also used to detect non-stationarity and as a baseline for anomaly detection.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

### Q73. What are all types of feature reduction techniques?

**Original Answer (from PDF):**
Feature SELECTION (keeps original features): Filter methods (correlation, chi-square, mutual information, variance threshold). Wrapper methods (RFE, forward/backward selection). Embedded methods (L1/Lasso regularisation, tree feature importance). Feature EXTRACTION (creates new features): PCA, Kernel PCA, LDA, TruncatedSVD/LSA, t-SNE, UMAP (for visualisation), Autoencoders (deep learning). Dimensionality reduction is the umbrella term covering both.

**Project-Specific Viva Answer:**
- Our feature engineering included: (1) `MachineAge` (`SaleYear - ManufactureYear`), (2) non-linear age features (`MachineAge_sq` and `LogAge`), (3) `UsagePerYear`, (4) cyclical date coordinate transformations, (5) binary `Has_Asset_History` flag, (6) resale price history statistics (`Asset_Prev_Mean_Price`, etc.), and (7) fold-level group mean features computed strictly inside the cross-validation loop to prevent validation leakage.

---

## 📁 Machine Learning Models & Algorithms
This section covers 69 questions related to machine learning models & algorithms.

### Q74. How does Logistic Regression work?

**Original Answer (from PDF):**
Logistic Regression applies the sigmoid function (σ(z) = 1/(1+e^−z)) to a linear combination of features to output a probability between 0 and 1. A threshold (usually 0.5) converts it to a class. It is a discriminative linear model. Despite the name 'regression,' it is used for classification.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q75. Why is Logistic Regression called 'Regression' if it is used for classification?

**Original Answer (from PDF):**
Because it models the log-odds (logit) of the probability as a linear regression: log(p/(1−p)) = w■x + b. The regression part refers to estimating a continuous probability, which is then thresholded for classification.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q76. What is the loss function of Logistic Regression?

**Original Answer (from PDF):**
Binary Cross-Entropy (Log Loss): L = −[y·log(p) + (1−y)·log(1−p)]. It penalises confident wrong predictions heavily. MSE is not used because it creates a non-convex loss surface for logistic regression, making optimisation harder.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q77. What is the 'C' parameter in Logistic Regression / SVM?

**Original Answer (from PDF):**
C is the inverse of regularisation strength. A large C means less regularisation (model tries hard to classify all training points correctly, risks overfitting). A small C means more regularisation (allows more misclassifications, simpler model, may underfit). So C is the trade-off between training accuracy and generalisation.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q78. What is the 'solver' parameter in Logistic Regression?

**Original Answer (from PDF):**
The solver specifies the optimisation algorithm: 'lbfgs' (default, good for multiclass, memory-efficient quasi-Newton), 'saga' (supports all penalties including L1, good for large datasets), 'liblinear' (good for small datasets, only binary+OVR), 'newton-cg' / 'sag' (for L2 only).

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q79. Is Logistic Regression a parametric or non-parametric model?

**Original Answer (from PDF):**
Parametric — it assumes a fixed functional form (linear relationship between features and log-odds) and learns a fixed number of parameters (weights). The model complexity does not grow with training data size.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q80. How does a Decision Tree work and how is the root node selected?

**Original Answer (from PDF):**
A Decision Tree recursively splits data on the feature and threshold that maximises information gain (or minimises impurity). For classification, impurity is measured by Gini index or Entropy. The feature with the highest information gain (= parent impurity − weighted child impurity) becomes the root. A pure node (all one class) has Gini=0 / Entropy=0 and is not split further.

**Project-Specific Viva Answer:**
- Decision trees are the fundamental building blocks of our ensemble (LightGBM and CatBoost). A tree splits nodes by finding feature thresholds (like `MachineAge > 10.5` or `RegionCode = 12`) that maximize Gini Impurity reduction or Information Gain (variance reduction for regression). Trees handle our technical categorical variables and missing values natively.

---

### Q81. What is entropy and information gain?

**Original Answer (from PDF):**
Entropy H = −Σ p_i·log2(p_i) measures disorder/uncertainty. A pure node (all one class) has H=0; maximum disorder has H=1 for binary. Information Gain = Parent entropy − weighted average of child entropies after a split. Higher IG = better split.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q82. What is the Gini index and its range?

**Original Answer (from PDF):**
Gini = 1 − Σ p_i² for each class i. Range: 0 (pure node) to 0.5 (binary, maximum impurity). The Decision Tree picks the split that minimises the weighted Gini of child nodes. Gini is faster to compute than entropy.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q83. What is the difference between a pure node and an impure node?

**Original Answer (from PDF):**
A pure node contains samples of only one class (Gini=0, Entropy=0) — no further splitting is needed. An impure node contains a mix of classes; splitting it can reduce impurity and improve predictions.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q84. What is Random Forest and how does it work?

**Original Answer (from PDF):**
Random Forest is a bagging ensemble of Decision Trees. Each tree is trained on a bootstrap sample (random sample with replacement) of the training data. At each split, only a random subset of features (√n_features for classification) is considered. The final prediction is the majority vote (classification) or average (regression). This reduces variance vs. a single deep tree.

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q85. Is Random Forest parametric or non-parametric?

**Original Answer (from PDF):**
Non-parametric — it makes no assumptions about the functional form of the data and complexity grows with data. The individual Decision Trees store split rules rather than learned weight vectors.

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q86. What are the parameters of Random Forest and what do they do?

**Original Answer (from PDF):**
n_estimators: number of trees (more = more stable but slower). max_depth: maximum depth of each tree (limits complexity, prevents overfitting). min_samples_split: min samples to split a node. min_samples_leaf: min samples in a leaf. max_features: features considered per split. bootstrap: whether to use bootstrap samples.

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q87. What is bagging and boosting?

**Original Answer (from PDF):**
Bagging (Bootstrap AGGregating): trains multiple models in PARALLEL on random subsets of data, then aggregates predictions (reduces variance). Example: Random Forest. Boosting: trains models SEQUENTIALLY, each correcting the errors of the previous (reduces bias). Examples: AdaBoost, Gradient Boosting, XGBoost, LightGBM.

**Project-Specific Viva Answer:**
- Bagging (e.g. Random Forest) trains independent trees in parallel on bootstrap samples and averages their predictions to reduce **variance**. Boosting (e.g. LightGBM, CatBoost) trains trees sequentially, where each new tree fits the errors (residuals) of the previous trees to reduce **bias**. In our project, boosting performed significantly better, lowering RMSLE from 0.2895 (Random Forest) to 0.1932 (LightGBM).

---

### Q88. Which reduces variance and which reduces bias — bagging or boosting?

**Original Answer (from PDF):**
Bagging reduces variance (models are independent, averaging cancels noise). Boosting reduces bias (sequential focus on difficult examples). Bagging is better when the model overfits; boosting is better when the model underfits.

**Project-Specific Viva Answer:**
- Bagging (e.g. Random Forest) trains independent trees in parallel on bootstrap samples and averages their predictions to reduce **variance**. Boosting (e.g. LightGBM, CatBoost) trains trees sequentially, where each new tree fits the errors (residuals) of the previous trees to reduce **bias**. In our project, boosting performed significantly better, lowering RMSLE from 0.2895 (Random Forest) to 0.1932 (LightGBM).

---

### Q89. How does XGBoost work?

**Original Answer (from PDF):**
XGBoost (Extreme Gradient Boosting) is a regularised, optimised gradient boosting framework. It builds trees sequentially, where each tree fits the negative gradient (residuals) of the loss function. It adds L1/L2 regularisation, handles missing values internally, uses histogram-based splits, and supports parallel tree construction. Key hyperparameters: n_estimators, learning_rate, max_depth, subsample, colsample_bytree.

**Project-Specific Viva Answer:**
- We initially used XGBoost as our primary model (baseline RMSLE 0.2045). However, during ensembling in v14, XGBoost predictions were too highly correlated with LightGBM, receiving 0% weight in our blend optimization. In v15, we replaced XGBoost with a second LightGBM model (`seed=123`) with different hyperparameters, which introduced seed diversity and helped lower our blended OOF RMSLE to 0.19326.

---

### Q90. What is the difference between XGBoost and LightGBM?

**Original Answer (from PDF):**
LightGBM uses a leaf-wise (best-first) tree growth strategy while XGBoost uses level-wise. Leaf-wise grows the leaf with the highest loss reduction, leading to deeper trees but faster convergence. LightGBM also uses histogram-based feature binning, making it faster and more memory-efficient on large datasets. XGBoost can be slower but sometimes more accurate on smaller datasets.

**Project-Specific Viva Answer:**
- We initially used XGBoost as our primary model (baseline RMSLE 0.2045). However, during ensembling in v14, XGBoost predictions were too highly correlated with LightGBM, receiving 0% weight in our blend optimization. In v15, we replaced XGBoost with a second LightGBM model (`seed=123`) with different hyperparameters, which introduced seed diversity and helped lower our blended OOF RMSLE to 0.19326.

---

### Q91. What does 'Light' in LightGBM mean?

**Original Answer (from PDF):**
Light refers to the lightweight, memory-efficient design: histogram-based feature binning reduces the data into discrete bins, cutting memory and compute. It supports GPU training and can handle very large datasets efficiently.

**Project-Specific Viva Answer:**
- LightGBM is a leaf-wise gradient boosting framework. It grows trees by splitting the leaf that reduces loss most, creating deeper, more precise splits than level-wise models like XGBoost. It also uses histogram-based binning, making it 6-10x faster. We used two LightGBMs: LGB1 (seed=42) and LGB2 (seed=123) with different hyperparameters to create a diverse ensemble.

---

### Q92. What is Gradient Boosting and how does it relate to XGBoost?

**Original Answer (from PDF):**
Gradient Boosting iteratively adds weak learners (usually small trees) that fit the negative gradient of the loss function (residuals). XGBoost is an implementation of gradient boosting with additional regularisation (L1/L2), efficient histogram-based splits, and parallelisation.

**Project-Specific Viva Answer:**
- We initially used XGBoost as our primary model (baseline RMSLE 0.2045). However, during ensembling in v14, XGBoost predictions were too highly correlated with LightGBM, receiving 0% weight in our blend optimization. In v15, we replaced XGBoost with a second LightGBM model (`seed=123`) with different hyperparameters, which introduced seed diversity and helped lower our blended OOF RMSLE to 0.19326.

---

### Q93. What is Gradient Descent?

**Original Answer (from PDF):**
Gradient Descent is an iterative optimisation algorithm that minimises a loss function by moving in the direction of the negative gradient. Update rule: w ← w − η·∇L(w), where η is the learning rate. Variants: Batch GD (all data), Stochastic GD (one sample), Mini-batch GD (subset).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q94. How does KNN (K-Nearest Neighbours) work?

**Original Answer (from PDF):**
KNN is a lazy, non-parametric algorithm. For a new point, it finds the k nearest training points (by Euclidean/other distance), then predicts the majority class (classification) or average (regression) among those k neighbours. Small k → low bias, high variance (overfitting); large k → high bias, low variance (underfitting). KNN is sensitive to feature scaling.

**Project-Specific Viva Answer:**
- KNN was evaluated as a baseline regressor but was not selected for our final ensemble because: (1) it has an $O(N)$ prediction time complexity which is extremely slow on our 138,701 training rows, (2) it relies on distance metrics which are highly sensitive to the scale of features and the hundreds of technical categorical columns in our data.

---

### Q95. Why is KNN called a lazy learner?

**Original Answer (from PDF):**
Because it does not build an explicit model during training — it simply stores all training data. All computation happens at prediction time (searching for nearest neighbours). This makes training O(1) but prediction O(n) for n training points.

**Project-Specific Viva Answer:**
- KNN was evaluated as a baseline regressor but was not selected for our final ensemble because: (1) it has an $O(N)$ prediction time complexity which is extremely slow on our 138,701 training rows, (2) it relies on distance metrics which are highly sensitive to the scale of features and the hundreds of technical categorical columns in our data.

---

### Q96. How does SVM (Support Vector Machine) work?

**Original Answer (from PDF):**
SVM finds the hyperplane that maximises the margin between classes (distance between the hyperplane and the nearest training points of each class, called support vectors). Only support vectors determine the hyperplane — other points don't matter. For non-linearly separable data, the kernel trick maps inputs to higher dimensions. C controls regularisation (slack variables).

**Project-Specific Viva Answer:**
- SVM (Support Vector Regression) was not used in the final ensemble because: (1) it has an $O(N^2)$ to $O(N^3)$ training complexity, making it computationally expensive on our 138,701 records, (2) it requires extensive scaling and is highly sensitive to outliers, which tree-based models like LightGBM handle naturally.

---

### Q97. What are kernel types in SVM?

**Original Answer (from PDF):**
Linear: for linearly separable data. RBF (Radial Basis Function): maps to infinite dimensions, good for non-linear data. Polynomial: computes polynomial combinations of features. Sigmoid: similar to neural network activation. RBF is the most commonly used default.

**Project-Specific Viva Answer:**
- SVM (Support Vector Regression) was not used in the final ensemble because: (1) it has an $O(N^2)$ to $O(N^3)$ training complexity, making it computationally expensive on our 138,701 records, (2) it requires extensive scaling and is highly sensitive to outliers, which tree-based models like LightGBM handle naturally.

---

### Q98. What is One-vs-One and One-vs-All (OvR) in multi-class SVM?

**Original Answer (from PDF):**
One-vs-All (OvR): train one binary classifier per class (class k vs. all others). Predict the class whose classifier gives the highest confidence. One-vs-One (OvO): train a classifier for every pair of classes (n(n-1)/2 classifiers), predict by majority vote. OvO is typically more accurate; OvR is faster to train.

**Project-Specific Viva Answer:**
- SVM (Support Vector Regression) was not used in the final ensemble because: (1) it has an $O(N^2)$ to $O(N^3)$ training complexity, making it computationally expensive on our 138,701 records, (2) it requires extensive scaling and is highly sensitive to outliers, which tree-based models like LightGBM handle naturally.

---

### Q99. What is L1 and L2 regularisation?

**Original Answer (from PDF):**
L1 (Lasso): adds Σ|w■| penalty. Produces sparse models (some weights go exactly to 0), useful for feature selection. L2 (Ridge): adds Σw■² penalty. Shrinks all weights towards 0 but rarely eliminates them, preventing overfitting while keeping all features. ElasticNet combines both.

**Project-Specific Viva Answer:**
- To regularize our tree models and prevent overfitting, we used: (1) L1 and L2 regularization parameters (`reg_alpha` and `reg_lambda` in LightGBM, `l2_leaf_reg` in CatBoost), (2) feature subsampling (`colsample_bytree=0.65` to `0.70`), and (3) row subsampling (`subsample=0.75` to `0.80`).

---

### Q100. What is overfitting and underfitting?

**Original Answer (from PDF):**
Overfitting: model learns training data too well including noise, so it performs well on train but poorly on unseen test data (high variance, low bias). Underfitting: model is too simple to capture patterns, performing poorly on both train and test (high bias, low variance).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q101. How do you detect if a model is overfitting or underfitting?

**Original Answer (from PDF):**
Compare training and validation/test scores. Overfitting: train score >> validation score (large gap). Underfitting: both train and validation scores are low. Ideal: train and validation scores are both high and close to each other.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q102. How do you reduce overfitting?

**Original Answer (from PDF):**
1) Get more training data. 2) Reduce model complexity (fewer layers/trees/depth). 3) Regularisation (L1/L2/dropout). 4) Cross-validation. 5) Early stopping. 6) Pruning (for decision trees). 7) Bagging/ensemble methods. 8) Feature selection (remove irrelevant features). 9) Data augmentation.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q103. How do you reduce underfitting?

**Original Answer (from PDF):**
1) Increase model complexity (more features, deeper trees, more neurons). 2) Add more relevant features. 3) Reduce regularisation. 4) Train longer. 5) Use a more powerful algorithm. 6) Feature engineering.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q104. If a model is overfitting and you use XGBoost, should you increase or decrease?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- We initially used XGBoost as our primary model (baseline RMSLE 0.2045). However, during ensembling in v14, XGBoost predictions were too highly correlated with LightGBM, receiving 0% weight in our blend optimization. In v15, we replaced XGBoost with a second LightGBM model (`seed=123`) with different hyperparameters, which introduced seed diversity and helped lower our blended OOF RMSLE to 0.19326.

---

### Q105. What are the key parameters of TfidfVectorizer?

**Original Answer (from PDF):**
max_features: vocabulary size limit. ngram_range: e.g., (1,2) includes unigrams and bigrams. min_df: ignore terms in fewer than min_df documents (reduces noise). max_df: ignore terms in more than max_df fraction (removes stop words automatically). stop_words: built-in or custom stop word list. sublinear_tf: apply log to TF.

**Project-Specific Viva Answer:**
- Our project is tabular and doesn't contain text features, so NLP techniques like TF-IDF or CountVectorizer were not used. Instead, we used target encoding and frequency encoding for categorical columns.

---

### Q106. What is the height vs. depth of a decision tree?

**Original Answer (from PDF):**
Depth: the length of the longest path from the root to a leaf (how many levels). Height of a node: the length of the longest path from that node to a leaf. For a binary tree with depth d, the maximum number of nodes = 2^(d+1) − 1.

**Project-Specific Viva Answer:**
- Decision trees are the fundamental building blocks of our ensemble (LightGBM and CatBoost). A tree splits nodes by finding feature thresholds (like `MachineAge > 10.5` or `RegionCode = 12`) that maximize Gini Impurity reduction or Information Gain (variance reduction for regression). Trees handle our technical categorical variables and missing values natively.

---

### Q107. What is num_leaves in LightGBM and how does it differ from max_depth?

**Original Answer (from PDF):**
LightGBM grows trees leaf-wise (best-first), not level-wise. num_leaves controls the maximum number of leaves in one tree — directly controls model complexity. A higher num_leaves = more complex model. max_depth can constrain depth too, but num_leaves is the primary LightGBM complexity control. Typical values: 20–150.

**Project-Specific Viva Answer:**
- LightGBM is a leaf-wise gradient boosting framework. It grows trees by splitting the leaf that reduces loss most, creating deeper, more precise splits than level-wise models like XGBoost. It also uses histogram-based binning, making it 6-10x faster. We used two LightGBMs: LGB1 (seed=42) and LGB2 (seed=123) with different hyperparameters to create a diverse ensemble.

---

### Q108. What is a weak learner?

**Original Answer (from PDF):**
A weak learner is a model that performs only slightly better than random guessing (accuracy slightly above 50% for binary classification). Examples: shallow decision tree (depth=1, called a 'decision stump'), linear model on complex data. Boosting combines many weak learners to form a strong learner.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q109. What is a baseline model?

**Original Answer (from PDF):**
A baseline model is the simplest possible model used as a reference point. For classification: predict the majority class always (DummyClassifier). For regression: predict the mean always (DummyRegressor). All subsequent models must outperform the baseline to justify their complexity.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q110. When is ensemble learning NOT recommended?

**Original Answer (from PDF):**
1) When you need high interpretability (single decision tree or logistic regression is better). 2) When training data is very small (ensembles may overfit). 3) When inference speed is critical (single lightweight model preferred). 4) When the problem is simple and a single model performs well enough. 5) When compute resources are very limited.

**Project-Specific Viva Answer:**
- Our ensemble blends LGB1 (30%), LGB2 (45%), and CatBoost (25%). When we combine predictions from diverse models (with different seeds, subsamples, and split structures), their individual errors cancel out, reducing model variance and improving generalisation.

---

### Q111. What is ExtraTreesClassifier (Extremely Randomised Trees)?

**Original Answer (from PDF):**
Like Random Forest but with additional randomness: both the feature AND the split threshold are chosen randomly (not the best threshold). This makes it faster to train and often reduces variance further, though it may increase bias slightly. It is also a bagging ensemble.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q112. What is AdaBoost?

**Original Answer (from PDF):**
Adaptive Boosting trains a sequence of weak learners, giving higher weights to misclassified samples so subsequent learners focus on them. Final prediction is a weighted vote of all learners. It was one of the first successful boosting algorithms. Sensitive to noisy data and outliers (unlike Gradient Boosting which is more robust).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q113. What is Stochastic Gradient Descent (SGD) vs. Batch Gradient Descent?

**Original Answer (from PDF):**
Batch GD: uses the entire training set to compute gradients — slow per iteration, guaranteed convergence but expensive. SGD: uses one sample at a time — fast, noisy updates, can escape local minima but may not converge to exact minimum. Mini-batch GD: compromise (small random batches) — fast and relatively stable, used in deep learning. sklearn's SGDClassifier uses SGD.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q114. What is min_samples_split vs. min_samples_leaf?

**Original Answer (from PDF):**
min_samples_split: minimum number of samples required to split an internal node. If a node has fewer samples, it becomes a leaf. min_samples_leaf: minimum number of samples required in a leaf node after splitting. Both prevent overfitting. E.g., min_samples_split=8 means a node with 7 samples will NOT split (becomes a leaf).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q115. What is the 'degree' parameter in SVC (Polynomial kernel)?

**Original Answer (from PDF):**
For kernel='poly', the polynomial kernel is K(x, y) = (γ·x■y + r)^d, where d = degree. degree=2: quadratic decision boundary (can model ellipses, parabolas). degree=3: cubic boundary. Higher degree = more complex, more risk of overfitting. Default is degree=3. For linearly separable data, use kernel='linear'. RBF kernel is generally preferred over polynomial as it has fewer hyperparameters.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q116. What is Naive Bayes and why is it 'Naive'?

**Original Answer (from PDF):**
Naive Bayes applies Bayes' theorem (P(class|features) ∝ P(features|class)·P(class)) with the 'naive' assumption that all features are conditionally independent given the class. This is rarely true in practice but simplifies computation dramatically. Despite the unrealistic assumption, it works well for text classification (spam detection, sentiment analysis).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q117. What is the difference between likelihood and probability?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q118. Probability: given fixed parameters, what is the chance of observing this data? P(data|parameters).

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q119. Likelihood: given observed data, how likely are different parameter values? L(parameters|data).

**Original Answer (from PDF):**
Likelihood is used in Maximum Likelihood Estimation (MLE) to fit model parameters.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q120. What is the difference between KMeans (clustering) and KNN (classification/regression)?

**Original Answer (from PDF):**
KMeans is unsupervised — groups unlabelled data into k clusters based on feature similarity (minimises inertia). KNN is supervised — predicts labels for new points based on the k nearest labelled training points. KMeans learns cluster centroids; KNN stores all training data (lazy learner). Different k: KMeans k = number of clusters; KNN k = number of neighbours.

**Project-Specific Viva Answer:**
- KNN was evaluated as a baseline regressor but was not selected for our final ensemble because: (1) it has an $O(N)$ prediction time complexity which is extremely slow on our 138,701 training rows, (2) it relies on distance metrics which are highly sensitive to the scale of features and the hundreds of technical categorical columns in our data.

---

### Q121. What is Ridge (L2) and Lasso (L1) Regression?

**Original Answer (from PDF):**
Ridge Regression adds L2 penalty (λΣw■²) to linear regression's loss function. Shrinks all coefficients towards zero but rarely eliminates them — good for multicollinearity. Lasso Regression adds L1 penalty (λΣ|w■|). Can shrink coefficients exactly to zero (sparse model), acting as automatic feature selection. Use Lasso when you believe many features are irrelevant.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q122. Can Logistic Regression be used for multi-class problems? How?

**Original Answer (from PDF):**
Yes. Two strategies: One-vs-Rest (OvR/OvA): train one binary classifier per class (k classifiers for k classes). One-vs-One (OvO): train a classifier for each pair of classes. Multinomial (softmax): extends logistic regression directly to k classes using the softmax function. In sklearn, multi_class='multinomial' or 'ovr'.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q123. What are tree-based algorithms? Name them.

**Original Answer (from PDF):**
Decision Tree, Random Forest (bagging of DTs), Extra Trees, AdaBoost (boosting with DT stumps), Gradient Boosting (sklearn), XGBoost, LightGBM, CatBoost. All are non-parametric, handle non-linearity naturally, and are relatively robust to outliers and feature scaling (no scaling needed).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q124. What is a parametric vs. non-parametric model?

**Original Answer (from PDF):**
Parametric: assumes a fixed functional form and learns a fixed number of parameters regardless of data size. Examples: Logistic Regression, Linear Regression, Naive Bayes, Neural Networks (fixed architecture). Non-parametric: makes fewer assumptions; model complexity can grow with data size. Examples: KNN, Decision Trees, Random Forest, SVM (with kernel).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q125. What is the difference between Linear Regression and Logistic Regression?

**Original Answer (from PDF):**
Linear Regression: predicts continuous outputs; uses MSE loss; output is unbounded (−∞ to +∞). Logistic Regression: predicts class probabilities; uses log-loss (cross-entropy); applies sigmoid to bound output to [0,1]; used for classification. Both use a linear combination of features, but logistic regression applies a non-linear sigmoid transformation.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q126. What is an ARIMA model and what are its parameters?

**Original Answer (from PDF):**
ARIMA (AutoRegressive Integrated Moving Average) is a time series model. Parameters: p (AutoRegressive order — number of lag observations), d (differencing order — number of times series is differenced to make it stationary), q (Moving Average order — number of lagged forecast errors). SARIMA extends ARIMA with seasonal components (P, D, Q, s).

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q127. What is the 'support' vector in SVM and why does it matter?

**Original Answer (from PDF):**
Support vectors are the training data points closest to the decision boundary (within or on the margin). They are the only points that define the hyperplane — all other points do not matter. This makes SVM memory-efficient and robust: adding points far from the boundary does not change the model. However, moving support vectors drastically changes the decision boundary.

**Project-Specific Viva Answer:**
- SVM (Support Vector Regression) was not used in the final ensemble because: (1) it has an $O(N^2)$ to $O(N^3)$ training complexity, making it computationally expensive on our 138,701 records, (2) it requires extensive scaling and is highly sensitive to outliers, which tree-based models like LightGBM handle naturally.

---

### Q128. Does bagging give ALL features to each tree in Random Forest?

**Original Answer (from PDF):**
No — this is a key distinction. In a basic Bagging ensemble (BaggingClassifier), each learner may use all features. But Random Forest adds an extra layer of randomness: at each split in each tree, only a RANDOM SUBSET of features is considered (typically sqrt(n_features) for classification, n_features/3 for regression). This decorrelates the trees further, reducing variance more than plain bagging. The max_features parameter controls this.

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q129. Why is Random Forest better than a single Decision Tree?

**Original Answer (from PDF):**
A single Decision Tree tends to overfit (high variance) — small changes in data produce very different trees. Random Forest reduces variance by: 1) Bootstrap sampling (each tree sees different data). 2) Feature randomness at each split (trees are decorrelated). 3) Averaging/voting over many trees cancels out individual errors. Result: RF has similar bias to a single deep tree but much lower variance, giving better generalisation.

**Project-Specific Viva Answer:**
- Decision trees are the fundamental building blocks of our ensemble (LightGBM and CatBoost). A tree splits nodes by finding feature thresholds (like `MachineAge > 10.5` or `RegionCode = 12`) that maximize Gini Impurity reduction or Information Gain (variance reduction for regression). Trees handle our technical categorical variables and missing values natively.

---

### Q130. Why are Bagging/Boosting better than a single Decision Tree?

**Original Answer (from PDF):**
Single DT: high variance (overfits) if deep, high bias (underfits) if shallow. Bagging: trains multiple DTs on bootstrap samples and averages — reduces variance, keeps bias similar. Boosting: trains DTs sequentially on residuals — reduces bias, slightly increases variance but net effect is lower total error. Both benefit from the wisdom of crowds: combining many weak/medium learners outperforms a single learner.

**Project-Specific Viva Answer:**
- Decision trees are the fundamental building blocks of our ensemble (LightGBM and CatBoost). A tree splits nodes by finding feature thresholds (like `MachineAge > 10.5` or `RegionCode = 12`) that maximize Gini Impurity reduction or Information Gain (variance reduction for regression). Trees handle our technical categorical variables and missing values natively.

---

### Q131. At what value of k in KNN is there overfitting vs. underfitting?

**Original Answer (from PDF):**
Small k (e.g., k=1): model memorises training data, very sensitive to noise → overfitting (high variance, low bias). As k increases: model becomes smoother and more generalised. Large k (e.g., k=n): model predicts the majority class for everything → underfitting (high bias, low variance). Optimal k is found via cross-validation, typically odd values to avoid ties. Rule of thumb: start with k=sqrt(n_samples).

**Project-Specific Viva Answer:**
- KNN was evaluated as a baseline regressor but was not selected for our final ensemble because: (1) it has an $O(N)$ prediction time complexity which is extremely slow on our 138,701 training rows, (2) it relies on distance metrics which are highly sensitive to the scale of features and the hundreds of technical categorical columns in our data.

---

### Q132. What is the height vs. depth of a decision tree? Binary tree node count?

**Original Answer (from PDF):**
Depth of the tree = number of edges on the longest path from root to leaf = number of levels − 1. Height of a node = length of longest path from that node to a leaf. Root has the maximum height = tree depth. Leaves have height = 0. For a FULL binary tree of depth d: maximum nodes = 2^(d+1) − 1. E.g., depth=3: max nodes = 2■ − 1 = 15. max_depth in sklearn = maximum depth allowed (limits tree growth to prevent overfitting).

**Project-Specific Viva Answer:**
- Decision trees are the fundamental building blocks of our ensemble (LightGBM and CatBoost). A tree splits nodes by finding feature thresholds (like `MachineAge > 10.5` or `RegionCode = 12`) that maximize Gini Impurity reduction or Information Gain (variance reduction for regression). Trees handle our technical categorical variables and missing values natively.

---

### Q133. Is RandomForestClassifier a bagging or boosting algorithm? How?

**Original Answer (from PDF):**
Bagging (Bootstrap Aggregating). Random Forest trains N trees INDEPENDENTLY and IN PARALLEL on bootstrap samples (random samples with replacement). Predictions are combined by majority vote (classification) or mean (regression). It is NOT boosting — boosting trains sequentially and corrects errors of previous models. Random Forest = Bagging + extra feature randomness (only sqrt(features) considered at each split).

**Project-Specific Viva Answer:**
- Bagging (e.g. Random Forest) trains independent trees in parallel on bootstrap samples and averages their predictions to reduce **variance**. Boosting (e.g. LightGBM, CatBoost) trains trees sequentially, where each new tree fits the errors (residuals) of the previous trees to reduce **bias**. In our project, boosting performed significantly better, lowering RMSLE from 0.2895 (Random Forest) to 0.1932 (LightGBM).

---

### Q134. What is the loss function in Random Forest Classifier?

**Original Answer (from PDF):**
Random Forest itself does not minimise a single differentiable loss function like gradient-based models. Each tree is built using impurity measures: Gini index or Entropy (Information Gain). The forest makes predictions by aggregating tree votes. There is no explicit gradient-based optimisation — RF is a non-parametric method. This distinguishes it from gradient boosting which explicitly minimises a loss function via gradient descent in function space.

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q135. What are the parameters of RandomForestClassifier and what do they do?

**Original Answer (from PDF):**
n_estimators: number of trees (more = more stable, more compute). max_depth: maximum tree depth (None = grow until pure). min_samples_split: minimum samples to consider splitting a node. min_samples_leaf: minimum samples required in a leaf. max_features: features to consider per split ('sqrt', 'log2', int, float). bootstrap: use bootstrap sampling (True = RF; False = Pasting). class_weight: handle imbalance. oob_score: use out-of-bag samples for evaluation. random_state: for reproducibility.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q136. What are the regularisation methods for Random Forest?

**Original Answer (from PDF):**
RF doesn't have explicit L1/L2 regularisation, but complexity is controlled via: max_depth (limit tree depth). min_samples_split & min_samples_leaf (require minimum samples to split or be a leaf — prevents over-splitting small node). max_features (feature randomness reduces overfitting). n_estimators (more trees = more stable but not more regularised per se). min_impurity_decrease (only split if impurity decreases by this amount).

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q137. What is the penalty parameter in Logistic Regression? What are the options?

**Original Answer (from PDF):**
penalty controls regularisation type: 'l2' (default): Ridge regularisation — shrinks all weights, rarely eliminates them. 'l1': Lasso regularisation — can zero out weights (feature selection). 'elasticnet': combination of L1 and L2 (requires solver='saga'). 'none': no regularisation (may overfit on small datasets). The strength of regularisation is controlled by C (inverse: smaller C = more regularisation).

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q138. What are the different solvers in Logistic Regression and when to use each?

**Original Answer (from PDF):**
'lbfgs' (default): L-BFGS quasi-Newton method. Supports L2 and none. Good for small-medium datasets, multiclass. 'liblinear': coordinate descent. Supports L1 and L2. Good for small datasets, only binary OvR. 'saga': Stochastic Average Gradient Amended. Supports all penalties including elasticnet. Best for large datasets. 'sag': Stochastic Average Gradient. Supports L2 and none. Faster on large datasets. 'newton-cg': Newton's method with conjugate gradient. L2 and none only. Rule: use 'lbfgs' by default; 'saga' for L1/elasticnet or large data; 'liblinear' for small binary tasks.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q139. What is the SGDClassifier vs. Logistic Regression in sklearn?

**Original Answer (from PDF):**
SGDClassifier with loss='log_loss' is equivalent to Logistic Regression but trained with Stochastic Gradient Descent — better for very large datasets (online learning). SGDClassifier with loss='hinge' is equivalent to a linear SVM. SGDClassifier with loss='modified_huber' is a robust version of log loss. Key difference: SGDClassifier updates weights one sample (or mini-batch) at a time, making it memory-efficient. LogisticRegression uses more sophisticated optimisers (L-BFGS, SAG) that converge more reliably on smaller datasets.

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q140. What is the 'C' parameter in SVC and how does it affect the model?

**Original Answer (from PDF):**
C is the regularisation parameter (inverse of regularisation strength). Large C: hard margin — tries to classify ALL training points correctly, narrow margin, complex boundary → overfitting risk. Small C: soft margin — allows some misclassifications (slack variables), wider margin, simpler boundary → better generalisation. Tune C with cross-validation. Combine with kernel and gamma for full tuning.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q141. Why did you use only boosting/leaf-based models and not other models?

**Original Answer (from PDF):**
A justified answer: Boosting models (XGBoost, LightGBM) consistently outperform linear models on tabular data with complex non-linear relationships. They handle missing values internally, require minimal preprocessing (no scaling needed for trees), naturally capture feature interactions, and their leaf-wise growth (LightGBM) is computationally efficient. I also tried linear models (Logistic Regression) as a baseline but they performed significantly worse, confirming the non-linear nature of the data.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

### Q142. What is ARIMA and how does it work? Explain its parameters p, d, q.

**Original Answer (from PDF):**
ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting model. p (AR order): number of lag observations included in the model (autoregressive part). Uses p past values to predict the next value. d (differencing order): number of times the series is differenced to make it stationary. If d=1: subtract consecutive values. q (MA order): number of lagged forecast errors in the prediction equation (moving average part). Uses q past errors to correct predictions. SARIMA adds seasonal terms (P, D, Q, s) for seasonal patterns. Determine p from PACF plot, q from ACF plot, d from ADF stationarity test.

**Project-Specific Viva Answer:**
- Our final model is a blended ensemble of LightGBM 1 (30% weight), LightGBM 2 (45% weight, seed=123 for diversity), and CatBoost (25% weight). Tree models are used because they handle multi-collinearity, non-linear relationships, and missing values natively. We discarded XGBoost because its predictions were too correlated with LightGBM, receiving 0% weight.

---

## 📁 Hyperparameter Tuning & Cross-Validation
This section covers 17 questions related to hyperparameter tuning & cross-validation.

### Q143. What is the difference between parameters and hyperparameters?

**Original Answer (from PDF):**
Parameters are learned from the data during training (e.g., weights in logistic regression, split thresholds in decision trees). Hyperparameters are set before training and control the learning process (e.g., n_estimators, max_depth, learning_rate, C, kernel). You tune hyperparameters; the model learns parameters.

**Project-Specific Viva Answer:**
- We tuned our models using a 5-Fold Cross-Validation loop. For each fold, we trained on 80% of the data and validated on 20%. Hyperparameters tuned in our best `v18` LightGBM include: `num_leaves=1023` (more capacity), `learning_rate=0.010` (precise training), `reg_alpha=0.2`, `reg_lambda=3.0`, and row/col subsampling, utilizing early stopping (300 rounds) to prevent overfitting.

---

### Q144. What is hyperparameter tuning and why is it needed?

**Original Answer (from PDF):**
HPT finds the optimal set of hyperparameters that maximise model performance on validation data. Without tuning, you rely on default values that may not suit your data. Tuning prevents underfitting (model too simple) or overfitting (model too complex).

**Project-Specific Viva Answer:**
- We tuned our models using a 5-Fold Cross-Validation loop. For each fold, we trained on 80% of the data and validated on 20%. Hyperparameters tuned in our best `v18` LightGBM include: `num_leaves=1023` (more capacity), `learning_rate=0.010` (precise training), `reg_alpha=0.2`, `reg_lambda=3.0`, and row/col subsampling, utilizing early stopping (300 rounds) to prevent overfitting.

---

### Q145. What is GridSearchCV?

**Original Answer (from PDF):**
GridSearchCV exhaustively tries every combination of the provided hyperparameter grid with cross-validation, selecting the combination with the best CV score. It is thorough but computationally expensive (fits = n_combinations × cv_folds).

**Project-Specific Viva Answer:**
- GridSearchCV performs an exhaustive search over all parameter combinations. With our large dataset (138,701 rows, 50 features), GridSearchCV would take days to fit. We used `RandomizedSearchCV` for our baseline XGBoost tuning, sampling a fixed number of combinations, which allowed us to find near-optimal hyperparameters (like learning rate, max depth, and regularization) in a fraction of the time.

---

### Q146. What is RandomizedSearchCV and how does it differ from GridSearchCV?

**Original Answer (from PDF):**
RandomizedSearchCV samples a fixed number (n_iter) of hyperparameter combinations at random. It is faster than GridSearchCV and often finds near-optimal results when the grid is large. Trade-off: it may miss the global optimum. Number of fits = n_iter × cv_folds.

**Project-Specific Viva Answer:**
- GridSearchCV performs an exhaustive search over all parameter combinations. With our large dataset (138,701 rows, 50 features), GridSearchCV would take days to fit. We used `RandomizedSearchCV` for our baseline XGBoost tuning, sampling a fixed number of combinations, which allowed us to find near-optimal hyperparameters (like learning rate, max depth, and regularization) in a fraction of the time.

---

### Q147. What is cross-validation (CV) and why is it used?

**Original Answer (from PDF):**
Cross-validation splits training data into k folds, trains on k−1 folds, and validates on the remaining fold, rotating k times. It gives a more reliable performance estimate than a single train/validation split and reduces variance due to the split. The cv parameter in GridSearchCV/RandomizedSearchCV controls this.

**Project-Specific Viva Answer:**
- We used 5-Fold Cross-Validation. We split our 138,701 rows into 5 folds. We train the ensemble (LightGBM 1, LightGBM 2, CatBoost) on 4 folds and validate on 1 fold, repeating 5 times. Our final test predictions are the average of the predictions from the 5 fold models (fold-averaging), which acts as a bagging ensemble to smooth out variance.

---

### Q148. How many total fits happen in GridSearchCV vs RandomizedSearchCV?

**Original Answer (from PDF):**
GridSearchCV: total fits = (number of all parameter combinations) × cv. RandomizedSearchCV: total fits = n_iter × cv. For example, a grid of 3×3×2=18 combinations with cv=5 → 90 fits in GridSearch; with n_iter=10, cv=5 → 50 fits in RandomizedSearch.

**Project-Specific Viva Answer:**
- GridSearchCV performs an exhaustive search over all parameter combinations. With our large dataset (138,701 rows, 50 features), GridSearchCV would take days to fit. We used `RandomizedSearchCV` for our baseline XGBoost tuning, sampling a fixed number of combinations, which allowed us to find near-optimal hyperparameters (like learning rate, max depth, and regularization) in a fraction of the time.

---

### Q149. What is learning rate and how does it affect model training?

**Original Answer (from PDF):**
Learning rate (η) controls the step size when updating model weights during gradient descent. A high learning rate converges fast but may overshoot the minimum (diverge or oscillate). A low learning rate is stable but slow to converge and may get stuck in local minima. Typical values: 0.01–0.1.

**Project-Specific Viva Answer:**
- In our LightGBM and CatBoost models, we set a slow learning rate of `0.010`. While a larger learning rate (e.g. 0.1) trains faster, it can overshoot the optimal minimum. A slower learning rate, combined with a large number of estimators (15,000) and early stopping, ensures precise optimization and prevents overfitting.

---

### Q150. n_estimators?

**Original Answer (from PDF):**
Decrease n_estimators (fewer trees reduces model complexity and overfitting). Also decrease learning_rate, max_depth, and increase min_child_weight/lambda. Conversely, if underfitting, increase n_estimators.

**Project-Specific Viva Answer:**
- We tuned our models using a 5-Fold Cross-Validation loop. For each fold, we trained on 80% of the data and validated on 20%. Hyperparameters tuned in our best `v18` LightGBM include: `num_leaves=1023` (more capacity), `learning_rate=0.010` (precise training), `reg_alpha=0.2`, `reg_lambda=3.0`, and row/col subsampling, utilizing early stopping (300 rounds) to prevent overfitting.

---

### Q151. What does the 'verbose' parameter do in sklearn models/GridSearchCV?

**Original Answer (from PDF):**
verbose controls the amount of logging output during fitting. verbose=0: silent. verbose=1: minimal (one line per model). verbose=2: one line per fold. Higher values give more detail. Useful for monitoring long HPT runs but should be turned off in production.

**Project-Specific Viva Answer:**
- GridSearchCV performs an exhaustive search over all parameter combinations. With our large dataset (138,701 rows, 50 features), GridSearchCV would take days to fit. We used `RandomizedSearchCV` for our baseline XGBoost tuning, sampling a fixed number of combinations, which allowed us to find near-optimal hyperparameters (like learning rate, max depth, and regularization) in a fraction of the time.

---

### Q152. What is the 'scoring' parameter in GridSearchCV? What does scoring='f1_macro' mean?

**Original Answer (from PDF):**
scoring specifies the metric used to evaluate each hyperparameter combination. 'f1_macro' computes F1 for each class independently and takes the unweighted mean — treats all classes equally regardless of support. Other options: 'accuracy', 'roc_auc', 'f1_weighted', 'neg_mean_squared_error' (for regression, negative because sklearn maximises). Choose scoring based on your task's priority.

**Project-Specific Viva Answer:**
- GridSearchCV performs an exhaustive search over all parameter combinations. With our large dataset (138,701 rows, 50 features), GridSearchCV would take days to fit. We used `RandomizedSearchCV` for our baseline XGBoost tuning, sampling a fixed number of combinations, which allowed us to find near-optimal hyperparameters (like learning rate, max depth, and regularization) in a fraction of the time.

---

### Q153. What happens if you increase/decrease n_estimators when model is overfitting?

**Original Answer (from PDF):**
If overfitting: decrease n_estimators (fewer trees = simpler model). Also decrease max_depth, increase min_samples_leaf, increase regularisation (lambda/alpha in XGBoost). If underfitting: increase n_estimators, increase max_depth, decrease min_samples_leaf.

**Project-Specific Viva Answer:**
- We tuned our models using a 5-Fold Cross-Validation loop. For each fold, we trained on 80% of the data and validated on 20%. Hyperparameters tuned in our best `v18` LightGBM include: `num_leaves=1023` (more capacity), `learning_rate=0.010` (precise training), `reg_alpha=0.2`, `reg_lambda=3.0`, and row/col subsampling, utilizing early stopping (300 rounds) to prevent overfitting.

---

### Q154. What happens if learning rate is too high or too low?

**Original Answer (from PDF):**
Too high: gradient descent overshoots the minimum — loss oscillates or diverges, model may not converge. Too low: convergence is very slow, may get stuck in local minima or require thousands of iterations. Optimal: start with 0.01–0.1 and tune. In boosting, a lower learning rate with more estimators often gives better results.

**Project-Specific Viva Answer:**
- In our LightGBM and CatBoost models, we set a slow learning rate of `0.010`. While a larger learning rate (e.g. 0.1) trains faster, it can overshoot the optimal minimum. A slower learning rate, combined with a large number of estimators (15,000) and early stopping, ensures precise optimization and prevents overfitting.

---

### Q155. What are colsample_bytree and subsample in XGBoost?

**Original Answer (from PDF):**
subsample: fraction of training samples randomly selected for each tree (e.g., 0.8 = 80%). Prevents overfitting by adding randomness (similar to bagging). colsample_bytree: fraction of features randomly selected for each tree. Both add stochasticity — similar to Random Forest's randomness but within a boosting framework.

**Project-Specific Viva Answer:**
- We initially used XGBoost as our primary model (baseline RMSLE 0.2045). However, during ensembling in v14, XGBoost predictions were too highly correlated with LightGBM, receiving 0% weight in our blend optimization. In v15, we replaced XGBoost with a second LightGBM model (`seed=123`) with different hyperparameters, which introduced seed diversity and helped lower our blended OOF RMSLE to 0.19326.

---

### Q156. What is time series cross-validation (TSCV)?

**Original Answer (from PDF):**
Standard k-fold CV randomly shuffles data — invalid for time series (future data leaks into training). Time Series CV (walk-forward validation) preserves temporal order: train on past, validate on future. Each fold expands the training set forward in time. sklearn: TimeSeriesSplit.

**Project-Specific Viva Answer:**
- We used 5-Fold Cross-Validation. We split our 138,701 rows into 5 folds. We train the ensemble (LightGBM 1, LightGBM 2, CatBoost) on 4 folds and validate on 1 fold, repeating 5 times. Our final test predictions are the average of the predictions from the 5 fold models (fold-averaging), which acts as a bagging ensemble to smooth out variance.

---

### Q157. Why should learning rate be 0.01, 0.1, 0.2 and NOT 1, 2, 3?

**Original Answer (from PDF):**
Learning rate η controls the step size in gradient descent. If η ≥ 1, the steps are too large — you overshoot the minimum repeatedly and the loss oscillates or diverges. In boosting (XGBoost/LightGBM), η=1 means each tree gets full weight — no shrinkage — so the model memorises training data (overfits). Small η (0.01–0.2) means each tree contributes a small amount, requiring more trees but leading to better generalisation. The convention is: lower LR + more estimators = better, more regularised model.

**Project-Specific Viva Answer:**
- In our LightGBM and CatBoost models, we set a slow learning rate of `0.010`. While a larger learning rate (e.g. 0.1) trains faster, it can overshoot the optimal minimum. A slower learning rate, combined with a large number of estimators (15,000) and early stopping, ensures precise optimization and prevents overfitting.

---

### Q158. What is the 'cv' parameter in GridSearchCV/RandomizedSearchCV — exactly what does it do?

**Original Answer (from PDF):**
cv specifies the cross-validation strategy: cv=5 (integer): uses 5-fold stratified CV for classification, regular for regression. Each fold: 4 folds for training, 1 fold for validation, rotated 5 times. cv=KFold(n_splits=5, shuffle=True): explicit control. cv=StratifiedKFold(n_splits=5): preserves class distribution per fold. More cv = more reliable estimate but more compute (cv=5 → 5x fits per combination). Total fits for GridSearchCV = n_combinations × cv. For RandomizedSearchCV = n_iter × cv.

**Project-Specific Viva Answer:**
- GridSearchCV performs an exhaustive search over all parameter combinations. With our large dataset (138,701 rows, 50 features), GridSearchCV would take days to fit. We used `RandomizedSearchCV` for our baseline XGBoost tuning, sampling a fixed number of combinations, which allowed us to find near-optimal hyperparameters (like learning rate, max depth, and regularization) in a fraction of the time.

---

### Q159. What is time series cross-validation (TSCV) and why can't you use regular k-fold?

**Original Answer (from PDF):**
Regular k-fold randomly shuffles data — this breaks temporal order, causing future data to leak into training (data leakage). TSCV (walk-forward validation): Fold 1: Train on [1..k], validate on [k+1..k+n]. Fold 2: Train on [1..2k], validate on [2k+1..2k+n]. Each fold only uses past to predict future. sklearn: TimeSeriesSplit(n_splits=5). This gives a realistic estimate of how the model will perform on genuinely unseen future data.

**Project-Specific Viva Answer:**
- We used 5-Fold Cross-Validation. We split our 138,701 rows into 5 folds. We train the ensemble (LightGBM 1, LightGBM 2, CatBoost) on 4 folds and validate on 1 fold, repeating 5 times. Our final test predictions are the average of the predictions from the 5 fold models (fold-averaging), which acts as a bagging ensemble to smooth out variance.

---

## 📁 Model Evaluation Metrics & Statistics
This section covers 26 questions related to model evaluation metrics & statistics.

### Q160. What is random_state=42 and why 42?

**Original Answer (from PDF):**
random_state seeds the random number generator so results are reproducible (same splits, same initialisation). 42 is a cultural reference to 'The Hitchhiker's Guide to the Galaxy' where 42 is 'the answer to everything.' Any integer works; 42 is just convention. Saying only the pop-culture answer without explaining reproducibility is incomplete.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q161. What is a confusion matrix?

**Original Answer (from PDF):**
A confusion matrix is a table comparing actual vs. predicted class labels. For binary classification: TP (correctly predicted positive), TN (correctly predicted negative), FP (actual negative predicted positive), FN (actual positive predicted negative). For multi-class, it is an N×N matrix where rows = actual, columns = predicted.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q162. What is Precision, Recall, and F1-score?

**Original Answer (from PDF):**
Precision = TP / (TP + FP) — of all predicted positives, how many are actually positive. Recall (Sensitivity) = TP / (TP + FN) — of all actual positives, how many did we catch. F1 = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean balancing both. Use Precision when false positives are costly; Recall when false negatives are costly (e.g., disease detection).

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q163. What is macro vs. weighted vs. micro average in classification report?

**Original Answer (from PDF):**
Macro average: compute metric for each class independently, then take unweighted mean — treats all classes equally. Weighted average: mean weighted by support (number of true instances per class) — accounts for class imbalance. Micro average: aggregate all TP, FP, FN globally before computing.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q164. What is 'support' in a classification report?

**Original Answer (from PDF):**
Support is the number of actual occurrences of each class in the test set. It tells you how many samples each metric is based on, helping identify if a class has very few test samples (making its metrics less reliable).

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q165. What is AUC-ROC and how do you interpret it?

**Original Answer (from PDF):**
ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs. False Positive Rate at various thresholds. AUC (Area Under Curve) summarises this in a single number: 0.5 = random classifier, 1.0 = perfect. Higher AUC = better ability to distinguish classes. Useful for imbalanced datasets or when you care about ranking predictions.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q166. Why is accuracy not always a good metric?

**Original Answer (from PDF):**
Accuracy is misleading when classes are imbalanced. E.g., a dataset with 95% negative samples: a model predicting all negatives gets 95% accuracy but detects no positives. F1-score, precision-recall, and AUC-ROC are better choices for imbalanced data.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q167. What is R2-score (coefficient of determination)?

**Original Answer (from PDF):**
R2 measures how well a regression model explains the variance in the target. R2 = 1 − (SS_residual / SS_total). R2=1 is a perfect fit; R2=0 means the model is no better than predicting the mean; R2 can be negative if the model is worse than the mean. It is scale-independent, unlike RMSE.

**Project-Specific Viva Answer:**
- R-squared measures the proportion of price variance explained by the model. Our final ensemble achieved $R^2 \approx 0.905$, meaning it explains 90.5% of the variance in heavy equipment selling prices. The remaining 9.5% represents unexplained noise, missing history, or unpredictable market changes.

---

### Q168. What is the difference between RMSE and R2?

**Original Answer (from PDF):**
RMSE (Root Mean Squared Error) = sqrt(mean((y_pred − y_true)²)). It is in the same unit as the target and penalises large errors more. R2 is unitless and measures the proportion of variance explained. RMSE is better for comparing model errors in absolute terms; R2 is better for explaining goodness of fit.

**Project-Specific Viva Answer:**
- R-squared measures the proportion of price variance explained by the model. Our final ensemble achieved $R^2 \approx 0.905$, meaning it explains 90.5% of the variance in heavy equipment selling prices. The remaining 9.5% represents unexplained noise, missing history, or unpredictable market changes.

---

### Q169. What is MSE and MAE?

**Original Answer (from PDF):**
MSE = mean((y_pred − y_true)²) — penalises large errors quadratically. MAE = mean(|y_pred − y_true|) — equally penalises all errors. MSE is differentiable everywhere (good for gradient-based methods); MAE is more robust to outliers.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q170. What is the OOB (Out-of-Bag) score in Random Forest?

**Original Answer (from PDF):**
Since each tree in a Random Forest is trained on a bootstrap sample (~63% of data), the remaining ~37% (out-of-bag samples) can be used as a validation set for that tree without a separate hold-out. The OOB score aggregates predictions across all trees for each sample's out-of-bag instances, giving a free estimate of generalisation performance.

**Project-Specific Viva Answer:**
- We evaluated Random Forest (a bagging ensemble) in our baseline stage (scoring ~0.2895 val RMSLE). Random Forest trains independent trees on bootstrap samples and averages them, which reduces variance. However, we found that gradient boosted trees (LightGBM and CatBoost) performed much better (OOF RMSLE ~0.193), because boosting sequentially corrects residuals, allowing the models to learn more complex relationships.

---

### Q171. What is Log Loss and why use it instead of MSE for classification?

**Original Answer (from PDF):**
Log Loss (Binary Cross-Entropy) = −(1/n)·Σ[y■·log(p■) + (1−y■)·log(1−p■)]. It penalises confident wrong predictions much more heavily than less confident ones (logarithmic penalty). MSE does not correctly penalise probabilistic outputs for classification — it creates a non-convex loss surface making optimisation harder and does not account for the probabilistic nature of class outputs.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q172. What is hinge loss (used in SVM)?

**Original Answer (from PDF):**
Hinge loss = max(0, 1 − y·f(x)) where y ∈ {−1, +1} and f(x) is the raw model score. If the correct class is predicted with confidence ≥ 1, loss = 0. Otherwise, loss grows linearly. It is used in SVM with LinearSVC or SGDClassifier(loss='hinge'). Unlike log loss, it does not output probabilities.

**Project-Specific Viva Answer:**
- SVM (Support Vector Regression) was not used in the final ensemble because: (1) it has an $O(N^2)$ to $O(N^3)$ training complexity, making it computationally expensive on our 138,701 records, (2) it requires extensive scaling and is highly sensitive to outliers, which tree-based models like LightGBM handle naturally.

---

### Q173. What is F2-score?

**Original Answer (from PDF):**
F_β = (1 + β²) × (Precision × Recall) / (β²·Precision + Recall). F2 (β=2) weights Recall twice as much as Precision. Use when false negatives are twice as costly as false positives — e.g., in medical diagnosis where missing a disease (FN) is worse than a false alarm (FP).

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q174. What is SMAPE?

**Original Answer (from PDF):**
Symmetric Mean Absolute Percentage Error = (100/n)·Σ |actual−predicted| / ((|actual| + |predicted|)/2). It is a scale-independent percentage error metric symmetric between over- and under-estimation. Used in some forecasting competitions. Range: 0–200%.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q175. Precision vs. Recall: which one to use for poisonous apple detection?

**Original Answer (from PDF):**
Use Recall (Sensitivity). In poisonous apple detection, a False Negative (predicting a poisonous apple as safe) could kill someone — this is far worse than a False Positive (flagging a safe apple as poisonous). Recall = TP/(TP+FN) directly measures the proportion of actual positives caught. Maximise Recall to minimise dangerous misses.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q176. What are the formulas for Precision, Recall, Accuracy, and F1?

**Original Answer (from PDF):**
Accuracy = (TP + TN) / (TP + TN + FP + FN) Precision = TP / (TP + FP) Recall = TP / (TP + FN) F1 = 2 × Precision × Recall / (Precision + Recall) = 2TP / (2TP + FP + FN)

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q177. What is a t-test?

**Original Answer (from PDF):**
A t-test determines whether the means of two groups are statistically significantly different. One-sample t-test: compare sample mean to a known value. Independent samples t-test: compare means of two independent groups. Paired t-test: compare means of the same group at two time points. The test statistic follows a t-distribution under the null hypothesis.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q178. What is a Chi-Square test and when is it used in ML?

**Original Answer (from PDF):**
Chi-Square (χ²) test of independence checks whether two categorical variables are statistically independent. In ML: used in SelectKBest(chi2) to select features with the strongest chi-square statistic with the target. Also used in EDA to confirm that categorical features are relevant to the target. Requires non-negative feature values.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q179. What is p-value and alpha level?

**Original Answer (from PDF):**
p-value: probability of observing results at least as extreme as the data, assuming the null hypothesis is true. If p < α (significance level, commonly 0.05), we reject the null hypothesis. α = 0.05 means we accept a 5% chance of false rejection (Type I error). In feature selection, low p-value → feature is significantly associated with target → keep it.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q180. What is Pearson correlation formula?

**Original Answer (from PDF):**
r = Σ[(x■ − x■)(y■ − ■)] / [sqrt(Σ(x■−x■)²) × sqrt(Σ(y■−■)²)]. Range: [−1, +1]. r = +1: perfect positive linear relationship. r = −1: perfect negative linear relationship. r = 0: no linear relationship (may still have non-linear relationship).

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q181. What is 0-centring (zero-centring) and why does it matter?

**Original Answer (from PDF):**
Zero-centring (mean subtraction) shifts the data so its mean is 0. In StandardScaler: z = (x − mean)/std achieves both zero-centring and unit variance. In neural networks, zero-centred inputs speed up convergence because gradients flow more symmetrically. Tanh activation is preferred over sigmoid partly because its output is zero-centred (range −1 to +1), unlike sigmoid (range 0 to 1) which always produces positive gradients causing zig-zag updates.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q182. What is the ROC-AUC curve and why plot it if F1-score is already plotted?

**Original Answer (from PDF):**
ROC curve and F1-score answer different questions. F1-score is computed at a FIXED threshold (usually 0.5) and summarises precision-recall balance at that point. ROC-AUC evaluates performance across ALL possible thresholds — it shows how well the model ranks positives above negatives regardless of threshold choice. Use both: F1 for operational threshold performance; AUC for overall discriminative ability. PR-AUC (Precision-Recall AUC) is better than ROC-AUC for severely imbalanced datasets.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q183. How does gradient boosting build 10 weak learners sequentially? Explain the process.

**Original Answer (from PDF):**
Step 1: Initialise with a simple prediction (e.g., mean of target for regression). Step 2: Compute pseudo-residuals = negative gradient of loss w.r.t. current prediction. Step 3: Train a shallow decision tree (weak learner) to predict these residuals. Step 4: Update prediction: F_m(x) = F_{m-1}(x) + η × tree_m(x), where η = learning rate. Step 5: Repeat steps 2–4 for m = 1 to 10 (or n_estimators). Each tree corrects the previous model's mistakes. Shrinkage (η < 1) prevents overfitting by making small incremental updates.

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

### Q184. Why did you plot an ROC curve if you are already using F1-score? Are they redundant?

**Original Answer (from PDF):**
No, they are complementary. F1-score is computed at a SINGLE threshold (0.5 by default) and tells you performance at that operating point. ROC-AUC evaluates performance ACROSS ALL thresholds — useful when you haven't decided on the optimal threshold. In production, you may tune the threshold based on business requirements (e.g., lower threshold for high-recall fraud detection). ROC-AUC also gives a threshold-independent view of discriminative power. So: use F1 to evaluate at your chosen threshold, AUC to evaluate the model's overall ranking ability.

**Project-Specific Viva Answer:**
- ROC curves are used in classification tasks (like MNIST) to evaluate performance across all decision thresholds. Since our project is a regression task, we did not plot ROC curves. Instead, we evaluated our predictions using residual scatter plots, error histograms, and OOF diagnostics plots.

---

### Q185. What is a multi-class confusion matrix — how do you identify TP, TN, FP, FN for each class?

**Original Answer (from PDF):**
For an N-class confusion matrix (rows=actual, cols=predicted), for each class k: TP_k = cm[k, k] (correctly predicted as class k). FP_k = sum(cm[:, k]) − cm[k, k] (predicted as k but actually other classes). FN_k = sum(cm[k, :]) − cm[k, k] (actually k but predicted as other classes). TN_k = total_samples − TP_k − FP_k − FN_k (everything else correctly not-k). Precision_k = TP_k / (TP_k + FP_k). Recall_k = TP_k / (TP_k + FN_k).

**Project-Specific Viva Answer:**
- Our primary evaluation metric is RMSLE (Root Mean Squared Log Error). RMSLE is scale-invariant and penalises percentage error equally across our 19x price range ($7,500 to $142,000), unlike RMSE which would penalise expensive machines 19x more. We also monitor R-squared ($R^2 \approx 0.905$, explaining 90.5% of variance) and MAE (typical prediction error is ~$5,400).

---

## 📁 Neural Networks & Deep Learning
This section covers 7 questions related to neural networks & deep learning.

### Q186. What are activation functions? Name common ones.

**Original Answer (from PDF):**
Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common ones: ReLU f(x) = max(0, x) — most common, avoids vanishing gradient. Sigmoid σ(x) = 1/(1+e^−x) — outputs [0,1], used in output layer for binary classification. Tanh: outputs [−1,1], stronger than sigmoid. Softmax: converts logits to probabilities for multi-class output.

**Project-Specific Viva Answer:**
- We did not use neural networks in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q187. What is the input and output of ReLU?

**Original Answer (from PDF):**
ReLU takes any real number as input. Output: max(0, x). Negative inputs → 0; positive inputs → same value. It is non-saturating for positive values (no vanishing gradient issue) but can 'die' (always output 0) for negative inputs.

**Project-Specific Viva Answer:**
- Neural networks were not used in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q188. What is the input and output of Tanh?

**Original Answer (from PDF):**
Tanh takes any real number as input. Output range: (−1, 1). It is zero-centred (unlike sigmoid), which can speed up convergence. However, it still suffers from the vanishing gradient problem for very large or small inputs.

**Project-Specific Viva Answer:**
- Neural networks were not used in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q189. What is a neural network and how does it work?

**Original Answer (from PDF):**
A neural network is a series of layers: input layer (receives features), hidden layers (learn intermediate representations via weighted sums + activation functions), output layer (produces final prediction). Training uses forward propagation (compute predictions) then backpropagation (compute gradients via chain rule) and gradient descent to update weights. Each neuron computes: output = activation(Σ w■x■ + b).

**Project-Specific Viva Answer:**
- Neural networks were not used in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q190. How many hidden layers can a neural network have?

**Original Answer (from PDF):**
Theoretically unlimited. A single hidden layer can approximate any continuous function (Universal Approximation Theorem). In practice: 1–2 hidden layers work for most tabular data tasks. Deep networks (many layers) are used for image, text, and audio. More layers = more representational power but harder to train (vanishing gradients, longer training).

**Project-Specific Viva Answer:**
- Neural networks were not used in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q191. What happens if you increase the number of hidden layers?

**Original Answer (from PDF):**
The model becomes more complex and can learn more abstract features. Benefits: better performance on complex tasks (images, NLP). Risks: vanishing/exploding gradients, slower training, higher risk of overfitting, requires more data and regularisation (dropout, batch normalisation).

**Project-Specific Viva Answer:**
- Neural networks were not used in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q192. What are optimisation functions (optimisers) in neural networks?

**Original Answer (from PDF):**
SGD (Stochastic Gradient Descent): basic gradient update. Momentum: accumulates past gradients to accelerate. RMSprop: adapts learning rate per parameter using running average of squared gradients. Adam (Adaptive Moment Estimation): combines Momentum + RMSprop — most popular default. AdaGrad: accumulates all past squared gradients (learning rate decreases over time).

**Project-Specific Viva Answer:**
- Neural networks were not used in our final solution because tree-based ensembles (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

## 📁 Scenario & Business Case Studies
This section covers 15 questions related to scenario & business case studies.

### Q193. What is Precision vs. Recall for fraud detection?

**Original Answer (from PDF):**
In fraud detection: Recall matters most — missing a fraudulent transaction (FN) is costly. But very low Precision means too many false alarms (FP), annoying customers. F1 or Fβ (β>1) balances both. AUC-ROC or PR-AUC is also commonly used since fraud data is heavily imbalanced.

**Project-Specific Viva Answer:**
- Unlike our regression project, fraud detection is a highly imbalanced binary classification problem (e.g. 99.9% normal, 0.1% fraud). My approach would focus on classification-specific techniques: (1) using Stratified K-Fold CV to preserve fraud ratios, (2) applying SMOTE or adjusting class weights, and (3) evaluating using Precision-Recall AUC (PR-AUC) or F1-score instead of accuracy.

---

### Q194. What is normal distribution (Gaussian distribution)?

**Original Answer (from PDF):**
A symmetric bell-shaped distribution defined by mean µ and standard deviation σ. 68% of data falls within µ±σ, 95% within µ±2σ, 99.7% within µ±3σ (empirical rule). Many natural phenomena follow this distribution. Assumed by linear/logistic regression, StandardScaler works best on normally distributed features.

**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q195. House price prediction: which model would you choose and explain its steps?

**Original Answer (from PDF):**
This is a regression problem (predicting continuous price). Best choices: Gradient Boosting (XGBoost/LightGBM) for accuracy, or Ridge/Lasso if interpretability is needed. Steps for linear regression: 1) Collect features (area, location, age, bedrooms). 2) Handle missing values. 3) Encode categoricals. 4) Scale numerics. 5) Train on historical data. 6) Evaluate with RMSE/R2. 7) Predict future price. For time-based prediction, also consider ARIMA or adding time-based features.

**Project-Specific Viva Answer:**
- Predicting heavy equipment auction prices is very similar to predicting house prices. Both are tabular regression tasks with skewed targets and high-cardinality location/region features. Our project's workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q196. Fraud transaction detection: how would you approach it?

**Original Answer (from PDF):**
1) Frame as binary classification (fraud/not-fraud). 2) Handle severe class imbalance (fraud is rare) — use SMOTE, class_weight='balanced', or AUC-PR as metric. 3) Feature engineer: transaction amount, time of day, frequency, location deviation. 4) Use tree-based models (XGBoost, Random Forest) or anomaly detection. 5) Evaluate with Recall (minimise missed fraud) and Precision (minimise false alarms). 6) Consider threshold tuning for business requirements.

**Project-Specific Viva Answer:**
- Unlike our regression project, fraud detection is a highly imbalanced binary classification problem (e.g. 99.9% normal, 0.1% fraud). My approach would focus on classification-specific techniques: (1) using Stratified K-Fold CV to preserve fraud ratios, (2) applying SMOTE or adjusting class weights, and (3) evaluating using Precision-Recall AUC (PR-AUC) or F1-score instead of accuracy.

---

### Q197. Social media company: what recommendations would you give to the owner based on?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q198. your project?

**Original Answer (from PDF):**
Based on your toxicity/comment classification model: 1) Flag high-probability toxic comments for human review before publishing. 2) Prioritise moderation resources on comment threads with highest toxic density. 3) Monitor trends by hour/day (temporal analysis). 4) Target specific toxicity types (hate speech, threats) for different escalation workflows. 5) Use model confidence scores to distinguish borderline vs. clear violations.

**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q199. If a dataset has 200 features, what would be your approach?

**Original Answer (from PDF):**
1) Start with EDA — correlation heatmap, variance threshold (remove near-zero variance features). 2) Use feature importance from a quick Random Forest to shortlist. 3) Apply RFE or SelectKBest to systematically rank features. 4) PCA/TruncatedSVD for dimensionality reduction if features are correlated. 5) L1 regularisation (Lasso) in linear models for automatic feature elimination. 6) Cross-validate performance as you reduce features to find the optimal number.

**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q200. If training accuracy is very low and test accuracy is also low, what is the problem?

**Original Answer (from PDF):**
This is underfitting (high bias). The model is too simple to capture the patterns in the data. Solutions: use a more complex model, add more relevant features, do more feature engineering, reduce regularisation, train for more iterations.

**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q201. If training accuracy is very high but test accuracy is low, what is the problem?

**Original Answer (from PDF):**
This is overfitting (high variance). The model memorises the training data but fails to generalise. Solutions: get more training data, reduce model complexity, add regularisation (L1/L2/dropout), use cross-validation, feature selection, early stopping, or bagging.

**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q202. What is the case of low training accuracy and high test accuracy?

**Original Answer (from PDF):**
This is unusual and often indicates data leakage — test data information has leaked into training, making test performance artificially inflated. Or it may indicate the training set is harder/noisier than the test set. Investigate if preprocessing was fit on the full dataset or if there is a train/test distribution mismatch.

**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q203. If you are in a company and this is your project, what 4–5 insights or recommendations?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- If acting as a business analyst for this project, my recommendations would be: (1) leverage the 21.4% test machine resale overlap to directly map price histories (this is the strongest predictor), (2) price machines based on regional preferences (captured by our Region x SubClass target encodings), and (3) set optimal reserve prices on auctions based on the model's typical prediction error (MedAPE of 10.7%).

---

### Q204. would you give?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q205. Think like a business analyst: 1) Which features are the strongest predictors of your target? 2) What?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q206. the model's predictions (use probabilities, not just hard labels)? 4) Where does the model fail (which?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

### Q207. class/segment has lowest recall)? 5) What business action can be taken for each prediction bucket?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- In our project, heavy equipment price prediction is very similar to house price prediction. Our workflow serves as a blueprint: (1) EDA to identify skewness and resale histories, (2) target log-transformation, (3) K-Fold CV, (4) fold-level target encoding for configuration cards, and (5) ensembling gradient boosted trees (LightGBM, CatBoost).

---

## 📁 Live Coding Challenges
This section covers 37 questions related to live coding challenges.

### Q208. What is feature scaling and why is it important?

**Original Answer (from PDF):**
Feature scaling brings numeric features to a comparable range so that distance-based (KNN, SVM) and gradient-based (logistic regression, neural networks) models are not dominated by features with large magnitudes. Without scaling, a feature with range 0–10,000 would overpower one with range 0–1.

**Project-Specific Viva Answer:**
- We applied `StandardScaler` to normalize numeric features to mean=0 and std=1, but ONLY for our baseline Ridge Regression model. Our final ensemble (LightGBM and CatBoost) consists of tree-based models that split nodes based on thresholds and rank order, making them scale-invariant. Scaling features does not affect tree splits, so it was omitted in our final pipeline.

---

### Q209. What are stop words?

**Original Answer (from PDF):**
Stop words are common words (e.g., 'the', 'is', 'and') that appear frequently but carry little meaning for most NLP tasks. Removing them reduces vocabulary size and noise. However, for some tasks (sentiment, negation detection) stop words may be important and should be kept.

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q210. How do you load a standard sklearn dataset?

**Original Answer (from PDF):**
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine iris = load_iris() # or load_digits() etc. X, y = iris.data, iris.target print(X.shape, y.shape)

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q211. How do you perform train-test split?

**Original Answer (from PDF):**
from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) print(X_train.shape, X_test.shape)

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q212. How do you build a basic sklearn Pipeline?

**Original Answer (from PDF):**
from sklearn.pipeline import Pipeline from sklearn.preprocessing import StandardScaler from sklearn.impute import SimpleImputer from sklearn.linear_model import LogisticRegression pipe = Pipeline([ ('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('model', LogisticRegression()) ]) pipe.fit(X_train, y_train) print(pipe.score(X_test, y_test))

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q213. How do you build a ColumnTransformer for mixed data?

**Original Answer (from PDF):**
from sklearn.compose import ColumnTransformer from sklearn.preprocessing import StandardScaler, OneHotEncoder from sklearn.impute import SimpleImputer from sklearn.pipeline import Pipeline num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]) cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]) preprocessor = ColumnTransformer([ ('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols) ])

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q214. How do you use SimpleImputer standalone?

**Original Answer (from PDF):**
from sklearn.impute import SimpleImputer import numpy as np imp = SimpleImputer(strategy='mean') # or 'median', 'most_frequent', 'constant' imp.fit(X_train[['col']]) X_train[['col']] = imp.transform(X_train[['col']]) X_test[['col']] = imp.transform(X_test[['col']])

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q215. How do you apply RFE to select top 2 features?

**Original Answer (from PDF):**
from sklearn.feature_selection import RFE from sklearn.linear_model import LogisticRegression model = LogisticRegression(max_iter=1000) rfe = RFE(estimator=model, n_features_to_select=2) rfe.fit(X_train, y_train) print('Selected features:', X.columns[rfe.support_].tolist())

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q216. How do you print column names and find null values?

**Original Answer (from PDF):**
print(df.columns.tolist()) # all column names print(df.dtypes) # data types print(df.isnull().sum()) # null count per column print(df.shape) # rows, columns

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q217. How do you separate numerical and categorical columns?

**Original Answer (from PDF):**
num_cols = df.select_dtypes(include='number').columns.tolist() cat_cols = df.select_dtypes(include='object').columns.tolist() print('Numeric:', num_cols) print('Categorical:', cat_cols)

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q218. How do you print value counts / frequency of a column?

**Original Answer (from PDF):**
print(df['column'].value_counts()) # frequency of each unique value print(df['column'].unique()) # unique values print(df['column'].nunique()) # count of unique values

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q219. How do you apply RandomizedSearchCV?

**Original Answer (from PDF):**
from sklearn.model_selection import RandomizedSearchCV param_dist = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 10, None]} rs = RandomizedSearchCV(estimator, param_dist, n_iter=10, cv=5, scoring='f1_macro', random_state=42) rs.fit(X_train, y_train) print(rs.best_params_, rs.best_score_)

**Project-Specific Viva Answer:**
- We used 5-Fold Cross-Validation. We split our 138,701 rows into 5 folds. We train the ensemble (LightGBM 1, LightGBM 2, CatBoost) on 4 folds and validate on 1 fold, repeating 5 times. Our final test predictions are the average of the predictions from the 5 fold models (fold-averaging), which acts as a bagging ensemble to smooth out variance.

---

### Q220. How do you print a classification report?

**Original Answer (from PDF):**
from sklearn.metrics import classification_report, confusion_matrix y_pred = model.predict(X_test) print(classification_report(y_test, y_pred)) print(confusion_matrix(y_test, y_pred))

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q221. How do you change the strategy in SimpleImputer from median to mode?

**Original Answer (from PDF):**
Change strategy='median' to strategy='most_frequent'. Note: 'most_frequent' is the exact string for mode imputation in sklearn.

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q222. How do you replace missing/unknown values in a column?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q223. df['col'] = df['col'].replace({'?': np.nan, 'unknown': np.nan, 'UNKNOWN': 'Other'})

**Original Answer (from PDF):**
df['col'].fillna(df['col'].mode()[0], inplace=True)

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q224. (high-risk vs. low-risk)?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q225. Code to replace '?' values with NaN and handle them?

**Original Answer (from PDF):**
Replace common missing value placeholders and then impute. import pandas as pd import numpy as np df = pd.read_csv('data.csv')

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q226. df.replace('?', np.nan, inplace=True)

**Original Answer (from PDF):**
df.dropna(inplace=True) # Separate features and target X = df.iloc[:, :-1] y = df.iloc[:, -1] # Encode if needed X = pd.get_dummies(X) # Scale X_scaled = StandardScaler().fit_transform(X) model = LogisticRegression(max_iter=1000) rfe = RFE(estimator=model, n_features_to_select=2) rfe.fit(X_scaled, y) selected = X.columns[rfe.support_] print('Top 2 features:', selected.tolist())

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q227. Code to build a confusion matrix manually (numpy) and compute precision/recall?

**Original Answer (from PDF):**
Build and interpret confusion matrix without sklearn. import numpy as np # Example: 4-class confusion matrix cm = np.array([ [678, 78, 82, 95], [ 75,600, 85, 93], [ 45, 89,569, 45], [ 13, 78, 65,700] ]) def calculate_precision(class_idx, cm): tp = cm[class_idx, class_idx] fp = np.sum(cm[:, class_idx]) - tp return tp / (tp + fp) if (tp + fp) > 0 else 0 def calculate_recall(class_idx, cm): tp = cm[class_idx, class_idx] fn = np.sum(cm[class_idx, :]) - tp return tp / (tp + fn) if (tp + fn) > 0 else 0 for i in range(cm.shape[0]): print(f'Class {i}: Precision={calculate_precision(i,cm):.3f}, Recall={calculate_recal l(i,cm):.3f}')

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q228. Code to load iris dataset, apply PCA and show variance explained?

**Original Answer (from PDF):**
Demonstrate PCA with variance explained before and after. from sklearn.datasets import load_iris from sklearn.preprocessing import StandardScaler from sklearn.decomposition import PCA import numpy as np iris = load_iris() X = iris.data # 4 features # Standardize first scaler = StandardScaler() X_scaled = scaler.fit_transform(X) # Apply PCA pca = PCA() pca.fit(X_scaled) print('Explained variance ratio:', pca.explained_variance_ratio_) print('Cumulative:', np.cumsum(pca.explained_variance_ratio_)) # Keep 2 components pca2 = PCA(n_components=2) X_pca = pca2.fit_transform(X_scaled) print('Original shape:', X_scaled.shape) print('After PCA shape:', X_pca.shape)

**Project-Specific Viva Answer:**
- PCA was not used in our final ensemble. Tree models perform binary splits on individual features and are robust to multi-collinearity; applying PCA would discard spatial feature alignment and make tree splits less interpretable. We only utilize PCA in our MNIST baseline coding challenge.

---

### Q229. Code to apply KMeans and find optimal k using elbow method?

**Original Answer (from PDF):**
Find the best k for KMeans clustering. from sklearn.datasets import load_digits from sklearn.cluster import KMeans import matplotlib.pyplot as plt digits = load_digits() X = digits.data inertias = [] k_range = range(2, 15) for k in k_range: km = KMeans(n_clusters=k, random_state=42, n_init=10) km.fit(X) inertias.append(km.inertia_) plt.plot(k_range, inertias, 'bo-') plt.xlabel('Number of clusters (k)') plt.ylabel('Inertia') plt.title('Elbow Method') plt.show() # Choose k at the 'elbow' — where inertia reduction slows

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q230. Code to apply RFE on a dataset and find top 2 features?

**Original Answer (from PDF):**
Full pipeline: load data, clean, apply RFE. import pandas as pd import numpy as np from sklearn.feature_selection import RFE from sklearn.linear_model import LogisticRegression from sklearn.preprocessing import StandardScaler

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q231. # Load data (example with '?' missing values)

**Original Answer (from PDF):**
df = pd.read_csv('data.csv')

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q232. Code to apply MinMaxScaler?

**Original Answer (from PDF):**
Scale features to [0, 1] range. from sklearn.preprocessing import MinMaxScaler scaler = MinMaxScaler(feature_range=(0, 1)) X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test) # Note: Only fit on training data, transform both

**Project-Specific Viva Answer:**
- We did not use `MinMaxScaler` in our final ensemble because tree models are scale-invariant. `MinMaxScaler` is highly sensitive to outliers (it compresses standard values if a huge outlier exists), which would be problematic for features like `OperationalHoursMeter` with extreme max values.

---

### Q233. Code to train ElasticNet and measure performance?

**Original Answer (from PDF):**
ElasticNet combines L1 and L2 regularisation. from sklearn.linear_model import ElasticNet from sklearn.metrics import r2_score, mean_squared_error model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42) # l1_ratio=0: pure Ridge, l1_ratio=1: pure Lasso model.fit(X_train, y_train) y_pred = model.predict(X_test) print('R2:', r2_score(y_test, y_pred)) print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q234. Code to print columns of a dataset and display unique values/frequencies?

**Original Answer (from PDF):**
Common pandas operations tested in coding rounds. import pandas as pd df = pd.read_csv('train.csv') # All column names print(df.columns.tolist()) # Unique values in a column print(df['religion'].unique()) # Value counts (frequency) print(df['label'].value_counts()) # Null values per column print(df.isnull().sum()) # Replace None and NaN with 'other' df['religion'] = df['religion'].replace({None: 'other', 'none': 'other'}) df['religion'].fillna('other', inplace=True)

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q235. Code a simple logistic regression from scratch using NumPy?

**Original Answer (from PDF):**
Implements forward pass and gradient update. import numpy as np class LogisticRegressionScratch: def __init__(self, lr=0.01, epochs=1000): self.lr = lr self.epochs = epochs def sigmoid(self, z): return 1 / (1 + np.exp(-z)) def fit(self, X, y): self.w = np.zeros(X.shape[1]) self.b = 0 for _ in range(self.epochs): z = X @ self.w + self.b y_hat = self.sigmoid(z) # Gradients dw = (1/len(y)) * X.T @ (y_hat - y) db = (1/len(y)) * np.sum(y_hat - y) self.w -= self.lr * dw self.b -= self.lr * db def predict(self, X): return (self.sigmoid(X @ self.w + self.b) >= 0.5).astype(int)

**Project-Specific Viva Answer:**
- Logistic Regression is a parametric linear classification model. It is not suitable for our regression task (continuous price prediction). We evaluated a linear baseline (Ridge Regression) for our continuous price prediction, but it performed poorly (RMSLE = 0.3800) because prices have highly non-linear relationships and complex feature interactions.

---

### Q236. Code to create a heatmap between two encoded categorical columns?

**Original Answer (from PDF):**
Encode categoricals, then plot correlation heatmap. import pandas as pd from sklearn.preprocessing import LabelEncoder import seaborn as sns import matplotlib.pyplot as plt # Encode V1, V2 le = LabelEncoder() df['V1_enc'] = le.fit_transform(df['V1'].astype(str)) df['V2_enc'] = le.fit_transform(df['V2'].astype(str)) # Correlation heatmap corr = df[['V1_enc', 'V2_enc']].corr() sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm') plt.show()

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q237. What does axis=0 vs. axis=1 mean in pandas?

**Original Answer (from PDF):**
axis=0 operates along rows (down the columns). E.g., df.drop('row_label', axis=0) drops a row; df.mean(axis=0) computes column-wise mean. axis=1 operates along columns (across rows). E.g., df.drop('col_name', axis=1) drops a column; df.sum(axis=1) sums across columns for each row.

**Project-Specific Viva Answer:**
- In our project, `axis=0` refers to rows (e.g., when dropping rows with invalid years) and `axis=1` refers to columns (e.g., when dropping `col18` and `col19` or when concatenating train and test dataframes).

---

### Q238. How do you convert a sparse matrix (sklearn output) to a pandas DataFrame?

**Original Answer (from PDF):**
With newer sklearn (>=1.2): set set_output(transform='pandas') on the transformer. Or: pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=feature_names). Or simply: pd.DataFrame(sparse_matrix.toarray(), columns=feature_names). Note: toarray() may be memory-intensive for very large matrices.

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q239. What is causality vs. correlation? Why is this distinction important in ML?

**Original Answer (from PDF):**
Correlation: two variables move together statistically (e.g., ice cream sales and drowning rates both rise in summer). Causality: one variable directly causes another. Correlation does NOT imply causation. In ML, models capture correlations in training data — they predict patterns but do not establish cause-effect. Confounding variables (like 'summer') can create spurious correlations. For business decisions, understanding causality (via A/B testing, causal inference) is essential beyond predictive modelling.

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q240. What is the MNIST pipeline question commonly asked in Level 2 vivas?

**Original Answer (from PDF):**
A commonly asked coding question: load MNIST, split 50:50 with random_state=42, create a Pipeline with StandardScaler + SVM, fit and print classification report. Then redo with PCA added before SVM and compare scores. from sklearn.datasets import fetch_openml from sklearn.model_selection import train_test_split from sklearn.pipeline import Pipeline from sklearn.preprocessing import StandardScaler from sklearn.svm import SVC from sklearn.decomposition import PCA from sklearn.metrics import classification_report mnist = fetch_openml('mnist_784', version=1, as_frame=False) X, y = mnist.data, mnist.target X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42) # Without PCA pipe1 = Pipeline([('scaler', StandardScaler()), ('svm', SVC())]) pipe1.fit(X_train, y_train) print(classification_report(y_test, pipe1.predict(X_test))) # With PCA pipe2 = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=100)), ('svm', SVC())]) pipe2.fit(X_train, y_train) print(classification_report(y_test, pipe2.predict(X_test)))

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q241. What is collinearity? If two columns are highly correlated, what should you do?

**Original Answer (from PDF):**
Collinearity: two features are highly linearly correlated (e.g., height in cm and height in inches). Multicollinearity: three or more features are linearly related. Problems: inflates variance of coefficient estimates in linear models, makes interpretation unreliable, can cause numerical instability. Detection: correlation matrix (|r| > 0.8–0.9), Variance Inflation Factor (VIF > 5–10). Solutions: 1) Drop one of the correlated features. 2) PCA (combines them). 3) Ridge Regression (L2 handles collinearity gracefully). 4) Create a combined feature (ratio or sum). Tree-based models are generally robust to collinearity. Code: Load iris dataset and fit Logistic Regression on the ENTIRE dataset (no train-test split). Sometimes asked to verify you know the sklearn API without necessarily needing a split. from sklearn.datasets import load_iris from sklearn.linear_model import LogisticRegression iris = load_iris() X, y = iris.data, iris.target lr = LogisticRegression(max_iter=200) lr.fit(X, y) print('Classes:', lr.classes_) print('Coefficients shape:', lr.coef_.shape) print('Training accuracy:', lr.score(X, y)) Code: Load iris dataset and fit SGDClassifier on the entire dataset. Tests knowledge of SGDClassifier as an alternative to LogisticRegression. from sklearn.datasets import load_iris from sklearn.linear_model import SGDClassifier from sklearn.preprocessing import StandardScaler iris = load_iris() X, y = iris.data, iris.target # SGD needs scaling X_scaled = StandardScaler().fit_transform(X) sgd = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000) sgd.fit(X_scaled, y) print('Accuracy:', sgd.score(X_scaled, y)) Code: Append the same dataset to itself (double the rows). Stack datasets vertically. Tests numpy/pandas stacking operations. Expected size: (300, 4) for iris (150 rows → 300). import numpy as np from sklearn.datasets import load_iris import pandas as pd iris = load_iris() X = iris.data # shape (150, 4) # Method 1: numpy vstack X_doubled = np.vstack([X, X]) print('Shape after vstack:', X_doubled.shape) # (300, 4) # Method 2: pandas concat df = pd.DataFrame(X) df_doubled = pd.concat([df, df], ignore_index=True) print('Shape after concat:', df_doubled.shape) # (300, 4) Code: Create a random matrix of shape (1000, 5) with values between 0 and 1. Basic numpy random generation test. import numpy as np # Values uniformly distributed in [0, 1) matrix = np.random.rand(1000, 5) print('Shape:', matrix.shape) # (1000, 5) print('Min:', matrix.min()) # ~0 print('Max:', matrix.max()) # ~1 print('First row:', matrix[0]) # If you need exact range [0, 1] # np.random.uniform(0, 1, size=(1000, 5)) Code: Create a cat_df with only object dtype columns, then apply OHE for ≤10 unique values and target encoding for >10. Tests advanced pandas + sklearn encoding logic. import pandas as pd from sklearn.preprocessing import OneHotEncoder # Step 1: Create cat_df with object columns only cat_df = df.select_dtypes(include='object').copy() # Step 2: Separate by unique value count low_card = [c for c in cat_df.columns if cat_df[c].nunique() <= 10] high_card = [c for c in cat_df.columns if cat_df[c].nunique() > 10] print('OHE columns:', low_card) print('Target encode columns:', high_card) # OHE for low cardinality ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') ohe_result = pd.DataFrame( ohe.fit_transform(cat_df[low_card]), columns=ohe.get_feature_names_out(low_card) ) # Target encoding for high cardinality (manual mean encoding) for col in high_card: means = df.groupby(col)['target'].mean() cat_df[col + '_te'] = cat_df[col].map(means) Code: Change TF-IDF Vectorizer to only produce trigrams (3-word sequences). Change ngram_range parameter so only trigrams (n=3 to n=3) are produced. from sklearn.feature_extraction.text import TfidfVectorizer # Unigrams only: ngram_range=(1, 1) # Unigrams + bigrams: ngram_range=(1, 2) # Trigrams only: ngram_range=(3, 3) # Range 6-10: ngram_range=(6, 10) <- what it is called: 6-grams to 10-grams vectorizer = TfidfVectorizer( ngram_range=(3, 3), # ONLY trigrams max_features=50000 ) X_tfidf = vectorizer.fit_transform(corpus) print('Vocabulary size:', len(vectorizer.vocabulary_)) print('Feature matrix shape:', X_tfidf.shape) Code: Write IQR-based outlier detection and treatment manually. Common coding question on outlier handling. import pandas as pd import numpy as np def detect_outliers_iqr(df, col):

**Project-Specific Viva Answer:**
- Features like `MachineAge` and `ManufactureYear` are highly collinear (since age = sale_year - manufacture_year). Linear models like Ridge Regression are highly sensitive to this. However, our final ensemble uses tree-based models (LightGBM and CatBoost) which are robust to collinearity because trees perform splits on single features at a time, simply selecting the one that reduces loss most.

---

### Q242. = df[col].quantile(0.25)

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q243. = df[col].quantile(0.75)

**Original Answer (from PDF):**
lower = Q1 - 1.5 * IQR upper = Q3 + 1.5 * IQR outliers = df[(df[col] < lower) | (df[col] > upper)] print(f'{col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}') print(f'Lower bound: {lower:.2f}, Upper bound: {upper:.2f}') print(f'Outlier count: {len(outliers)}') return lower, upper # Clip (Winsorise) outliers def clip_outliers(df, col): lower, upper = detect_outliers_iqr(df, col) df[col] = df[col].clip(lower, upper) return df Code: Print recall score for a model (instead of accuracy_score). Tests knowledge of sklearn.metrics API. from sklearn.metrics import recall_score, precision_score, f1_score y_pred = model.predict(X_test) # Binary recall = recall_score(y_test, y_pred) print('Recall:', recall) # Multi-class recall_macro = recall_score(y_test, y_pred, average='macro') recall_weighted = recall_score(y_test, y_pred, average='weighted') print('Recall (macro):', recall_macro) print('Recall (weighted):', recall_weighted) Code: Random Forest Classifier with GridSearchCV hyperparameter tuning. Full implementation of RFC with GridSearchCV. from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection import GridSearchCV param_grid = { 'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10], 'max_features': ['sqrt', 'log2'] } rfc = RandomForestClassifier(random_state=42) grid = GridSearchCV( rfc, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1 ) grid.fit(X_train, y_train) print('Best params:', grid.best_params_) print('Best CV score:', grid.best_score_) print('Test score:', grid.score(X_test, y_test)) Code: Load Decision Tree and plot the tree. Visualise a fitted decision tree. from sklearn.datasets import load_iris from sklearn.tree import DecisionTreeClassifier, plot_tree from sklearn.model_selection import train_test_split import matplotlib.pyplot as plt iris = load_iris() X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, test_size=0.2, random_state=42) dt = DecisionTreeClassifier(max_depth=3, random_state=42) dt.fit(X_train, y_train) print('Test accuracy:', dt.score(X_test, y_test)) plt.figure(figsize=(15, 8)) plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True) plt.title('Decision Tree') plt.show()

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

### Q244. Code: StandardScaler on the target variable (y). When and why?

**Original Answer (from PDF):**
Scaling the target y is useful in regression when the target has a large range/skew. Never scale y for classification. After prediction, inverse-transform to get original scale. from sklearn.preprocessing import StandardScaler import numpy as np # Only for regression tasks y_scaler = StandardScaler() y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel() y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel() # Train model on scaled y model.fit(X_train, y_train_scaled) y_pred_scaled = model.predict(X_test) # Inverse transform to get original scale y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel() print('Predictions (original scale):', y_pred_original[:5]) Code: Change HPT from RandomizedSearchCV to GridSearchCV in a notebook. Know how to switch between the two — same param_grid, different class. from sklearn.model_selection import GridSearchCV, RandomizedSearchCV from sklearn.ensemble import GradientBoostingClassifier param_grid = { 'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1] } # RandomizedSearchCV (original) # rs = RandomizedSearchCV(GBC(), param_grid, n_iter=10, cv=5) # Switch to GridSearchCV (change: remove n_iter, use GridSearchCV) grid = GridSearchCV( GradientBoostingClassifier(random_state=42), param_grid, # same param_grid works for both cv=5, scoring='f1_macro', n_jobs=-1 ) grid.fit(X_train, y_train) print('Best:', grid.best_params_)

**Project-Specific Viva Answer:**
- This coding challenge is a common viva hands-on task. For our heavy equipment project, we build Scikit-learn pipelines to automate preprocessing (like simple median imputing and ordinal encoding) and wrap them inside ColumnTransformers. This prevents validation fold leakage during cross-validation, keeping our training and evaluation clean.

---

## 📁 Miscellaneous & Tricky Questions
This section covers 36 questions related to miscellaneous & tricky questions.

### Q245. How should I present my notebook?

**Original Answer (from PDF):**
Walk through your notebook section by section: problem statement → EDA → preprocessing (imputation, encoding, scaling) → feature engineering → model training → hyperparameter tuning → model comparison → insights. Explain WHY you did each step, not just WHAT. Take 20–30 minutes. A thorough explanation leaves less room for additional questions.

**Project-Specific Viva Answer:**
- Start with a one-minute overview: this is a heavy-equipment price regression problem and RMSLE is the metric. Then show EDA, preprocessing, feature engineering, the 5-fold CV/OOF design, the three-model blend, and the score journey. For every step say **what changed, why it was tried, and what happened to the score**. Finish with the best valid result: v18 Kaggle RMSLE 0.18732.

---

### Q246. What insights did you gain from EDA?

**Original Answer (from PDF):**
Mention at least 3–5 concrete findings: class imbalance ratios, distribution shapes (skewed/normal), key correlations, outlier presence, missing value patterns, and any feature relationships that informed your preprocessing or feature engineering choices.

**Project-Specific Viva Answer:**
- My EDA showed a right-skewed price target, so I used `log1p(TargetValue)`; prices span roughly $7,500 to $142,000. It also revealed high-missingness columns, nonlinear depreciation through `MachineAge`, and many high-cardinality equipment configuration fields. Those observations led to missing-value flags/categorical handling, date and age features, and leak-safe target encodings rather than one-hot encoding every configuration.

---

### Q247. When to use L1 vs. L2?

**Original Answer (from PDF):**
Use L1 when you suspect many features are irrelevant (built-in feature selection). Use L2 when all features are potentially relevant but you want to prevent large weights (most common default). ElasticNet is a safe middle ground.

**Project-Specific Viva Answer:**
- In our LightGBM and CatBoost models, we tuned both L1 regularization (`reg_alpha=0.2`) to encourage feature sparsity and L2 regularization (`reg_lambda=3.0` and `l2_leaf_reg=3.0`) to shrink split weights, preventing our deep trees from overfitting to noise.

---

### Q248. What is the bias-variance tradeoff?

**Original Answer (from PDF):**
Bias: error due to overly simplistic assumptions in the learning algorithm (underfitting). Variance: error due to sensitivity to fluctuations in training data (overfitting). Increasing model complexity decreases bias but increases variance. The goal is to find a balance that minimises total error on test data.

**Project-Specific Viva Answer:**
- Bias is error from an overly simple model; variance is error from sensitivity to the training sample. In my project, Ridge underfit nonlinear price patterns, while overly specific/sparse target encodings could overfit. Five-fold OOF evaluation, early stopping, regularization, smoothing and rejecting v19's overfitting SaleMonth interactions were practical ways I controlled this tradeoff.

---

### Q249. What are the types of Machine Learning?

**Original Answer (from PDF):**
Supervised Learning: model learns from labelled data (classification, regression). Unsupervised Learning: finds patterns in unlabelled data (clustering, dimensionality reduction). Semi-supervised: uses a small amount of labelled + large unlabelled data. Reinforcement Learning: agent learns by interacting with an environment and receiving rewards/penalties.

**Project-Specific Viva Answer:**
- The main types are supervised learning (labeled inputs and target), unsupervised learning (structure without labels), semi-supervised learning, and reinforcement learning. My project is supervised learning, specifically regression, because each training row has a known auction selling price.

---

### Q250. What is the difference between Machine Learning and Deep Learning?

**Original Answer (from PDF):**
ML algorithms typically require manual feature engineering and work well on structured tabular data. Deep Learning uses multi-layer neural networks that automatically learn hierarchical features from raw data (images, text, audio). DL needs much more data and compute but often outperforms traditional ML on unstructured data.

**Project-Specific Viva Answer:**
- Deep learning was not used because gradient boosted trees (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q251. What is a loss function?

**Original Answer (from PDF):**
A loss function measures how far the model's predictions are from the true values. The model is trained by minimising this function via gradient descent. For classification: Cross-Entropy / Log Loss. For regression: MSE, MAE, Huber Loss. The loss function must be differentiable for gradient-based optimisation.

**Project-Specific Viva Answer:**
- A loss function quantifies error during model fitting and is what the optimiser tries to reduce. In this project I model `log1p(price)` and assess the resulting predictions with RMSLE; on this transformed scale, squared error is closely aligned with the RMSLE objective. Do not confuse a training loss with the competition reporting metric.

---

### Q252. What is K-Means clustering and how does it work?

**Original Answer (from PDF):**
K-Means partitions data into k clusters. Steps: 1) Randomly initialise k centroids. 2) Assign each point to the nearest centroid. 3) Recompute centroids as the mean of assigned points. 4) Repeat 2–3 until centroids converge. Limitation: requires specifying k, sensitive to initialisation and outliers.

**Project-Specific Viva Answer:**
- In our project, we solved a supervised regression task (predicting auction prices) where K-Means or unsupervised clustering was not used. However, if we wanted to segment used machinery into tiers based on wear and age (e.g. 'collector items' vs 'heavy workload' machines), we could run K-Means on `MachineAge` and `OperationalHoursMeter` before feeding them into LightGBM.

---

### Q253. What is the elbow method for finding optimal k in K-Means?

**Original Answer (from PDF):**
Plot inertia (sum of squared distances to nearest centroid) vs. number of clusters k. The 'elbow' — where the rate of decrease sharply slows — suggests the optimal k. Beyond the elbow, adding more clusters gives diminishing returns.

**Project-Specific Viva Answer:**
- In our project, we solved a supervised regression task (predicting auction prices) where K-Means or unsupervised clustering was not used. However, if we wanted to segment used machinery into tiers based on wear and age (e.g. 'collector items' vs 'heavy workload' machines), we could run K-Means on `MachineAge` and `OperationalHoursMeter` before feeding them into LightGBM.

---

### Q254. What is oversampling and when does it cause problems?

**Original Answer (from PDF):**
Oversampling increases the minority class representation. Random oversampling duplicates minority samples (risks exact duplicate noise). SMOTE creates synthetic samples. Problem: if applied before train/test split, synthetic minority test samples leak training information → overly optimistic evaluation. Always apply oversampling only within the training fold.

**Project-Specific Viva Answer:**
- Oversampling duplicates or synthesizes minority-class examples. It can overfit duplicated cases and must be performed only inside training folds. I did not use it because the task is continuous-price regression rather than imbalanced classification.

---

### Q255. What is the difference between a tree and a graph?

**Original Answer (from PDF):**
A Graph is a general structure: a set of nodes (vertices) connected by edges. Edges can form cycles, nodes can have any number of connections, no hierarchy required. A Tree is a special CONNECTED, ACYCLIC graph with a hierarchical structure: one root node, each node (except root) has exactly one parent, no cycles. In ML: Decision Trees are trees (hierarchical, acyclic). Neural networks with skip connections (ResNets) form DAGs (directed acyclic graphs). Knowledge graphs are general graphs.

**Project-Specific Viva Answer:**
- LightGBM and CatBoost build decision trees, which are hierarchical, directed, acyclic graphs. A tree must have a single root node and no cycles, meaning predictions flow in a single direction from top to bottom.

---

### Q256. What is gradient vanishing?

**Original Answer (from PDF):**
During backpropagation, gradients are multiplied through each layer via the chain rule. With many layers and sigmoid/tanh activations (which saturate near 0 or 1), gradients become exponentially small as they propagate back — early layers learn very slowly. Solutions: ReLU activation (does not saturate for positive values), batch normalisation, residual connections (skip connections), gradient clipping.

**Project-Specific Viva Answer:**
- Deep learning was not used because gradient boosted trees (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q257. What is a generative vs. discriminative model?

**Original Answer (from PDF):**
Discriminative model: directly models the boundary between classes P(y|X). Examples: Logistic Regression, SVM, Neural Networks, Decision Trees. Generative model: models the joint distribution P(X,y) or P(X|y) and P(y), then uses Bayes' theorem. Examples: Naive Bayes, Gaussian Mixture Models, GANs. Logistic Regression is discriminative despite being named 'regression.'

**Project-Specific Viva Answer:**
- Deep learning was not used because gradient boosted trees (LightGBM and CatBoost) outperform neural networks on tabular datasets, are faster to train, and require less preprocessing (like standard scaling). However, activation functions like ReLU and Tanh are standard for neural nets; Tanh is zero-centered [-1,1], while ReLU solves the vanishing gradient problem.

---

### Q258. What is the difference between CountVectorizer and TF-IDF?

**Original Answer (from PDF):**
CountVectorizer simply counts how many times each word appears in a document (raw term frequency). TF-IDF = TF × IDF — it down-weights common words (high document frequency) and up-weights rare, informative words. TF-IDF generally gives better results for text classification because common words like 'the' or 'is' get penalised and do not dominate.

**Project-Specific Viva Answer:**
- Our project is tabular and doesn't contain text features, so NLP techniques like TF-IDF or CountVectorizer were not used. Instead, we used target encoding and frequency encoding for categorical columns.

---

### Q259. What is BERT and when would you use it?

**Original Answer (from PDF):**
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer-based language model. It understands context bidirectionally (unlike TF-IDF which is bag-of-words). Use BERT for complex NLP tasks (sentiment analysis, NER, Q&A;) when you have sufficient compute and data. For simple classification with limited compute, TF-IDF + traditional ML is often preferred.

**Project-Specific Viva Answer:**
- Our used machinery dataset consists of structured tabular columns (numeric hours, years, and categorical spec configurations) and does not contain natural language text. Thus, NLP techniques like TF-IDF, Tokenization, BERT, or word clouds were not used. Instead, we focused on Target Encoding and frequency encoding.

---

### Q260. What is the difference between load_* and fetch_* functions in sklearn?

**Original Answer (from PDF):**
load_* (e.g., load_iris): loads a small dataset bundled with sklearn from local memory — fast, no internet needed. fetch_* (e.g., fetch_20newsgroups, fetch_openml): downloads larger datasets from the internet on first call, then caches locally. If you use load_ when fetch_ is needed, you will get an error.

**Project-Specific Viva Answer:**
- In our notebooks, we utilized Pandas for data loading (`pd.read_csv`), sorting, and grouping, and Scikit-learn for building preprocessing pipelines and ColumnTransformers. For example, `axis=1` was used to drop columns with >99% missing values (`col18` and `col19`), and sparse matrices were converted to dense arrays for compatibility with tree models.

---

### Q261. What is the StackingClassifier?

**Original Answer (from PDF):**
Stacking (stacked generalisation) trains multiple base models (level-0 learners) and feeds their predictions as features to a meta-learner (level-1 learner). Unlike bagging (parallel, same algorithm) or boosting (sequential), stacking combines diverse algorithms. The meta-learner learns how to best combine base model predictions. Can be powerful but risks overfitting if not done with cross-validated predictions.

**Project-Specific Viva Answer:**
- We used a blended ensemble (simple weighted average of LGB1, LGB2, and CatBoost predictions) rather than a StackingClassifier to reduce model training time and prevent overfitting.

---

### Q262. What is collinearity and how do you handle it?

**Original Answer (from PDF):**
Collinearity (multicollinearity) occurs when two or more features are highly correlated with each other (not necessarily with the target). It inflates variance of coefficient estimates in linear models, making them unstable and hard to interpret. Detection: correlation matrix, Variance Inflation Factor (VIF). Solutions: drop one of the correlated features, use PCA, or use Ridge Regression (L2 regularisation).

**Project-Specific Viva Answer:**
- Features like `MachineAge` and `ManufactureYear` are highly collinear (since age = sale_year - manufacture_year). Linear models like Ridge Regression are highly sensitive to this. However, our final ensemble uses tree-based models (LightGBM and CatBoost) which are robust to collinearity because trees perform splits on single features at a time, simply selecting the one that reduces loss most.

---

### Q263. What is the difference between a metric and a loss function?

**Original Answer (from PDF):**
A loss function is used DURING training to optimise the model — it must be differentiable so gradients can be computed (e.g., cross-entropy, MSE, hinge loss). A metric is used AFTER training to evaluate model performance for humans — it does not need to be differentiable (e.g., accuracy, F1-score, AUC-ROC). Example: you train a classifier with log-loss but evaluate it using F1-score. The same function can be both (e.g., MSE is both a loss and a regression metric), but they serve different roles.

**Project-Specific Viva Answer:**
- A loss is optimized during training; a metric is reported to judge performance. They may be the same, but need not be. Here the business/competition metric is RMSLE. By training on `log1p(TargetValue)`, errors are evaluated on a scale aligned with RMSLE, then predictions are restored with `expm1()`.

---

### Q264. If a dataset size is increased, which model would perform well?

**Original Answer (from PDF):**
Models that scale well with more data: Deep Learning (neural networks) — benefits most from large data. Gradient Boosting (XGBoost/LightGBM) — continues to improve. Logistic Regression — also scales well but hits a ceiling (linear model). Models that struggle: KNN — prediction time O(n), becomes very slow. SVM with RBF kernel — O(n²) to O(n³) training complexity. General rule: more data helps all models, but neural networks and boosting benefit most.

**Project-Specific Viva Answer:**
- There is no universal winner; performance depends on signal, feature types, compute budget and validation. For this large tabular dataset with many categorical variables, gradient-boosted trees (LightGBM and CatBoost) were strong and practical. I would revalidate after adding data rather than promise that a particular model must improve.

---

### Q265. Can you solve a regression problem as a classification problem?

**Original Answer (from PDF):**
Yes, by binning the continuous target into discrete buckets (e.g., price ranges: low/medium/high). This trades some precision for interpretability. The reverse (classification as regression) is also possible — treat class labels as numbers and threshold the output, though log-odds modelling (logistic regression) is more principled. The examiner's specific question: 'can you solve this regression problem using classification?' - answer yes, by discretising the target variable into meaningful ranges.

**Project-Specific Viva Answer:**
- Yes, by binning a continuous target, but that loses price precision and creates arbitrary thresholds. I did not do this because the task requires a continuous auction price and Kaggle uses RMSLE. Regression preserves the information needed to minimize proportional price error.

---


---

### Q266. Is data imbalanced or balanced — how do you know?

**Original Answer (from PDF):**
Check class distribution: df['target'].value_counts() or df['target'].value_counts(normalize=True). Rule of thumb: if the minority class is < 10–20% of the dataset, consider it imbalanced. Visual: bar plot of class counts. In the viva context, check your training data's class distribution before and after train_test_split. If imbalanced, use stratify=y in split, class_weight='balanced' in models, SMOTE, or appropriate metrics (F1, AUC) instead of accuracy.

**Project-Specific Viva Answer:**
- For classification, inspect counts/proportions per class and use class-aware metrics. My target has no classes, so class imbalance does not apply. The relevant distribution issue was a right-skewed continuous price target; I addressed it with `log1p`, not SMOTE or class weights.

---

### Q267. What is autocorrelation in time series?

**Original Answer (from PDF):**
Autocorrelation measures the correlation of a time series with its own past values (lagged versions of itself). ACF (AutoCorrelation Function) plots correlation at each lag. PACF (Partial ACF) shows direct correlation with each lag after removing effects of intermediate lags. Used to: determine if a time series is stationary, choose ARIMA parameters (p from PACF, q from ACF), and detect seasonality patterns.

**Project-Specific Viva Answer:**
- While we did not run formal hypothesis tests like t-tests or Chi-Square tests in our final pipeline, we did use statistical indicators like Pearson correlation for linear relations (e.g. year made vs. price) and Eta-squared ($\eta^2$) to analyze categorical associations. This statistics-driven EDA directly led to encoding powerful predictors like `Spec_SubClass` ($\eta^2 = 0.24$).

---

### Q268. How do you compare time taken by two models?

**Original Answer (from PDF):**
Use Python's time module or %%timeit magic in Jupyter: import time start = time.time() model.fit(X_train, y_train) end = time.time() print(f'Training time: {end-start:.2f}s') Or use timeit. For sklearn, you can also check fit_time_ from cross_validate(). Key insight: SVM O(n²–n³), KNN O(1) train but O(n) predict, tree models O(n log n), neural networks depend on architecture and epochs. import time # Model 1 start = time.time() model1.fit(X_train, y_train) time1 = time.time() - start # Model 2 start = time.time() model2.fit(X_train, y_train) time2 = time.time() - start print(f'Model1: {time1:.3f}s, Model2: {time2:.3f}s')

**Project-Specific Viva Answer:**
- Measure comparable wall-clock training/inference time under the same hardware, data split, folds, preprocessing and early-stopping settings; report both time and validation score. I prioritized the blend because its RMSLE improvement justified its additional compute, while 5-fold CV also makes the comparison more reliable.

---

### Q269. What is the difference between authentication and authorization?

**Original Answer (from PDF):**
Authentication: verifying WHO you are (identity verification — login with username/password, OTP, biometrics). Example: checking your student ID at the viva. Authorization: verifying WHAT you are

**Project-Specific Viva Answer:**
- Since this project is a Kaggle machine learning challenge, we focused on training and ensembling. If we were to deploy this model as a production web API, authentication (verifying a user's identity) and authorization (verifying their permissions to predict or view price logs) would be standard requirements.

---

### Q270. allowed to do (permissions — after login, can you access admin panel?). Example: after ID verification,?

**Original Answer (from PDF):**


**Project-Specific Viva Answer:**
- Authentication and authorization are web development security concepts. Since our notebooks focus on model training and Kaggle prediction submissions, security protocols were not part of the pipeline. However, if we deployed this model as a production web API for auction sellers, verifying identity (authentication) and user prediction quotas (authorization) would be necessary.

---

### Q271. can you submit to this form? Auth flow: Authenticate first → then Authorize. JWT tokens handle both in?

**Original Answer (from PDF):**
modern APIs. Explain the decision tree building process step by step. 1) Start with all training data at the root node. 2) For each feature, evaluate all possible split thresholds. 3) Compute impurity reduction (Information Gain = Parent Impurity − weighted child impurities) for each feature-threshold combination. 4) Choose the feature and threshold with the highest Information Gain as the split. 5) Recursively repeat for each child node. 6) Stop when: node is pure (all one class), max_depth is reached, node has fewer samples than min_samples_split, or Information Gain < threshold. 7) Remaining nodes become leaves; assign majority class (classification) or mean (regression).

**Project-Specific Viva Answer:**
- Since this project is a Kaggle machine learning challenge, we focused on training and ensembling. If we were to deploy this model as a production web API, authentication (verifying a user's identity) and authorization (verifying their permissions to predict or view price logs) would be standard requirements.

---

### Q272. What is the relationship between information gain and entropy?

**Original Answer (from PDF):**
Entropy H(S) = −Σ p■·log■(p■) measures impurity of set S. Information Gain IG(S, A) = H(S) − Σ (|S■|/|S|)·H(S■) where the sum is over all values v of attribute A, and S■ is the subset where A=v. IG measures how much entropy is REDUCED by splitting on attribute A. High IG = attribute A separates classes well = good split. Gini Index is an alternative impurity measure (doesn't use log, slightly faster).

**Project-Specific Viva Answer:**
- Entropy measures class-label impurity; information gain is the reduction in entropy after a split: parent entropy minus weighted child entropy. These are primarily taught for classification trees. My regression trees instead choose splits that reduce a regression loss/variance, but the same principle applies: choose splits that make child nodes more homogeneous.

---

### Q273. What is the range of the Gini index and how is it calculated?

**Original Answer (from PDF):**
Gini = 1 − Σ p■² where p■ is the fraction of samples belonging to class i. Range: 0 (pure node, all one class) to 0.5 (binary, maximum impurity, 50/50 split). For multi-class with c classes: max Gini = 1 − 1/c (when classes are equally distributed). Lower Gini = purer node. Decision Tree minimises weighted Gini of child nodes. Gini vs Entropy: both work similarly; Gini is faster (no log), Entropy can be slightly more informative for multi-class.

**Project-Specific Viva Answer:**
- For classification, Gini impurity is `1 - sum(p_k^2)`, ranging from 0 (pure) to `1 - 1/K` for K equally likely classes; for binary classes its maximum is 0.5. I did not calculate Gini in my project because it is regression; the boosted trees optimize regression loss on log-price.

---

### Q274. What is Reinforcement Learning and give examples?

**Original Answer (from PDF):**
Reinforcement Learning (RL) is a type of ML where an AGENT learns to make decisions by interacting with an ENVIRONMENT. The agent receives REWARDS for good actions and PENALTIES for bad ones, and learns a POLICY (strategy) to maximise cumulative reward. No labelled dataset needed — the agent explores and exploits. Examples: game-playing AI (AlphaGo, chess engines), robot navigation, recommendation systems, autonomous driving, trading algorithms. Key terms: state, action, reward, policy, value function, Q-learning, policy gradient. Explain all types of Machine Learning with examples. Supervised Learning: model learns from labelled (X, y) data. Classification: predict category (spam/not-spam, disease/no-disease). Regression: predict continuous value (house price, stock price). Unsupervised Learning: finds patterns in unlabelled data. Clustering: group similar points (KMeans, DBSCAN, Hierarchical). Dimensionality Reduction: PCA, t-SNE, Autoencoders. Association: find rules (Apriori — market basket analysis). Semi-supervised: small labelled + large unlabelled (e.g., self-training). Self-supervised: create labels from data itself (e.g., BERT predicts masked words). Reinforcement Learning: agent learns via reward/penalty feedback.

**Project-Specific Viva Answer:**
- Reinforcement learning learns a policy through rewards from interactions, for example game playing, robot control or ad bidding. I did not use it: the project has a fixed labeled table of past auctions, so supervised regression is the correct formulation.

---

### Q275. Why random_state=42? What if the examiner says they are 'fed up' with the Hitchhiker's Guide?

**Original Answer (from PDF):**
The real answer: random_state is any integer seed for the random number generator — it ensures reproducibility (same results every run). 42 is a convention in the ML community borrowed from 'The Hitchhiker's Guide to the Galaxy' (42 = answer to everything). But if the examiner pushes: 'Any non-negative integer works. 42 is chosen by convention in the data science community for no mathematical reason — you could use 0, 1, 2023, or any number. The important property is that once set, results are reproducible for debugging, sharing, and paper replication.'



**Project-Specific Viva Answer:**
- `random_state=42` fixes pseudo-random operations such as shuffled KFold splits so results can be reproduced and compared fairly across notebook versions. The number itself has no special ML property; any documented fixed seed would work. I also used a different seed for the second LightGBM to add controlled ensemble diversity.

---


---

### Q276. In diabetes dataset, which metric is the best choice and why?

**Original Answer (from PDF):**
For diabetes prediction (binary: diabetic/not-diabetic): Recall (Sensitivity) is most important — missing a diabetic patient (False Negative) is dangerous (they don't get treatment). High Recall minimises FN. But very low Precision means too many false alarms (unnecessary anxiety, costs). Best choice: F1-score or F2-score (weights Recall higher than Precision). Also report AUC-ROC for threshold-independent evaluation. Avoid accuracy if data is imbalanced (most people are non-diabetic).

**Project-Specific Viva Answer:**
- Choose the metric from the task and cost of mistakes. If the diabetes target is continuous disease progression, use MAE/RMSE/RMSLE as appropriate; if it is diagnosis classification, use recall, precision, F1, ROC-AUC or PR-AUC based on false-negative cost and class balance. This is separate from my project, where RMSLE was specified by Kaggle for price regression.

---

### Q277. What is SARIMA and when to use it over ARIMA?

**Original Answer (from PDF):**
SARIMA (Seasonal ARIMA) = ARIMA(p,d,q)(P,D,Q,s) where P,D,Q are seasonal counterparts of p,d,q and s is the seasonal period (e.g., s=12 for monthly data, s=7 for weekly). Use SARIMA when data shows clear seasonality (e.g., retail sales spike every December). ARIMA handles non-seasonal patterns; SARIMA handles both. Identify seasonality via: ACF/PACF plots (spikes at multiples of s), time series decomposition (seasonal_decompose), or visual inspection.

**Project-Specific Viva Answer:**
- ARIMA models non-seasonal autoregressive, differencing and moving-average effects. SARIMA adds seasonal AR, differencing and MA terms, so use it when a univariate time series has stable repeating seasonality. I did not use ARIMA/SARIMA because this is tabular price regression with many asset attributes, not a single ordered time-series forecast.

---

### Q278. What is log transformation of the target variable and why use it?

**Original Answer (from PDF):**
If the target variable is right-skewed (many low values, few very high values), log transform (y_new = log(y+1)) compresses the range, making it more normally distributed. Benefits: reduces the effect of extreme outliers on MSE/RMSE loss, can improve linear model performance, predictions are more stable. After predicting log(y), apply inverse transform: y = exp(y_pred) − 1. Power transform (Box-Cox) is a generalisation: y^λ for optimal λ found by MLE. sklearn: PowerTransformer(method='box-cox') or FunctionTransformer(np.log1p).

**Project-Specific Viva Answer:**
- I used `y_log = log1p(TargetValue)` because auction prices are right-skewed and the evaluation metric is RMSLE. The transform compresses large prices and makes training focus on proportional rather than absolute error. After predicting, I use `expm1(pred_log)` and clip only impossible/extreme outputs before submission.

---
