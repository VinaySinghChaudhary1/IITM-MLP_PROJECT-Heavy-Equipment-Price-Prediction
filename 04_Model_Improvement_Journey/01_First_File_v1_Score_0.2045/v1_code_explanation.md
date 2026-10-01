# 📖 v1 Code Explanation — Complete Story Guide
## `22f2001153-notebook-1.ipynb` — First File Code
### Heavy Equipment Price Prediction | Kaggle Score: ~0.20845 (Baseline)

---

> **How to read this guide:** Imagine you are trying to predict the price of a used bulldozer or excavator at an auction. You have data about thousands of past auctions. Your goal is to build a "smart calculator" that uses historical data to predict prices for new auctions. This notebook is the FIRST version — building the entire ML pipeline from scratch, comparing 5 different models, and submitting the best one.

---

## 🏗️ Competition Overview

| Item | Value |
|---|---|
| Problem | Predict auction price of heavy industrial machinery |
| Metric | RMSLE (Root Mean Squared Log Error) — **lower is better** |
| Competition Target | RMSLE ≤ 0.85 |
| Train rows | 138,701 |
| Test rows | 15,000 |
| Best model (baseline) | XGBoost (tuned) |
| Val RMSLE | ~0.2045 |
| Kaggle Score | ~0.20845 |

---

## ⚠️ IMPORTANT: How to Submit on Kaggle

1. Go to Kaggle → Competition → "Code" → **"+ New Notebook"**
2. Click **"File" → "Import Notebook"** → Upload `22f2001153-notebook-1.ipynb`
3. Click **"Add Data"** → Add competition dataset
4. Click **"Run All"** (takes ~1-2 hours)
5. Submit the generated `submission.csv`

---

## 🔧 CELL 1 — Default Kaggle Boilerplate (Skipped)

Standard Kaggle template cell — lists competition files. No changes needed.

**Output:**
```
/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/sample_submission.csv
/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/train.csv
/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/metadata.csv
/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/test.csv
```

---

## 📦 CELL 2 — Markdown: Title + Overview

This is a markdown cell (text explanation, no code). It describes:
- The problem: predict auction price of heavy machinery
- Metric: RMSLE ≤ 0.85
- Data: 138,701 train rows, 15,000 test rows
- The planned ML pipeline: EDA → Feature Engineering → Model Comparison → Tuning → Submission

---

## 📚 CELL 3 — Load All Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
```

### 🔎 What is this doing?

Opens the entire toolbox of Python packages we need.

| Library | Purpose |
|---|---|
| numpy, pandas | Fast math and table operations |
| matplotlib, seaborn | Charts and visualizations |
| sklearn (multiple) | ML pipeline, preprocessing, models |
| Ridge | Linear regression model |
| RandomForestRegressor | Tree ensemble model (bagging) |
| HistGradientBoostingRegressor | Fast gradient boosting |
| DummyRegressor | Dummy baseline (predicts the mean) |
| XGBRegressor | XGBoost — boosting algorithm |
| lightgbm | LightGBM — fast gradient boosting |

**Output:**
```
✅ All libraries loaded!
   NumPy 2.4.6 | Pandas 2.3.3
```

---

## 📂 CELLS 4-5 — Load the Competition Data

```python
train_df = pd.read_csv('/kaggle/input/.../train.csv', low_memory=False)
test_df  = pd.read_csv('/kaggle/input/.../test.csv',  low_memory=False)
sample   = pd.read_csv('/kaggle/input/.../sample_submission.csv')
metadata = pd.read_csv('/kaggle/input/.../metadata.csv')
```

### 🔎 What is this doing?

Reads all four competition files from Kaggle's cloud storage into memory.

| File | Purpose |
|---|---|
| `train.csv` | 138,701 historical auctions WITH prices — for training |
| `test.csv` | 15,000 auctions WITHOUT prices — we predict these |
| `sample_submission.csv` | Shows expected format for our predictions |
| `metadata.csv` | Extra info about columns |

**Output:**
```
============================================================
  DATA LOADING COMPLETE
============================================================
  train_df : 138,701 rows × 50 columns
  test_df  :  15,000 rows × 49 columns
  sample   :  15,000 rows ×  2 columns
  metadata :    ...
============================================================
```

---

## 👁️ CELLS 6-8 — First Look at the Data

### CELL 7 (Markdown) — What is happening?

Explains what we're about to do: peek at data to understand the structure.

### CELL 8 (Code) — `train_df.head(3)` + Info

Shows the first 3 rows and column types.

**Key observations:**
- Column types: some text (object), some numbers (float/int)
- `TargetValue` = the price we want to predict
- `TransactionID`, `AssetID` = ID columns (not useful as features)
- Many columns contain text like "Wheel Loader", "Series II", etc.
- Some columns have NaN (missing values) — common in real auction data

---

## 📐 CELLS 9-10 — Dataset Shape

```python
print("Train Dataset shape:")
display(train_df.shape)  # → (138701, 50)
print("Test Dataset shape:")
display(test_df.shape)   # → (15000, 49)
```

**Output:**
```
Train Dataset shape: (138701, 50)
Test Dataset shape:  (15000, 49)
```

**Why 49 columns in test vs 50 in train?** The test set doesn't have `TargetValue` (the price) — that's what we need to predict!

---

## 📊 CELLS 11-12 — Numerical Summary Statistics

```python
display(train_df.describe())
```

### 🔎 What is this doing?

The `describe()` function shows statistical summary of all numerical columns:

| Statistic | Meaning |
|---|---|
| count | How many non-missing values |
| mean | Average value |
| std | Standard deviation (spread) |
| min | Smallest value |
| 25%/50%/75% | Quartiles |
| max | Largest value |

**Key observations from actual output:**
- `TargetValue` range: $7,500 to $142,000 (huge range!)
- Mean price: ~$41,521
- Wide std: ~$26,361 (prices are very spread out)
- `ProductConfigID` has many unique values (very high cardinality)

---

## 📋 CELLS 13-14 — Column Names

```python
train_df.columns
```

Shows all 50 column names — helps understand what data is available.

**Key columns:** `TransactionID`, `AssetID`, `TargetValue`, `ProductConfigID`, `ManufactureYear`, `TransactionDate`, `OperationalHoursMeter`, `InventoryGroupCategory`, `FunctionalClassification`, `RegionCode`, plus many text specification columns (`Spec_*`, `col*`).

---

## 📝 CELLS 15-16 — Submission Format Check

```python
sample.columns  # ['TransactionID', 'TargetValue']
```

Confirms the submission file needs exactly 2 columns: `TransactionID` and `TargetValue`. We must predict `TargetValue` for every row in test.csv.

---

## 🎯 CELLS 17-18 — First Baseline Submission (Mean Prediction)

```python
# Predict the training mean for ALL test rows
mean_price = train_df['TargetValue'].mean()  # ≈ $41,521
submission = pd.DataFrame({
    'TransactionID': test_df['TransactionID'],
    'TargetValue': mean_price  # Same prediction for everyone!
})
submission.to_csv('submission.csv', index=False)
```

### 🔎 What is this doing?

**Story:** Before building any smart model, we establish a **baseline** — the simplest possible prediction. We predict the SAME average price ($41,521) for every single test row.

This is our starting point. Any real model must beat this!

**Output:**
```
Done!
   TransactionID  TargetValue
0        1139307     41521.87
1        1139419     41521.87
2        1139482     41521.87
...
```

**This baseline would give an RMSLE of ~0.6694** — the DummyRegressor score we saw later.

---

## 🔍 CELL 19 — Markdown: EDA Section Header

Describes what the next section will cover:
1. Target distribution analysis
2. Missing value analysis  
3. Categorical features bivariate analysis
4. Numerical features distribution
5. Correlation analysis
6. Price by machine type and region

---

## 📈 CELLS 20-22 — EDA Step 1: Target Variable Analysis

```python
# Plot TargetValue distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train_df['TargetValue'], bins=50)  # Raw prices
axes[1].hist(np.log1p(train_df['TargetValue']), bins=50)  # Log prices
```

### 🔎 What is this doing?

Plots two histograms of the target variable (price):

**Chart 1 — Raw Price Distribution:**
- X-axis: price in dollars
- Y-axis: count of auctions at that price
- Shape: Right-skewed (most machines are cheap, a few are very expensive)
- Problem: Very uneven distribution makes modeling harder

**Chart 2 — Log Price Distribution:**
- X-axis: log(1 + price)  
- Y-axis: count
- Shape: Bell-curve (much more symmetric and normal!)
- Insight: We should train models on LOG(price) for better predictions, then reverse at the end

**Why log-transform the target?**
- Raw price: $7,500 to $142,000 — a 19x range
- After log: narrow, symmetric range — much easier for models to learn
- Error in log scale = proportional error in original scale (a model that's 5% off on a $10K machine gets penalized the same as a model that's 5% off on a $100K machine)

---

## ❓ CELLS 23-24 — EDA Step 2: Missing Value Analysis

```python
# Calculate missing percentage per column
missing = train_df.isnull().sum() / len(train_df) * 100
missing_sorted = missing[missing > 0].sort_values(ascending=False)
```

### 🔎 What is this doing?

Calculates what % of each column is missing (NaN) and plots a bar chart.

**Key observations:**
- Many specification columns have 50-90%+ missing values (common in auction data — not all details are recorded)
- `OperationalHoursMeter` has significant missing values (meters often broken or not recorded)
- `ManufactureYear` sometimes missing (unknown machine age)
- Strategy: Use imputation (fill in missing values) rather than dropping

---

## 📊 CELLS 25-26 — EDA Step 3: Categorical Features Bivariate Analysis

```python
# Box plots: price by category
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col in cat_cols[:6]:
    plot_boxplot(col, ...)
```

### 🔎 What is this doing?

Creates box plots showing the distribution of prices for each category.

**How to read a box plot:**
- The box shows the middle 50% of prices (25th to 75th percentile)
- The line in the middle of the box = median price
- The "whiskers" extend to 1.5× the box width
- Dots beyond whiskers = outliers

**Key insights:**
- Different machine types have very different price ranges (e.g., cranes vs. compactors)
- `RegionCode` has significant impact on price
- `FunctionalClassification` is very predictive (eta² confirmed later)

---

## 📉 CELLS 27-28 — EDA Step 4: Numerical Features Analysis

```python
# Histogram for each numerical column
for col in numeric_cols:
    plt.hist(train_df[col].dropna(), bins=30)
```

### 🔎 What is this doing?

Plots the distribution of numerical columns:
- `ManufactureYear`: Peaks in 2004-2008 (most machines sold are from mid-2000s)
- `OperationalHoursMeter`: Right-skewed (most machines have <5000 hours, some have 20000+)
- `SaleYear`: Peaks in 2009-2012 (recession and post-recession auctions)

**Why analyze these?** To decide whether we need transformations (log, clip) and to spot outliers.

---

## 🔗 CELLS 29-30 — EDA Step 5: Correlation Analysis

```python
corr_matrix = train_df[numeric_cols + ['TargetValue']].corr()
sns.heatmap(corr_matrix, annot=True)
```

### 🔎 What is this doing?

Computes and plots the Pearson correlation between numerical features and the target price.

**How to read this chart:**
- Values range from -1 to +1
- +1 = as feature increases, price increases perfectly
- -1 = as feature increases, price decreases perfectly
- 0 = no linear relationship
- Colors: dark red = strong positive, dark blue = strong negative, white = no correlation

**Key findings:**
- `ManufactureYear` is positively correlated with price (newer machines cost more)
- `OperationalHoursMeter` is negatively correlated (more hours = more wear = lower price)
- `SaleYear` is somewhat correlated (economic conditions change over time)

---

## 🗺️ CELLS 31-32 — EDA Step 6: Price by Machine Type and Region

```python
# Average price by FunctionalClassification
price_by_type = train_df.groupby('FunctionalClassification')['TargetValue'].mean()
# Average price by RegionCode  
price_by_region = train_df.groupby('RegionCode')['TargetValue'].mean()
```

### 🔎 What is this doing?

Creates bar charts showing average price per:
1. **Machine Type** (FunctionalClassification): e.g., "Crane" machines cost on average $80K, "Compactor" machines $25K
2. **Region** (RegionCode): prices vary significantly by geography

**Key insight:** Both machine type AND region are strong predictors — this is why we later add interaction features (machine type × region) in v15/v16.

---

## 🔧 CELLS 33-34 — Markdown: Strategy Overview

Outlines the preprocessing strategy:
1. Split date → Year, Month, Quarter, etc.
2. Compute Machine Age = SaleYear - ManufactureYear
3. Engineer usage rate = Hours / (Age + 1)
4. Create "is present" flags for optional columns
5. Impute missing values
6. Encode categorical features with OrdinalEncoder

---

## ⚙️ CELL 35 — Feature Engineering

```python
def engineer_features(df):
    # Date decomposition
    df['SaleYear']    = pd.to_datetime(df['TransactionDate']).dt.year
    df['SaleMonth']   = pd.to_datetime(df['TransactionDate']).dt.month
    df['SaleQuarter'] = pd.to_datetime(df['TransactionDate']).dt.quarter
    
    # Machine Age
    df['MachineAge'] = df['SaleYear'] - df['ManufactureYear']
    
    # Usage intensity
    df['UsagePerYear'] = df['OperationalHoursMeter'] / (df['MachineAge'] + 1)
    
    # Presence flags for optional columns
    for col in optional_cols:
        df[f'{col}_present'] = df[col].notna().astype(int)
    
    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)
```

### 🔎 What is this doing?

Transforms raw data into meaningful numeric signals. Adds 18 new features.

| Feature | What it tells the model |
|---|---|
| SaleYear | Economic conditions change year-to-year |
| SaleMonth | Seasonal pricing patterns |
| SaleQuarter | Q1 vs Q4 auction dynamics |
| MachineAge | Older = cheaper (generally) |
| UsagePerYear | Intensity of use = wear rate |
| `{col}_present` | Was this optional spec recorded? Often correlates with price |

**Important:** This function is applied IDENTICALLY to train and test (no leakage!).

**Output:**
```
Feature engineering applied!
  Train: (138701, 50) → (138701, 68)
  Test : (15000, 49)  → (15000, 67)
```
(+18 new features added to each)

---

## 📐 CELL 37 — Prepare Feature Matrices

```python
ID_COLS  = ['TransactionID', 'AssetID']
TARGET   = 'TargetValue'

y_all    = np.log1p(train_df[TARGET].values)    # Log-transform target
X_all    = train_df.drop(columns=ID_COLS + [TARGET, 'TransactionDate'])
X_test   = test_df.drop(columns=ID_COLS + ['TransactionDate'])

# 80/20 train/val split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
```

### 🔎 What is this doing?

1. **`y_all = log1p(price)`**: Log-transforms prices. Models train better on log-scale.
2. **`X_all`**: All 65 feature columns (minus ID and target).
3. **80/20 split**: 80% of data for training, 20% held out for validation (simulating unseen data).

**Output:**
```
X_all   : (138701, 65)
y_all   : (138701,)  (log1p scale)
X_test  : (15000, 65)

Split 80/20:
  X_train: (110960, 65)   X_val: (27741, 65)

Target (log1p) range:
  Min 8.923 → $7,500
  Max 11.864 → $142,000
```

---

## 🔬 CELL 38-40 — Sklearn Preprocessing Pipeline

```python
# Separate numeric and categorical columns
numeric_cols    = X_all.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_all.select_dtypes(include=['object']).columns.tolist()

# Numeric pipeline: impute missing with median, then scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing with 'Unknown', then ordinal encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Combine into one transformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])
```

### 🔎 What is this doing?

Creates a **preprocessing pipeline** — a sequence of transformations applied automatically before training.

**Why use a pipeline?**
- It applies the SAME transformations to both train and test data
- Prevents data leakage (imputation statistics computed only on training data, then applied to test)
- Makes the whole process reproducible

**Numeric pipeline:**
1. `SimpleImputer(strategy='median')`: Fills missing numbers with the column median. E.g., if ManufactureYear is missing and the median is 2004, use 2004.
2. `StandardScaler()`: Scales features to have mean=0 and std=1. Important for Ridge regression.

**Categorical pipeline:**
1. `SimpleImputer(fill_value='Unknown')`: Fills missing text with "Unknown".
2. `OrdinalEncoder`: Converts text to numbers. "Bulldozer"=0, "Excavator"=1, etc. The order is arbitrary but consistent.

**Output:**
```
Numeric  cols (24): ['ProductConfigID', 'ManufactureYear', ...]
Categorical cols (41): ['InventoryGroupCategory', ...]
```

---

## 🏆 CELL 41-42 — Helper Functions (RMSLE + Submission)

```python
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def make_submission(model, X_test, test_ids, filename='submission.csv'):
    raw_preds = np.expm1(model.predict(X_test_preprocessed))
    raw_preds = np.maximum(raw_preds, 0)
    ...
```

### 🔎 What is this doing?

Two helper functions:

**`rmsle`**: The competition metric. Lower = better.
- `np.maximum(y_pred, 0)`: Never predict negative prices
- `np.log1p(...)`: Measures proportional errors (fair across all price ranges)

**`make_submission`**: Saves predictions to CSV for Kaggle submission.

**Output:** `✅ Helpers ready.`

---

## 🤖 CELLS 43-47 — Model Comparison (5 Algorithms)

This is the most important section! We test 5 completely different models:

### Model 0: DummyRegressor (Baseline)
```python
dummy = DummyRegressor(strategy='mean')
```
**What:** Predicts the training mean ($41,521) for every row.
**Val RMSLE: 0.6694** — this is our floor (must beat this!)

---

### Model 1: Ridge Regression (Linear)
```python
ridge = Ridge(alpha=10.0)
```
**What:** A linear model: `price = w1 × feature1 + w2 × feature2 + ... + bias`
**Val RMSLE: 0.5296**

**Pros:** Fast, interpretable, works well when relationships are linear.
**Cons:** Cannot capture non-linear patterns (e.g., 20-year-old machines depreciate differently than 5-year-old ones).

**What is `alpha=10.0`?** L2 regularization strength. Higher = stronger penalty on large weights = simpler model.

---

### Model 2: Random Forest (Bagging)
```python
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
```
**What:** Trains 200 decision trees on different random subsets of data and features. Averages their predictions.
**Val RMSLE: 0.2895**

**Pros:** Handles non-linearity, robust to outliers, provides feature importance.
**Cons:** Slow to train, high memory usage.

**What is `max_depth=15`?** Each tree can make at most 15 sequential decisions (splits). Deeper = more complex = can overfit.

---

### Model 3: HistGradientBoosting
```python
hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05)
```
**What:** A fast gradient boosting model built into sklearn. Uses histogram binning for speed.
**Val RMSLE: 0.2327**

**Pros:** Handles NaN natively (no imputation needed!), fast.
**Cons:** Fewer tuning options than XGBoost/LightGBM.

---

### Model 4: XGBoost (Best!)
```python
xgb = XGBRegressor(
    n_estimators=500, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    enable_categorical=True, random_state=42
)
```
**What:** Extreme Gradient Boosting — the classic competition winning algorithm.
**Val RMSLE: 0.2208** ✅ Best so far!

**Key parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| n_estimators | 500 | Build 500 sequential trees |
| max_depth | 7 | Each tree can be up to 7 levels deep |
| learning_rate | 0.1 | Step size for corrections |
| subsample | 0.8 | Use 80% of rows per tree |
| colsample_bytree | 0.8 | Use 80% of features per tree |

---

### Model 5: LightGBM
```python
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, num_leaves=63, learning_rate=0.1, random_state=42
)
```
**What:** Microsoft's fast gradient boosting. Grows trees leaf-first (vs. level-first in XGBoost).
**Val RMSLE: 0.2217** — Close second!

---

## 📊 CELL 48-49 — Model Comparison Bar Chart

```python
fig, ax = plt.subplots(figsize=(12, 6))
names  = list(model_results.keys())
scores = list(model_results.values())
ax.barh(names, scores)
ax.axvline(x=0.85, color='red', label='Kaggle cutoff 0.85')
```

### 🔎 How to read this chart:

- **X-axis:** RMSLE score (lower is better!)
- **Y-axis:** Model names
- **Red vertical line:** Competition cutoff (0.85) — all models beat this!
- **Shorter bar = better model**

**Full comparison table from actual output:**

| Model | Val RMSLE | Status |
|---|---|---|
| DummyRegressor (mean) | 0.6694 | Baseline |
| Ridge Regression | 0.5296 | Beats dummy |
| Random Forest | 0.2895 | Good! |
| HistGradientBoosting | 0.2327 | Better! |
| **XGBoost** | **0.2208** | **Best!** ✅ |
| LightGBM | 0.2217 | Close second |

**Key insight:** Tree-based gradient boosting models (XGBoost, LightGBM) massively outperform linear models on this dataset. This makes sense — price relationships are highly non-linear (machine age doesn't linearly reduce price; there are complex interactions between machine type, region, age, etc.).

---

## 🎛️ CELLS 50-52 — Hyperparameter Tuning (XGBoost)

```python
param_grid = {
    'xgb__n_estimators':       [300, 500, 800, 1000],
    'xgb__max_depth':          [6, 7, 8, 9, 10],
    'xgb__learning_rate':      [0.05, 0.08, 0.1, 0.15],
    'xgb__subsample':          [0.6, 0.7, 0.8],
    'xgb__colsample_bytree':   [0.6, 0.7, 0.8],
    'xgb__min_child_weight':   [3, 5, 7],
    'xgb__gamma':              [0, 0.1, 0.3],
    'xgb__reg_alpha':          [0.1, 0.3, 1.0],
    'xgb__reg_lambda':         [0.5, 1.0, 2.0],
}

search = RandomizedSearchCV(
    pipeline, param_grid,
    n_iter=25,    # Test 25 random combinations
    cv=3,         # 3-fold cross-validation
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
)
search.fit(X_train, y_train)
```

### 🔎 What is this doing?

**Story:** Imagine trying many different "settings" for your camera to get the best photo. Instead of trying all possible combinations (which would take days), RandomizedSearchCV randomly picks 25 combinations and evaluates each using 3-fold CV.

**Each hyperparameter explained:**

| Parameter | Description | Tuned Range |
|---|---|---|
| n_estimators | Number of boosting trees | 300–1000 |
| max_depth | Max depth per tree | 6–10 |
| learning_rate | Size of corrections per tree | 0.05–0.15 |
| subsample | Row sampling fraction per tree | 0.6–0.8 |
| colsample_bytree | Feature sampling fraction per tree | 0.6–0.8 |
| min_child_weight | Min sum of weights in leaf node | 3–7 |
| gamma | Min loss reduction for split | 0–0.3 |
| reg_alpha | L1 regularization | 0.1–1.0 |
| reg_lambda | L2 regularization | 0.5–2.0 |

**Best Parameters Found (actual output):**
```
Best Parameters:
  colsample_bytree    : 0.8
  gamma               : 0
  learning_rate       : 0.08
  max_depth           : 8
  min_child_weight    : 5
  n_estimators        : 1000
  reg_alpha           : 0.3
  reg_lambda          : 1.0
  subsample           : 0.7
```

**Key insight:** Deeper trees (depth=8 > original 7) with more iterations (1000 > 500) and slightly slower learning rate (0.08 < 0.1) tends to be better for complex datasets like this.

---

## 🏋️ CELLS 53-54 — Train Final Model on All Data

```python
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(**best_params, random_state=42))
])
# Train on 100% of training data (not just 80%)
final_model.fit(X_all, y_all)
```

### 🔎 What is this doing?

Now that we've found the best hyperparameters using 80% of data, we **retrain on 100% of training data**. More data = better model!

**Why not always train on 100%?**
- We needed the 20% validation set to evaluate models and tune hyperparameters
- Once tuning is done, we use ALL data for the final model

**Output:**
```
Training final model on 100% of training data ...

Final model trained on 138,701 samples, 65 features

=======================================================
  PERFORMANCE SUMMARY
=======================================================
  4. XGBoost (tuned)              0.2208  ← Val RMSLE
  (others for comparison...)
```

---

## 🔍 CELLS 55-57 — Error & Residual Analysis

```python
# Calculate residuals
y_val_pred_raw = np.expm1(final_model.predict(X_val_preprocessed))
residuals = y_val - y_val_pred_raw
percent_errors = (y_val_pred_raw - y_val) / y_val * 100
```

### 🔎 What is this doing?

Creates 3 diagnostic plots to understand WHERE the model makes mistakes:

**Chart 1: Actual vs. Predicted Scatter**
- X-axis: actual price, Y-axis: predicted price
- Perfect model: all points on the diagonal
- Points above diagonal: model underpredicts (predicts lower than actual)
- Points below diagonal: model overpredicts

**Chart 2: Residuals Distribution**
- X-axis: prediction error (actual - predicted), Y-axis: frequency
- Centered bell curve at 0 = unbiased predictions
- Skewed distribution = systematic errors

**Chart 3: Percentage Error Distribution**
- Shows % error distribution (clipped at ±40%)
- Tight bell at 0% = consistently accurate

**Key observations:**
- Most errors are within ±20%
- Some outliers exist (very expensive or unusual machines)

---

## ⭐ CELLS 58-59 — Feature Importance Chart

```python
importance = final_model['xgb'].feature_importances_
feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False).head(20)
```

### 🔎 What is this doing?

XGBoost's `feature_importances_` tells us which features it relied on most for splitting decisions.

**How to read this chart:**
- X-axis: Importance score (higher = more important)
- Y-axis: Feature name
- Longer bars = model relied on this feature more

**Typical top features for this model:**

| Feature | Importance | Why |
|---|---|---|
| MachineAge | High | Age is the strongest depreciation driver |
| ManufactureYear | High | Directly related to age |
| SaleYear | High | Economic conditions vary by year |
| OperationalHoursMeter | High | Usage = wear = lower price |
| ProductConfigID (encoded) | Medium | Machine type strongly predicts price |

**Key insight:** Time-related features (age, manufacture year, sale year) dominate. This is typical for depreciation-based pricing problems.

---

## 📤 CELLS 60-62 — Final Prediction & Submission

```python
# Predict on test set
raw_preds = final_model.predict(X_test_preprocessed)
final_prices = np.expm1(raw_preds)   # Reverse log transform
final_prices = np.maximum(final_prices, 0)  # No negative prices

# Save to CSV
sub = pd.DataFrame({
    'TransactionID': test_df['TransactionID'],
    'TargetValue': final_prices
})
sub.to_csv('submission.csv', index=False)
```

### 🔎 What is this doing?

1. **Predict:** Run test data through the final XGBoost model
2. **Reverse transform:** `np.expm1()` converts log-scale predictions back to dollar values
3. **Clip negatives:** Ensure no prediction is negative
4. **Save:** Create the submission CSV file

**Output:**
```
✅ submission.csv saved (15,000 rows)

Submission statistics:
  Rows   : 15,000
  Min    : $    7,251.37
  Max    : $  141,053.44
  Mean   : $   40,796.54
  Median : $   34,579.79
  Negative predictions: 0  ✅ None

=======================================================
  SUBMISSION READY!
  File : submission.csv
  Val RMSLE : 0.2045  (target <= 0.85)
=======================================================
```

**Kaggle Score:** ~0.20845 — this became the starting point for all future improvements!

---

## 📊 CELL 63-64 — Prediction Distribution Sanity Check

```python
fig, axes = plt.subplots(1, 2)
axes[0].hist(train_df['TargetValue'], bins=50, label='Train actual prices')
axes[1].hist(final_prices, bins=50, label='Test predictions')
```

### 🔎 What is this doing?

**Story:** A sanity check — our model's predicted prices should look similar to the training data price distribution. If not, something is wrong!

**How to read this comparison:**
- Left chart: actual training prices (this is "ground truth" distribution)
- Right chart: our predictions for test data
- They should look similar in shape and range

**Good sign:** Similar bell-curve shape, similar range.
**Red flag:** If predictions are mostly at $7K or $142K, the model has a problem.

---

## 🏆 Score Journey (Starting from v1)

| Version | Key Strategy | Kaggle Score |
|---|---|---|
| **v1 (this notebook)** | **5 models compared, XGBoost tuned** | **~0.20845** |
| v6 | Increased trees (6000) + 255 leaves + lr=0.015 | 0.19086 |
| v10 | Leak-Free Target Encoding + Asset History | 0.19020 |
| v14 | Fold-Safe Frequency + Group Stats | 0.18964 |
| v15 | Expanded TE (7 cols) + Interaction TE + Dual LGB | 0.18847 |
| v16 | Cyclical Dates + 3 Interaction TEs + Non-Linear Age | 0.18763 |

---

## 📚 Glossary

| Term | Simple Explanation |
|---|---|
| **RMSLE** | Root Mean Squared Log Error — competition metric (lower = better) |
| **Baseline** | The simplest possible prediction (e.g., always predict the mean) |
| **Feature Engineering** | Creating new useful columns from existing ones (e.g., MachineAge from Year and ManufactureYear) |
| **Train/Val Split** | Dividing data into 80% for training and 20% for evaluating performance |
| **Imputation** | Filling in missing (NaN) values with a statistic like median or "Unknown" |
| **OrdinalEncoder** | Converting text categories (e.g., "Bulldozer") to integers (e.g., 0, 1, 2) |
| **StandardScaler** | Normalizes features to mean=0, std=1 — important for linear models |
| **Pipeline** | A chain of preprocessing + model steps that applies transformations automatically |
| **Gradient Boosting** | ML technique building trees sequentially, each fixing previous errors |
| **RandomizedSearchCV** | Tests random hyperparameter combinations to find the best settings |
| **Hyperparameter** | A model setting you must choose manually (e.g., max_depth, learning_rate) |
| **Overfitting** | When a model memorizes training data but fails on new data |
| **Feature Importance** | Measure of how much each feature contributed to model predictions |
| **Residual** | Prediction error: actual value − predicted value |
| **Log Transform** | Applying log(1+x) to compress large ranges and normalize distributions |
| **Ensemble** | Combining multiple models for better accuracy than any single model |
| **n_estimators** | Number of trees in a gradient boosting model |
| **max_depth** | Maximum depth (levels of splits) per tree |
| **learning_rate** | How much each new tree corrects previous errors (smaller = more careful) |
