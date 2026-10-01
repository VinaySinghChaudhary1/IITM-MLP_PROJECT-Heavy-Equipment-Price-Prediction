# 📖 v6 Code Explanation — Easy Story Guide
## `22f2001153_Improved_v6.ipynb`
### Heavy Equipment Price Prediction | Kaggle Score: **0.19086** ✅

---

> **How to read this guide:** Imagine you are trying to predict the price of a used bulldozer or excavator at an auction. You have data about thousands of past auctions. Your goal is to build a "smart calculator" that uses this historical data to predict prices for new auctions. Every section below explains one part of the notebook, step by step, in simple language.

---

## ⚠️ IMPORTANT: How to submit on Kaggle

1. Go to Kaggle → Competition → "Code" → **"+ New Notebook"**
2. Click **"File" → "Import Notebook"** → Upload `22f2001153_Improved_v6.ipynb`
3. Click **"Add Data"** → Add competition dataset
4. Click **"Run All"** (takes ~30–40 minutes)
5. Submit the generated `submission.csv`

> ⚠️ Always upload as a **fresh new notebook**. Never copy-paste into an old notebook.

---

## 📊 Competition Overview

| Item | Value |
|---|---|
| Train rows | 138,701 |
| Test rows | 15,000 |
| Target | Price of heavy equipment at auction |
| Metric | RMSLE — **lower is better** |
| Previous best score | 0.19969 (v4) |
| v6 score | **0.19086** ✅ (best at this point) |
| Classmate benchmark | 0.18356 |
| Competition cutoff | 0.15000 |

---

## 🔧 CELL 1 — Default Kaggle Cell

This is the standard Kaggle template cell. It lists all files in the competition dataset folder.
**You don't need to change this.** It just confirms your dataset is attached correctly.

**Files available:**
- `train.csv` — historical auction data with prices
- `test.csv` — auction data without prices (we predict these)
- `sample_submission.csv` — the required submission format
- `metadata.csv` — extra descriptions of the columns

---

## 📦 CELL 2 — Load All Libraries

```python
import subprocess, sys
try:
    import catboost
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "catboost", "-q"], check=True)
...
SEED = 42
```

**🔎 What is this doing?**
Think of this cell as opening your toolbox before starting a job.

- `catboost`: CatBoost is one of our three machine learning models. Kaggle might not have it pre-installed, so we try to install it if it's missing.
- `numpy (np)`: Does fast math on numbers.
- `pandas (pd)`: Works with tables of data (like Excel, but in Python).
- `matplotlib.pyplot`: Draws graphs and charts.
- `seaborn (sns)`: Makes prettier statistical charts.
- `sklearn.model_selection.KFold`: Splits data into training and validation folds.
- `lightgbm (lgb)`: LightGBM — our strongest model.
- `XGBRegressor`: XGBoost — a reliable second model.
- `CatBoostRegressor`: CatBoost — handles text/category features natively.
- `SEED = 42`: A magic number to make all random operations **reproducible** (same results every run).

**✅ Expected output:** `Libraries loaded.`

---

## 🛠️ CELL 3 — Helper Functions

```python
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def make_submission(preds, ids, filename='submission.csv'):
    ...
```

**🔎 What is this doing?**

We define two helper functions:

### `rmsle(y_true, y_pred)`
This is the **competition metric**. Lower RMSLE = better predictions.
- `y_true`: The actual known prices.
- `y_pred`: Our model's predicted prices.
- `np.maximum(y_pred, 0)`: Makes sure we never predict a negative price.
- `np.log1p(...)`: Applies log(1+x) transformation. This penalizes percentage errors equally regardless of price level — a $1,000 error on a $10,000 machine is treated the same as a $10,000 error on a $100,000 machine.

### `make_submission(preds, ids, filename)`
Saves our predictions as a CSV file for Kaggle submission.

**✅ No output** (functions are just defined, not run yet)

---

## 📂 CELL 4 — Load the Data

```python
train_df = pd.read_csv('/kaggle/input/.../train.csv')
test_df  = pd.read_csv('/kaggle/input/.../test.csv')
sample   = pd.read_csv('/kaggle/input/.../sample_submission.csv')
metadata = pd.read_csv('/kaggle/input/.../metadata.csv')
```

**🔎 What is this doing?**
This loads the competition data files into memory.

- `train_df`: 138,701 rows WITH actual prices — used to train the model.
- `test_df`: 15,000 rows WITHOUT prices — we predict these.
- `sample`: The required format for our submission file.
- `metadata`: Column descriptions.

**📊 Key Observations:**
| Dataset | Rows | Columns |
|---------|------|---------|
| Train | 138,701 | 50 |
| Test | 15,000 | 49 |

- **Price range:** $7,500 to $142,000
- **Average price:** $41,522
- **Median price:** $35,000

**✅ Expected output:**
```
TRAIN :  138,701 x 50
TEST  :   15,000 x 49
Target  min=$7,500  median=$35,000  mean=$41,522  max=$142,000
```

---

## 🔨 CELL 5 — Feature Engineering

```python
def engineer_features(df):
    # Date features
    df['SaleYear'] = df['TransactionDate'].dt.year
    df['SaleMonth'] = df['TransactionDate'].dt.month
    ...
    # Machine Age
    df['MachineAge'] = df['SaleYear'] - df['ManufactureYear']
    ...
```

**🔎 What is this doing?**

This is the most important data preparation step — transforming raw columns into useful numeric signals.

### Date Decomposition
A single `TransactionDate` (e.g., `2012-11-16`) is split into many useful signals:
- `SaleYear`, `SaleMonth`, `SaleDay`: Year, month, day of sale.
- `SaleDayOfWeek`: Was it sold on Monday (0) or Friday (4)?
- `SaleDayOfYear`: What day number of the year (1–365)?
- `SaleQuarter`, `SaleWeek`: Quarter and week number.
- `IsMonthEnd`, `IsYearEnd`: Flags for auction timing patterns.

### Data Cleaning
- `ManufactureYear ≤ 1920` → Set to `NaN` (data entry errors — no 1910 bulldozers!)
- `OperationalHoursMeter == 0` → Set to `NaN` (broken or missing meter)

### Machine Age
- `MachineAge = SaleYear - ManufactureYear`: Older machines are worth less.

### Usage Rate Features
- `LogHours = log1p(OperationalHoursMeter)`: Log of total usage hours
- `UsagePerYear = Hours / (Age + 1)`: How hard was the machine used per year?
- `HoursPresent`: Was the hours meter available? (1=yes, 0=no)

### Ordinal Encoding (text → numbers with order)
- `UtilizationTier`: Low=0, Medium=1, High=2
- `AssetScaleFactor`: Mini=0, Compact=1, Small=2, Medium=3, Large=4+

### Text Columns
- `'None or Unspecified'` → `NaN` (blank value, not real data)
- Convert remaining text to `category` dtype

**📊 Key Observations (from output):**
- ManufactureYear is missing for **14,732 rows (10.6%)** — significant!
- OperationalHours is missing for **88,226 rows (63.6%)** — majority missing!

**✅ Expected output:**
```
Applying feature engineering...
Train: (138701, 65)  |  Test: (15000, 64)
ManufactureYear NaN : 14,732 (10.6%)
OperationalHours NaN: 88,226 (63.6%)
```

---

## 📐 CELL 6 — Prepare Feature Matrices

```python
y_all        = np.log1p(train_df['TargetValue'].values)
X_all        = train_eng.drop(columns=ID_COLS + ['TargetValue'], errors='ignore')
X_test_final = test_eng.drop(columns=ID_COLS, errors='ignore')
cat_features = X_all.select_dtypes(include=['category', 'object']).columns.tolist()
```

**🔎 What is this doing?**

We separate the data into **inputs (X)** and **target (y)**:

- `y_all = log1p(TargetValue)`: We train on log-prices instead of raw prices. Reverse with `expm1()` at the end.
- `X_all`: 62 features for training (65 total − `TransactionID`, `AssetID`, `TargetValue`)
- `X_test_final`: 62 features for test predictions
- `cat_features`: The 40 text/category columns (needed for CatBoost)

**✅ Expected output:**
```
Feature matrix : (138701, 62)
Test matrix    : (15000, 62)
Cat features   : 40
Num features   : 22
```

---

## 🔄 CELL 7 — 5-Fold Cross Validation Loop (Main Training)

**🔎 What is this doing?**

This is the **heart of the notebook** — training three gradient boosting models with 5-fold cross validation.

### What is 5-Fold Cross Validation?
We divide the training data into 5 equal parts (folds):
1. Train on 4 parts (80%), validate on 1 part (20%)
2. Rotate which part is used for validation
3. Repeat 5 times → get predictions for EVERY row

This gives us reliable **OOF (Out-Of-Fold)** predictions.

### Three Models Trained:

#### 🟢 LightGBM (LGB) — Best Performer
```python
lgb.LGBMRegressor(
    n_estimators=6000,    # Up to 6000 trees
    num_leaves=255,        # Very expressive leaf-first trees
    learning_rate=0.015,   # Slow, careful learning
    early_stopping=200,    # Stop if no improvement in 200 rounds
)
```
- `num_leaves=255`: This is the KEY upgrade from v4! More leaves = more expressive model = better accuracy.
- `learning_rate=0.015`: Another upgrade from v4. Slower rate + more trees = better results.

#### 🔵 XGBoost (XGB)
```python
XGBRegressor(
    n_estimators=6000,
    max_depth=9,
    learning_rate=0.015,
    early_stopping_rounds=200,
)
```

#### 🟡 CatBoost (CAT)
```python
CatBoostRegressor(
    iterations=6000,
    learning_rate=0.015,
    cat_features=cat_features,   # Handles all 40 category columns natively
)
```

**📊 Actual Fold Results (v6):**
| Fold | LGB | XGB | CAT |
|------|-----|-----|-----|
| Fold 1 | 0.19663 (iter=5901) | 0.20266 (iter=5999) | 0.20733 |
| Fold 2 | 0.20001 (iter=5998) | 0.20558 (iter=5993) | 0.20925 |
| Fold 3 | 0.19998 (iter=4964) | 0.20561 (iter=5993) | 0.21017 |
| Fold 4 | 0.19961 (iter=5997) | 0.20585 (iter=5994) | 0.21099 |
| Fold 5 | 0.19914 (iter=5325) | 0.20415 (iter=5998) | 0.20925 |

| Model | OOF RMSLE | Notes |
|-------|-----------|-------|
| **LightGBM** | **0.19908** | Best single model |
| XGBoost | 0.20477 | Solid second |
| CatBoost | 0.20940 | Most consistent |

---

## ⚖️ CELL 8 — Find Optimal Blend Weights

```python
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 6)
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        score = rmsle(...)
```

**🔎 What is this doing?**

**Story:** Ask three expert appraisers. Expert 1 (LGB) is very accurate. Expert 2 (XGB) is decent. Expert 3 (CAT) is a different perspective. We try all possible "trust percentages" and pick the best.

We test all combinations (in 5% steps) and find what minimizes blended OOF RMSLE.

**📊 Optimal Blend Found:**
| Weight | Model | Value |
|--------|-------|-------|
| w1 | LightGBM | **65%** |
| w2 | XGBoost | **5%** |
| w3 | CatBoost | **30%** |

- **Blended OOF RMSLE: 0.19627** (vs previous best 0.19969 — BIG improvement!)

**✅ Expected output:**
```
Best weights  : LGB=0.65 XGB=0.05 CAT=0.30
Blended RMSLE : 0.19627
Previous best : 0.19969  ->  IMPROVED!
```

---

## 📤 CELL 9 — Build Final Submission

```python
test_blend_log = w1 * test_lgb_avg + w2 * test_xgb_avg + w3 * test_cat_avg
final_prices   = np.expm1(test_blend_log)
final_prices   = np.clip(final_prices, price_min, price_max)
```

**🔎 What is this doing?**

1. **Average across folds:** For each test row, average predictions from all 5 folds (more stable than single run).
2. **Blend models:** Apply optimal weights (LGB=65%, XGB=5%, CAT=30%).
3. **`np.expm1()`:** Reverse the log transformation → actual dollar prices.
4. **Clip prices:** Never below $7,500 or above $137,500 (stays in observed range).

**✅ Expected output:**
```
Clipped to [$7,500, $137,500]
Saved: submission.csv  |  Rows=15,000  Min=$7,500  Median=$34,666  Max=$137,500
```

**Kaggle Score: 0.19086** ✅ (improved from v4's 0.19969!)

---

## 📈 CELL 10 — Diagnostics and Analysis

**🔎 What is this doing?**
Creates 3 diagnostic charts and prints key performance metrics.

### Three Plots:
1. **Actual vs Predicted** — Scatter plot showing accuracy
2. **Residuals Distribution** — Distribution of prediction errors
3. **% Error Distribution** — Percentage error spread

### Performance Metrics:
| Metric | v6 Value | Interpretation |
|--------|----------|----------------|
| Blended OOF RMSLE | **0.19627** | Main competition metric |
| R² | 0.9122 | 91.2% of price variance explained |
| MAE | $5,152 | Average absolute error per prediction |
| Median % Error | 10.0% | Half of predictions within 10% of true price |

### Top 15 Features by LightGBM Importance:
| Feature | Importance | What it means |
|---------|-----------|---------------|
| `SaleDayOfYear` | 169,183 | Day of year strongly affects price |
| `SaleDay` | 136,119 | Day of month |
| `MachineAge` | 129,772 | How old the machine is |
| `ManufactureYear` | 126,656 | Year the machine was made |
| `SaleYear` | 93,346 | Which year it sold |
| `RegionCode` | 80,183 | Geographic location |
| `OperationalHoursMeter` | 71,555 | Total hours of use |
| `ProductConfigID` | 63,782 | Machine configuration |
| `UsagePerYear` | 60,517 | Average annual usage intensity |
| `SaleDayOfWeek` | 51,903 | Day of the week |
| `Spec_BaseClass` | 44,909 | Machine base specification |

**🔑 Insight:** Date-related features are the MOST important! `SaleDayOfYear` is #1. This suggests strong seasonal pricing patterns in the auction market.

---

## 📋 Cells 11–59 — Reference EDA Cells (Commented Out)

These are analysis cells from an earlier version of the notebook. They contain:
- **Section 2:** Detailed data loading and shape inspection
- **Section 3:** Exploratory Data Analysis (EDA) — target distribution, missing values, categorical analysis, correlations
- **Section 4:** Earlier preprocessing pipeline code
- **Section 5:** Model comparison across multiple algorithm families
- **Section 6:** Hyperparameter tuning with RandomizedSearchCV
- **Section 7:** Residual analysis
- **Section 8:** Earlier submission generation
- **Milestone 1 Answers:** Q1–Q14 answers for the course assignment

These cells are all commented out (`# ...`) so they do **NOT run** when you click "Run All". They are kept for reference only.

---

## 🏆 Score Journey Summary

| Notebook | Key Change | OOF RMSLE | Kaggle Score |
|----------|-----------|-----------|--------------|
| v4 | Basic 3-model ensemble | ~0.198 | 0.19969 |
| **v6** | `num_leaves=255` + `lr=0.015` + 6000 trees | **0.19627** | **0.19086** |
| v8 | Target encoding (with bug) | worse | 0.19996 |
| v10 | Leak-free TE + Asset History | 0.19604 | 0.19020 |

### What made v6 better than v4?
1. **`num_leaves=255`** (was ~63 in v4) — much more expressive trees
2. **`learning_rate=0.015`** (was ~0.05 in v4) — slower, more accurate learning
3. **6000 trees** (was ~1000 in v4) — more boosting rounds
4. **Optimal blending** — grid search for best LGB/XGB/CAT weights

---

## 📚 Glossary

| Term | Simple Explanation |
|---|---|
| **RMSLE** | Root Mean Squared Log Error — competition metric (lower = better) |
| **OOF Predictions** | Out-of-fold predictions — model predicts on data it never trained on |
| **Early Stopping** | Stop adding trees when model stops improving on validation data |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Gradient Boosting** | Builds trees sequentially, each one fixing the previous one's errors |
| **num_leaves=255** | Each tree can have up to 255 branches (very expressive) |
| **Learning Rate** | How much each new tree corrects errors (smaller = more careful) |
| **Ensemble/Blending** | Combining predictions from multiple models for better accuracy |
| **Target Encoding** | Replacing a category with its average target price |
| **Feature Engineering** | Transforming raw data into useful numeric signals |
