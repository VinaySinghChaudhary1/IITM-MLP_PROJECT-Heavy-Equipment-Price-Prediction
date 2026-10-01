# 📖 v10 Code Explanation — Easy Story Guide
## `22f2001153_Improved_v10.ipynb`
### Heavy Equipment Price Prediction | Score Journey: 0.19969 → 0.19086 (v6) → **0.19020 (v10)**

---

> **How to read this guide:** Imagine you are trying to predict the price of a used bulldozer or excavator at an auction. You have data about thousands of past auctions. Your goal is to build a "smart calculator" that uses this historical data to predict prices for new auctions. Every section below explains one part of the notebook, step by step, in simple language.

---

## ⚠️ IMPORTANT: How to submit on Kaggle

1. Go to Kaggle → Competition → "Code" → **"+ New Notebook"**
2. Click **"File" → "Import Notebook"** → Upload `22f2001153_Improved_v10.ipynb`
3. Click **"Add Data"** → Add competition dataset
4. Click **"Run All"** (takes ~35–50 minutes)
5. Submit the generated `submission.csv`

> ⚠️ Always upload as a **fresh new notebook**. Never copy-paste code into an old notebook.

---

## 📊 Competition Overview

| Item | Value |
|---|---|
| Train rows | 138,701 |
| Test rows | 15,000 |
| Target | Price of heavy equipment at auction |
| Metric | RMSLE (Root Mean Squared Log Error) — **lower is better** |
| Our best score (v6) | 0.19086 |
| v10 score | **0.19020** ✅ |
| Classmate benchmark | 0.18356 |
| Competition cutoff | 0.15000 |

---

## 🔧 CELL 1 — Default Kaggle Cell (markdown only)
This is the standard Kaggle template cell. It lists all files in the competition dataset folder.
**You don't need to change this.** It just confirms your dataset is attached correctly.

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
np.random.seed(SEED)
print("ALL LIBRARIES LOADED OK.")
```

**🔎 What is this doing?**
Think of this cell as opening your toolbox before starting a job.

- `catboost`: CatBoost is one of our three machine learning models. Kaggle might not have it pre-installed, so we try to install it if it's missing.
- `numpy (np)`: Does fast math on numbers.
- `pandas (pd)`: Works with tables of data (like Excel, but in Python).
- `matplotlib.pyplot`: Draws graphs and charts.
- `sklearn.model_selection.KFold`: Splits data into training and validation folds.
- `lightgbm (lgb)`: LightGBM — our strongest model.
- `xgboost XGBRegressor`: XGBoost — a reliable second model.
- `catboost CatBoostRegressor`: CatBoost — a model that handles categories natively.
- `SEED = 42`: A magic number to make all random operations **reproducible** (you get the same results every time).

**✅ Expected output:** `ALL LIBRARIES LOADED OK.`

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
This is the **competition metric**. Every competition has a scoring metric — this one uses RMSLE (Root Mean Squared Log Error).
- `y_true`: The actual known prices.
- `y_pred`: Our model's predicted prices.
- `np.maximum(y_pred, 0)`: Makes sure we never predict a negative price (that would be nonsense!).
- `np.log1p(...)`: Applies the log(1+x) transformation. This reduces the impact of very high-priced equipment on the error score. It means a $1,000 error on a $10,000 machine is penalized equally to a $10,000 error on a $100,000 machine.
- **Lower RMSLE = Better predictions.**

### `make_submission(preds, ids, filename)`
Saves our predictions as a CSV file that Kaggle can read and score.

**✅ Expected output:** `Helper functions ready.`

---

## 📂 CELL 4 — Load the Data

```python
train_df = pd.read_csv('/kaggle/input/.../train.csv', low_memory=False)
test_df  = pd.read_csv('/kaggle/input/.../test.csv')
sample   = pd.read_csv('/kaggle/input/.../sample_submission.csv')
```

**🔎 What is this doing?**
This loads the competition data files into memory.

- `train_df`: Contains 138,701 rows of historical auction sales WITH the actual price (called `TargetValue`). We use this to **train** our model.
- `test_df`: Contains 15,000 rows of auctions WITHOUT the price. We need to **predict** these prices.
- `sample`: Shows the expected format for our submission file.

**📊 Key Observations:**
- **Train:** 138,701 rows × 50 columns
- **Test:** 15,000 rows × 49 columns (one less because there is no `TargetValue` in the test set)
- **Price range:** $7,500 to $142,000 (all prices are log-transformed during training)

**✅ Expected output:**
```
TRAIN :  138,701 rows x 50 cols
TEST  :   15,000 rows x 49 cols
Price range: $7,500 - $142,000
```

---

## ⏳ CELL 5 — Leak-Free Asset History Features

```python
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
full_df = full_df.sort_values('TransactionID').reset_index(drop=True)
...
full_df['Asset_Prev_Sales_Count'] = grouped.cumcount().astype(np.int32)
full_df['Asset_Prev_Mean_Price'] = ...
full_df['Asset_Prev_Last_Price'] = grouped['log_target'].shift(1)
full_df['Asset_Days_Since_Last_Sale'] = grouped['TransactionDate_dt'].diff().dt.days
```

**🔎 What is this doing?**

**Story:** A tractor might be sold 3 or 4 times over its lifetime. Each time it is sold, the price depends partly on its previous selling price. A machine that sold for $80,000 last year is likely worth more than one that sold for $20,000 last year.

We combine train and test data into one big table and sort it by `TransactionID` (which represents time — lower ID = earlier sale). Then, for each machine (`AssetID`), we look at what happened in its previous sales to create 4 new features:

| Feature | What it means |
|---|---|
| `Asset_Prev_Sales_Count` | How many times was this machine sold before this sale? |
| `Asset_Prev_Mean_Price` | What was its average price across all previous sales? |
| `Asset_Prev_Last_Price` | What was the price in the most recent previous sale? |
| `Asset_Days_Since_Last_Sale` | How many days passed since the last sale? |

**🔐 Why is this "leak-free"?**
"Leakage" is when information from the future accidentally sneaks into your features, making your model appear to perform better than it really is. This is a critical mistake!

We prevent leakage by:
1. Sorting by `TransactionID` (= time order).
2. Using `cumcount()` and `shift(1)` to look ONLY at past sales, never the current one.
3. For test rows with `TargetValue = NaN`, the log target calculation treats them as 0 in cumulative sums.

**✅ Expected output:** `Asset history features added. Train: (138701, 54) | Test: (15000, 54)`
(Both now have 54 columns: 50 original + 4 new asset history features)

---

## 🔨 CELL 6 — Base Feature Engineering

```python
def preprocess_df(df):
    # Date features
    df['SaleYear'] = df['TransactionDate'].dt.year
    df['SaleMonth'] = df['TransactionDate'].dt.month
    ...
    # Machine Age
    df['MachineAge'] = df['SaleYear'] - df['ManufactureYear']
    # Ordinal Encoding
    tier_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['UtilizationTierOrd'] = df['UtilizationTier'].map(tier_map)
    ...
```

**🔎 What is this doing?**

This is the **feature engineering** step — transforming raw data into meaningful numeric signals that our models can understand.

### Date Features
A single `TransactionDate` (like `2012-11-16`) is split into many useful numbers:
- `SaleYear`, `SaleMonth`, `SaleDay`: The year, month, and day of the sale.
- `SaleDayOfWeek`: Was it sold on a Monday (0) or Friday (4)?
- `SaleDayOfYear`: What day of the year was it (1-365)?
- `SaleQuarter`: What quarter of the year (1=Jan-Mar, 4=Oct-Dec)?
- `IsMonthEnd`, `IsYearEnd`, etc.: Flag for special dates (month/year end auctions often have different pricing).

### Data Cleaning
- `ManufactureYear <= 1920` → Set to `NaN` (these are clearly data entry errors — no 1910 bulldozers!)
- `OperationalHoursMeter == 0` → Set to `NaN` (0 hours means the meter was broken or not recorded)

### Machine Age
- `MachineAge = SaleYear - ManufactureYear`: A 2010 machine sold in 2020 has an age of 10 years. Older machines are generally worth less.

### Ordinal Encoding
Some text columns have a clear order (hierarchy):
- `UtilizationTier`: Low=0, Medium=1, High=2. ("High" is better than "Low" — it's not just a label)
- `AssetScaleFactor`: Mini=0, Compact=1, Small=2, Medium=3, Large/Medium=4, Large=5. (Bigger = higher number)

Converting these to numbers helps tree models split them correctly.

### Usage Rate Features
- `LogHours = log1p(OperationalHoursMeter)`: Log of the usage hours (reduces the effect of extreme values)
- `UsagePerYear = OperationalHoursMeter / (MachineAge + 1)`: How hard was this machine used per year?
- `HoursPresent`: Was the hours meter available? (1 = yes, 0 = no)

### Category Group Relative Features
- `CategoryMeanHours`: Average hours meter for this machine category
- `HoursRelative = Hours / CategoryMeanHours`: Is this machine used more or less than average for its category?
- `CategoryMeanAge`, `AgeRelative`: Same idea for age

### Clean Text Columns
- `'None or Unspecified'` → `NaN` (these are blank/unknown values, not a real category)
- Convert all remaining text columns to `category` dtype (LightGBM handles these natively)

**✅ Expected output:** `Train preprocessed: (138701, 75) | Test preprocessed: (15000, 75)`
(Both now have 75 columns)

---

## 📐 CELL 7 — Prepare Feature Matrices

```python
ID_COLS = ['TransactionID', 'AssetID']
y_all        = np.log1p(train_proc['TargetValue'].values)
X_all        = train_proc.drop(columns=ID_COLS + ['TargetValue'], errors='ignore')
X_test_final = test_proc.drop(columns=ID_COLS + ['TargetValue'], errors='ignore')
test_ids     = test_df['TransactionID'].values
cat_cols = X_all.select_dtypes(include=['category', 'object']).columns.tolist()
```

**🔎 What is this doing?**

We separate the data into the **inputs** (X) and the **target** (y):

- `y_all = np.log1p(TargetValue)`: We use `log(1 + price)` instead of the raw price. This is because prices have a wide range ($7,500 to $142,000). Log transformation makes the distribution more symmetric and improves model performance.
- `X_all`: Train features — 72 columns (75 total minus `TransactionID`, `AssetID`, `TargetValue`)
- `X_test_final`: Test features — 72 columns (same as X_all)
- `cat_cols`: List of the 40 categorical features (text/category columns)
- `test_ids`: The `TransactionID` values from the test set (needed for the submission file)

**✅ Expected output:**
```
Feature matrix : (138701, 72)
Test matrix    : (15000, 72)
Cat features   : 40
Num features   : 32
```

---

## ⭐ CELL 8 — The K-Fold Target Encoding Function

```python
def add_oof_target_encoding(X_tr, y_tr, X_vl, X_te, cols, n_splits=5, alpha=15.0, seed=42):
    ...
    for col in cols:
        # For Training: compute TE using inner 5-fold (never use its own target)
        # For Validation/Test: compute TE using full training fold
```

**🔎 What is this doing?**

**Story:** Imagine you know that a "Bulldozer D8" machine is on average worth $75,000. If you tell the model "this machine is a Bulldozer D8 and the average price of Bulldozer D8 is $75,000," the model has useful information!

This is called **Target Encoding** (TE) — we replace each category with the average target price for that category.

But there is a danger: **Target Leakage!**

If we calculate the average using the current row's own price and then use that average as a feature for the same row, the model can "cheat." During training it will see the almost-exact target as a feature, and during testing it will only see the noisy average. This is leakage!

### How we fix it (K-Fold Target Encoding):

We split the training data into 5 internal "inner" folds. For each row:
- Its Target Encoding is computed using the **other 4 folds** — NOT its own fold.
- This way, no row ever sees its own target in its feature calculation.

| Dataset | How TE is computed |
|---|---|
| Training rows | From the other 4 inner folds (no self-contamination) |
| Validation rows | From the full training fold |
| Test rows | From the full training fold |

The `alpha=15.0` is a **smoothing factor**. For rare categories with very few examples, it blends the category average toward the global average to prevent overfitting on rare categories.

**We apply TE to 4 high-cardinality columns:** `Spec_FullDescriptor` (3,505 unique values), `Spec_BaseClass` (1,249 unique values), `ProductConfigID` (37,209 unique values), `FunctionalClassification` (65 unique values).

**✅ Expected output:** `OOF Target Encoding function compiled.`

---

## 🔄 CELL 9 — The 5-Fold Cross Validation Loop

**🔎 What is this doing?**

This is the **main training cell**. We train 3 machine learning models using 5-fold cross validation:

### What is 5-Fold Cross Validation?
We divide the training data into 5 equal parts (folds). In each iteration:
1. We train on 4 folds (80% of data)
2. We predict on the remaining 1 fold (20% of data)
3. We do this 5 times, each time using a different fold for validation

This gives us **OOF (Out-Of-Fold) predictions** — predictions for every training row, made when that row was NOT in the training set. These OOF predictions are very reliable estimates of real-world performance.

### Our Three Expert Models:

#### 🟢 LightGBM (lgb_m) — Our Strongest Model
```python
lgb.LGBMRegressor(
    n_estimators=10000,   # Maximum 10,000 trees (but early stopping will stop it earlier)
    num_leaves=255,        # Each tree can have up to 255 leaf nodes (wide, expressive trees)
    learning_rate=0.015,   # Small learning rate = slower but more accurate
    early_stopping=250,    # Stop if no improvement in 250 rounds
    ...
)
```
- **Insight:** LightGBM is a gradient boosting model that grows trees **leaf-first** (finding the most impactful splits regardless of depth). `num_leaves=255` makes it very expressive.
- **Actual performance:** OOF RMSLE = **0.19807** (improved from v6's 0.19908!)
- **Avg iterations:** ~5,992 trees (early stopping did not trigger aggressively)

#### 🔵 XGBoost (xgb_m) — Secondary Model
```python
XGBRegressor(
    n_estimators=10000,
    max_depth=9,           # Trees grow level-by-level up to 9 levels
    learning_rate=0.015,
    early_stopping_rounds=250,
    enable_categorical=True,  # Handles category columns natively
    ...
)
```
- **Actual performance:** OOF RMSLE = **0.20510** (slightly worse than v6's 0.20477)
- **Avg iterations:** ~6,594 trees

#### 🟡 CatBoost (cat_m) — Third Model
```python
CatBoostRegressor(
    iterations=10000,
    learning_rate=0.015,
    grow_policy='Lossguide',   # Leaf-first growth (like LightGBM)
    max_leaves=255,
    cat_features=cat_fold,     # Explicitly passed category columns
    early_stopping_rounds=250,
    ...
)
```
- **Actual performance:** OOF RMSLE = **0.21008** (slightly worse than v6's 0.20940)

**📊 Actual Fold Results:**
| Fold | LGB | XGB | CAT |
|---|---|---|---|
| Fold 1 | 0.19617 (iter=5651) | 0.20306 (iter=6532) | 0.20773 |
| Fold 2 | 0.19867 (iter=5881) | 0.20567 (iter=7755) | 0.21047 |
| Fold 3 | 0.19874 (iter=6196) | 0.20558 (iter=5952) | 0.21066 |
| Fold 4 | 0.19876 (iter=6336) | 0.20652 (iter=7585) | 0.21237 |
| Fold 5 | 0.19798 (iter=5895) | 0.20462 (iter=6144) | 0.20913 |

---

## ⚖️ CELL 10 — Finding the Best Blend Weights

```python
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = max(round(1.0 - w1 - w2, 6), 0.0)
        score = rmsle(np.expm1(y_all),
                      np.expm1(w1*oof_lgb + w2*oof_xgb + w3*oof_cat))
```

**🔎 What is this doing?**

**Story:** Imagine asking three expert appraisers for a price estimate. Expert 1 (LGB) is very accurate. Expert 2 (XGB) is decent. Expert 3 (CAT) is the weakest. How much should you trust each expert?

We test all possible weight combinations (in 5% increments) and find the combination that produces the lowest RMSLE on our OOF predictions.

**📊 Optimal Blend Found:**
- **LGB weight: 75%** (most trustworthy)
- **XGB weight: 0%** (not contributing)
- **CAT weight: 25%**
- **Blended OOF RMSLE: 0.19604** (vs v6's 0.19627 — improvement!)

**Key Insight:** XGBoost got zero weight! This means LGB + CAT together outperforms any combination that includes XGBoost. This is because LGB and CAT make different types of errors (diverse predictions), while XGBoost makes similar errors to LGB without adding new information.

---

## 📤 CELL 11 — Building the Final Submission

```python
test_blend   = w1*test_lgb_folds.mean(1) + w2*test_xgb_folds.mean(1) + w3*test_cat_folds.mean(1)
final_prices = np.expm1(test_blend)
plo = train_df['TargetValue'].quantile(0.001)
phi = train_df['TargetValue'].quantile(0.999)
final_prices = np.clip(final_prices, plo, phi)
```

**🔎 What is this doing?**

1. **Average fold predictions:** For each test row, we average the predictions from all 5 folds for each model (fold-averaged ensemble). This reduces variance.
2. **Blend models:** Use the optimal weights found in Cell 10.
3. **`np.expm1()`:** Reverse the `log1p` transformation to get actual dollar prices.
4. **Clip prices:** Never predict below the 0.1% percentile ($7,500) or above the 99.9% percentile ($137,500). This prevents unrealistically extreme predictions.

**✅ Expected output:**
```
Clipped to [$7,500, $137,500]
Saved submission.csv: 15,000 rows | min=$7,827 median=$34,708 max=$137,500
```

---

## 📈 CELL 12 — Diagnostics and Analysis

**🔎 What is this doing?**

Creates three diagnostic plots and prints key performance metrics for our OOF (out-of-fold) predictions:

### Plot 1: Actual vs Predicted Scatter Plot
- X-axis = actual price, Y-axis = predicted price
- Points close to the red diagonal line = accurate predictions
- Points far from the line = large errors

### Plot 2: Residuals Distribution
- Shows the distribution of prediction errors (actual - predicted)
- Ideally centered around 0 (no systematic over/under-prediction)

### Plot 3: Percentage Error Distribution
- Shows what % error we make, clipped to +/-40%
- A tight distribution around 0% = consistently accurate predictions

**📊 Actual Metrics:**
| Metric | Value | Meaning |
|---|---|---|
| Blended OOF RMSLE | 0.19604 | Main competition metric (lower = better) |
| R² | ~0.9095 | 90.95% of price variance is explained |
| MAE | ~$3,428 | On average, we're off by $3,428 |
| Median % Error | ~7.1% | Half of our predictions are within 7.1% of actual price |

**Top 20 Features by LightGBM Importance:**
LightGBM's feature importance tells us which features it used most for splitting. The top features typically include:
- `Spec_FullDescriptor_TE` (target encoded machine type)
- `SaleDayOfYear` (day of year)
- `MachineAge` (how old the machine is)
- `ManufactureYear` (year the machine was made)
- `Asset_Prev_Mean_Price` (history of past prices)
- `SaleYear`, `RegionCode`, etc.

---

## 🏆 Score Journey Summary

| Notebook | Key Improvement | Kaggle Score |
|---|---|---|
| Original | Basic model | 0.20845 |
| v4 | 5-fold CV + 3 models | 0.19969 |
| v6 | 6000 trees + 255 leaves + lr=0.015 | **0.19086** |
| v8 | Target encoding (with leakage bug) | 0.19996 ❌ |
| **v10** | **Leak-free TE + Asset History** | **0.19020** ✅ |

---

## 📚 Glossary

| Term | Simple Explanation |
|---|---|
| **RMSLE** | Root Mean Squared Log Error — competition metric (lower = better) |
| **OOF Predictions** | Out-of-fold predictions — model predicts on data it never saw during training |
| **Early Stopping** | Stop adding trees when the model stops improving on validation data |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Target Leakage** | When future/secret information accidentally leaks into training features |
| **Target Encoding** | Replacing a category with its average target value |
| **Smoothing (alpha)** | Blending rare category averages toward the global average to reduce overfitting |
| **Ensemble/Blending** | Combining predictions from multiple models for better accuracy |
| **Gradient Boosting** | An ML technique that builds trees sequentially, each fixing the previous one's errors |
| **num_leaves=255** | Each decision tree can have up to 255 leaf nodes (wide, expressive trees) |
| **Learning Rate** | How much each new tree corrects the previous errors (smaller = more careful) |
