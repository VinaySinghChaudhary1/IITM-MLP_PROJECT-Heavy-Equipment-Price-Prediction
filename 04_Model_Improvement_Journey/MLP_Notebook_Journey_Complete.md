# The Complete Story of My Kaggle Journey (v1 to v18)
## Heavy Equipment Selling Price Prediction — Notebook-by-Notebook Walkthrough
### Student: 22f2001153 | Final Kaggle Score: **0.18732** (RMSLE) | Journey: v1 → v6 → v10 → v14 → v15 → v16 → v17 → v18

---

> **HOW TO USE THIS GUIDE:** This document explains the complete story of how we progressed from a baseline RMSLE of **0.2045** to a final score of **0.18732** through eight notebook versions. Each section covers what we did, why we did it, the actual outputs/scores, key terms, and likely viva questions.

---

## Competition Overview

| Item | Details |
|---|---|
| **Competition** | Heavy Equipment Selling Price Prediction |
| **Platform** | Kaggle (private class competition) |
| **Task Type** | Supervised Machine Learning — **Regression** |
| **Metric** | **RMSLE** — Root Mean Squared Log Error (lower is better) |
| **Cutoff** | < 0.20 (to pass Level-1 Viva eligibility) |
| **Training Data** | 138,701 auction records x 50 columns |
| **Test Data** | 15,000 records (no prices — we predict these) |
| **Libraries** | NumPy, Pandas, Sklearn, LightGBM, CatBoost, Seaborn (XGBoost removed in v15) |
| **Our Best Score** | **0.18732** (Improvement #8, v18) |

### What is RMSLE?

RMSLE = sqrt( (1/n) * sum( (log(1 + y_pred) - log(1 + y_true))^2 ) )

**Why RMSLE and not RMSE?** Equipment prices range from $7,500 to $142,000 — a 19x range. RMSE would give 19x more importance to errors on expensive machines. RMSLE penalizes *percentage* errors equally regardless of price scale: a 10% error on a $10,000 machine gets the same penalty as 10% on a $100,000 machine. This is fair for wide-range price prediction.

### Score Progression Summary

| Notebook | Kaggle Score | OOF RMSLE | Key Improvement |
|---|---|---|---|
| **First File (v1)** | ~0.2050 | 0.2045 | Baseline: XGBoost + RandomizedSearchCV |
| **Improvement #2 (v6)** | **0.19086** | 0.19627 | Three-model ensemble + 5-Fold CV |
| **Improvement #3 (v10)** | **0.19020** | 0.19604 | Leak-Free Asset History + OOF Target Encoding |
| **Improvement #4 (v14)** | **0.18964** | 0.19542 | Fold-level group features (fixed leakage bug) |
| **Improvement #5 (v15)** | **0.18847** | 0.19415 | Expanded TE (7 cols) + Interaction TE + Dual LGB (replaced XGB) |
| **Improvement #6 (v16)** | **0.18763** | 0.19359 | Cyclical Dates (sin/cos) + 3 Interaction TEs + Non-Linear Age |
| **Improvement #7 (v17)** | **0.18757** | 0.19359 | SaleDay Cyclical + LogHours×LogAge + 3 Time Interaction TEs |
| **Improvement #8 (v18)** | **0.18732** | **0.19326** | AgeBucket + 4 Dense Interaction TEs + LGB1 num_leaves=1023 |

### Notebook Structure in the Final Combined Notebook

The single Kaggle submission notebook contains all improvement stages:
1. **Improvement No. 1** — Code from First File notebook (commented out)
2. **Improvement No. 2** — Code from v6 notebook (commented out)
3. **Improvement No. 3** — Code from v10 notebook (commented out)
4. **Improvement No. 4** — Code from v14 notebook (commented out)
5. **Improvement No. 5** — Code from v15 notebook (commented out)
6. **Improvement No. 6** — Code from v16 notebook (commented out)
7. **Improvement No. 7** — Code from v17 notebook (commented out)
8. **Improvement No. 8** — Code from v18 notebook (**actively running** — produces final score)

> **Why commented out?** Kaggle requires a single notebook per submission. Earlier code is preserved as evidence of the learning journey, while v18 runs and produces the actual submission.

---

# IMPROVEMENT #1: The First Notebook (Baseline)
## File: 22f2001153-notebook-1.ipynb | Kaggle Score: ~0.2050 | Val RMSLE: 0.2045

---

## The Story

The first notebook was our **starting point** — a structured EDA followed by a standard ML pipeline. It answered the question: *How accurately can we predict equipment prices using XGBoost?*

Think of this as building the foundation of a house. Before the second floor, you need solid ground. The first notebook gave us:
1. Deep understanding of 138,701 auction records with 50 features
2. A working pipeline from raw CSV to Kaggle submission
3. A baseline of ~0.2050 to beat in all future notebooks

---

## Step-by-Step Walkthrough

### Step 1 — Libraries and Setup
```python
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import lightgbm as lgb
SEED = 42
```
**Why SEED = 42?** Reproducibility — every random operation (shuffling, sampling, train/val split) gives the same result each time the notebook runs. This is essential for scientific reproducibility.

### Step 2 — Helper Functions
```python
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # Never negative
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def make_submission(model, X_test_proc, test_ids, log_target=True):
    preds = model.predict(X_test_proc)
    if log_target:
        preds = np.expm1(preds)  # Reverse log1p
    preds = np.maximum(preds, 0)
    return pd.DataFrame({'TransactionID': test_ids, 'TargetValue': preds})
```
**Why define rmsle manually?** Sklearn does not have RMSLE as a built-in scorer. We implement it ourselves to match the competition's exact evaluation formula.

### Step 3 — Data Loading
**Actual output:**
```
TRAIN : 138,701 x 50
TEST  :  15,000 x 49
Target  min=$7,500  median=$35,000  mean=$41,522  max=$142,000
```

### Step 4 — Target Variable Analysis (EDA)
Three plots created:
1. **Original scale histogram** — Right-skewed: most machines $15K–$50K, few $100K+
2. **Log scale histogram** — log1p(TargetValue) is approximately normally distributed!
3. **Box plot** — Shows median, IQR, and outliers
**Why log-transform the target?** Right-skewed data confuses ML models during training. log1p compresses the scale to approximately normal. Also, RMSLE is a log-scale metric — training on log-prices directly optimizes our competition metric. We always predict log1p(price) and reverse with expm1() at prediction time.

### Step 5 — Missing Value Analysis
```
col18 : 99.95% missing → DROPPED entirely (less than 0.05% data)
col19 : 99.95% missing → DROPPED entirely
CabinType  : ~91% missing → kept, created CabinType_present flag
col5       : ~90% missing → kept, created col5_present flag
```
**Why _present flag columns?** The FACT that a value is missing is itself useful information. If OperationalHoursMeter is not recorded, it might mean the meter is broken or the seller chose not to disclose it. The binary flag (1=recorded, 0=missing) captures this signal.

### Step 6 — Key EDA Insights
**Categorical insights (from box plots):**
- `InventoryGroupDescription`: Wheel Loaders ($44K median) > Skid Steer Loaders ($15K)
- `UtilizationTier`: High > Medium > Low — natural ordering exists (important for encoding!)
- `CabinType`: Enclosed cabin machines > open/canopy machines
- `AssetScaleFactor`: Larger machines = higher prices
**Numerical insights (from scatter plots):**
- `ManufactureYear` vs Price: +0.20 linear correlation (newer = more expensive)
- `OperationalHoursMeter` vs Price: ~0.00 LINEAR correlation BUT highly non-linear relationship
**Key insight:** Near-zero linear correlation for hours means linear models (Ridge Regression) cannot capture the hours-price relationship. But tree models find non-linear splits — this is why XGBoost dominates Ridge in our comparison.

### Step 7 — Feature Engineering
```python
def engineer_features(df):
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['SaleYear']        = df['TransactionDate'].dt.year
    df['SaleMonth']       = df['TransactionDate'].dt.month
    df['SaleQuarter']     = df['TransactionDate'].dt.quarter
    df['SaleDayOfYear']   = df['TransactionDate'].dt.dayofyear
    df['MachineAge']      = df['SaleYear'] - df['ManufactureYear']
    df['UsagePerYear']    = df['OperationalHoursMeter'] / (df['MachineAge'] + 1)
    for col in high_missing_cols:
        df[col + '_present'] = df[col].notna().astype(int)
    return df
```
**Why MachineAge?** Captures depreciation. A machine made in 1995 selling in 2012 has 17 years of wear. More informative than using ManufactureYear alone.
**Why UsagePerYear?** Normalizes hours by age. 5,000 hours over 2 years = heavily used machine. 5,000 hours over 10 years = lightly used machine.

### Step 8 — Preprocessing Pipeline
```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
])
```
**Why sklearn Pipeline?** Chains preprocessing steps so: (a) fit() is only called on training data, (b) transform() applies the same parameters to test data — preventing data leakage.

### Step 9 — Five Models Compared (Actual Results)
| Model | Val RMSLE | Notes |
|---|---|---|
| Ridge Regression | ~0.3800 | Linear — cannot capture non-linear price patterns |
| Random Forest | 0.2895 | Bagging ensemble — better but much slower |
| HistGradientBoosting | 0.2327 | Fast but fewer hyperparameter options |
| **XGBoost** | **0.2208** | **Best — gradient boosting with L1+L2 regularization** |
| LightGBM | 0.2217 | Nearly as good as XGBoost, runs much faster |

### Step 10 — RandomizedSearchCV Hyperparameter Tuning
**Best parameters found:**
```
learning_rate    : 0.08    max_depth        : 8
n_estimators     : 1000   colsample_bytree : 0.8
subsample        : 1.0    reg_alpha        : 0.3
reg_lambda       : 1.0    min_child_weight : 5
CV RMSLE : 0.2120   |  Val RMSLE : 0.2208
```

### Step 11 — Final Metrics and Submission
```
RMSLE  : 0.2045    MAE    : $5,399.57
RMSE   : $8,060    R2     : 0.9052 (explains 90.5% of price variance!)
MedAPE : 10.73%    (typical prediction off by ~11%)

Submission: 15,000 rows | Min=$7,251 | Median=$34,580 | Max=$141,053
Negative predictions: 0 ✓
```
**Kaggle Score: ~0.2050** — Beats the 0.85 absolute threshold but NOT the competitive <0.20 cutoff. We need to improve!

---

# IMPROVEMENT #2: Three-Model Ensemble + 5-Fold CV
## File: 22f2001153@improve_6.ipynb | Kaggle Score: **0.19086** | OOF RMSLE: 0.19627

---

## The Story

After scoring ~0.2050, two limitations held us back:
1. **Single model**: No diversity — one model's errors are unchecked
2. **80/20 split**: 20% of data never trains the model

The insight: top Kaggle submissions use **ensembles** — multiple diverse models whose errors cancel each other out. Think of polling multiple experts: averaging their answers beats any one expert.

**Key innovations in v6:**
1. CatBoost added alongside LightGBM + XGBoost (three diverse models)
2. 5-Fold Cross Validation — every row gets to validate!
3. Blend weight optimization via grid search over OOF predictions
4. Fold-averaged test predictions (5 models averaged = more stable)

---

## Step-by-Step Walkthrough

### Step 1 — CatBoost Introduction
**Why CatBoost?** CatBoost handles categorical features **natively** without needing OrdinalEncoder. It uses Ordered Boosting to prevent target leakage in its internal categorical encoding, providing excellent ensemble diversity.

### Step 2 — Feature Engineering (Same as v1)
Same engineer_features() function: date decomposition (SaleYear/Month/Quarter), MachineAge, UsagePerYear, _present flags.

### Step 3 — 5-Fold Cross Validation Loop
```python
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(len(X_all))
oof_xgb = np.zeros(len(X_all))
oof_cat = np.zeros(len(X_all))
test_lgb_folds = np.zeros((len(X_test_final), N_FOLDS))
test_xgb_folds = np.zeros((len(X_test_final), N_FOLDS))
test_cat_folds = np.zeros((len(X_test_final), N_FOLDS))

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_all)):
    X_tr, X_vl = X_all.iloc[tr_idx], X_all.iloc[vl_idx]
    y_tr, y_vl = y_all[tr_idx], y_all[vl_idx]

    # LightGBM (6000 max trees, early stop at 200 rounds)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=6000, learning_rate=0.02, num_leaves=511,
        colsample_bytree=0.7, subsample=0.7, subsample_freq=1,
        reg_alpha=0.5, reg_lambda=1.0, min_child_samples=20, n_jobs=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_lgb[vl_idx] = lgb_model.predict(X_vl)
    test_lgb_folds[:, fold] = lgb_model.predict(X_test)
    # XGBoost and CatBoost follow same pattern...
```

### Step 4 — Blend Weight Optimization (Grid Search)
```python
best_blend, best_w = 999.0, (0.65, 0.05, 0.30)
for w1 in np.arange(0.0, 1.01, 0.05):           # LGB weight
    for w2 in np.arange(0.0, 1.01-w1, 0.05):   # XGB weight
        w3 = round(1.0 - w1 - w2, 6)            # CAT = remainder
        blend = w1*oof_lgb + w2*oof_xgb + w3*oof_cat
        score = rmsle(np.expm1(y_all), np.expm1(blend))
        if score < best_blend:
            best_blend, best_w = score, (w1, w2, w3)
```
**Actual output:** Best weights: LGB=0.65, XGB=0.05, CAT=0.30. Blended OOF RMSLE: **0.19627**.

### Step 5 — Final Test Predictions and Submission
```python
test_blend = 0.65 * test_lgb_folds.mean(axis=1) + 0.05 * test_xgb_folds.mean(axis=1) + 0.30 * test_cat_folds.mean(axis=1)
final_prices = np.clip(np.expm1(test_blend), 7500, 137500)
```
**Kaggle Score: 0.19086** — Beats the <0.20 cutoff!

---

# IMPROVEMENT #3: Leak-Free Asset History + OOF Target Encoding
## File: 22f2001153@improve_10.ipynb | Kaggle Score: **0.19020** | OOF RMSLE: 0.19604

---

## The Story

After 0.19086, the breakthrough came from a key observation: **many machines are resold multiple times.** The same bulldozer (identified by AssetID) can appear in training (sold 2008) and in test (to be predicted for 2012).
**Data investigation revealed:** 21.4% of test machines had previously appeared in training data! For these machines, we have DIRECT price history — the strongest predictor.
**Also: A painful lesson between v8 and v10.** In v8, we tried naive target encoding to handle 3,505 unique equipment configurations. The score WORSENED from 0.19086 to 0.19996. The model 'cheated' during training using its own answers, but this cheat didn't work on test data. v10 fixed this with proper Out-of-Fold encoding.

**Two key innovations drove v10:**
1. Leak-Free Asset History Features (4 new features from past sale data)
2. OOF Target Encoding (replacing naive/cheating encoding)

---

## Step-by-Step Walkthrough

### Step 1 — The Discovery: 21.4% Asset Overlap
overlap = set(test_df['AssetID']) & set(train_df['AssetID'])
pct = len(overlap) / len(test_df['AssetID'].unique()) * 100
If a machine sold for $45,000 in 2010, it will likely sell for ~$40,000 in 2012 (depreciation). Direct history beats any indirect engineered feature!

### Step 2 — Building Leak-Free Asset History Features
```python
full_df = pd.concat([train_df, test_df]).sort_values('TransactionID').reset_index(drop=True)
full_df['log_target'] = np.log1p(full_df['TargetValue'])
grouped = full_df.groupby('AssetID')
full_df['Asset_Prev_Sales_Count'] = grouped.cumcount().astype(np.int32)
full_df['cum_sum_log'] = grouped['log_target'].cumsum()
full_df['cum_sum_prev'] = full_df['cum_sum_log'] - full_df['log_target'].fillna(0)
full_df['Asset_Prev_Mean_Price'] = np.where(
    full_df['cum_count_prev'] > 0,
    full_df['cum_sum_prev'] / full_df['cum_count_prev'], np.nan)
full_df['Asset_Prev_Last_Price'] = grouped['log_target'].shift(1)
full_df['Asset_Days_Since_Last_Sale'] = (
    full_df['TransactionDate'] - grouped['TransactionDate'].shift(1)).dt.days
```

### Step 3 — Extended Feature Engineering (v10 additions)
```python
df['IsMonthEnd']   = df['TransactionDate'].dt.is_month_end.astype(np.int8)
df['IsMonthStart'] = df['TransactionDate'].dt.is_month_start.astype(np.int8)
df['IsWeekend']    = (df['SaleDayOfWeek'] >= 5).astype(np.int8)
df.loc[df['ManufactureYear'] <= 1920, 'ManufactureYear'] = np.nan  # entry errors
df.loc[df['OperationalHoursMeter'] == 0, 'OperationalHoursMeter'] = np.nan  # broken meter
df['UtilizationTier'] = df['UtilizationTier'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['AssetScaleFactor'] = df['AssetScaleFactor'].map(
    {'Mini': 0, 'Compact': 1, 'Small': 2, 'Medium': 3, 'Large': 4, 'Extra Large': 5})
```

### Step 4 — OOF Target Encoding
Inside each outer fold (training = 80% of data), we split AGAIN into 5 inner folds. For each training row, compute its encoding using only the other 4 inner folds — it NEVER sees its own target value. Smoothing alpha=15.0 blends rare categories toward global mean. Applied to 4 columns: Spec_FullDescriptor, Spec_BaseClass, ProductConfigID, FunctionalClassification.

### Step 5 — Results
**Blend weights:** LGB=0.75, XGB=0.00, CAT=0.25 | **Blended OOF RMSLE:** **0.19604**
**Kaggle Score: 0.19020** — Improvement of 0.00066 over v6!

---

# IMPROVEMENT #4: Fold-Level Group Features (Leak-Fixed Ensemble)
## File: 22f2001153@improve_14.ipynb | Kaggle Score: **0.18964** | OOF RMSLE: 0.19542

---

## The Story

After achieving 0.19020 with v10, intermediate versions v12 and v13 introduced frequency encoding and group-mean features but caused the score to WORSEN to ~0.191–0.192.
**Root cause — two types of leakage:**
**Type 1 — Scale Mismatch:** Frequency encoding computed on ALL 138,701 training rows. ProductConfigID_Freq for a common config = 900 in training. But test set has only 15,000 rows, so same config appears only ~100 times in test. The model trained expecting '900 = common config' but saw '100' during test prediction. This 9x discrepancy severely confused all models.
**Type 2 — Validation Fold Leakage:** Group statistics computed on the whole training dataset included validation fold rows, which influenced their own encodings.
**The v14 Fix:** Compute ALL group statistics INSIDE the cross-validation loop, using ONLY the training fold data. Validation and test rows look up statistics from training rows only.
**Result: 0.19020 → 0.18964 — Our Best Score at that point!**

---

## Step-by-Step Walkthrough

### Step 1 — Same Foundation as v10
Asset history features and base preprocessing identical to v10. Key difference: frequency encoding and group-mean features REMOVED from base preprocessing — they now compute fold-by-fold inside the CV loop.

### Step 2 — New Function: add_fold_group_features()
```python
def add_fold_group_features(X_tr, y_tr, X_vl, X_te):
    X_tr, X_vl, X_te = X_tr.copy(), X_vl.copy(), X_te.copy()
    global_mean = float(y_tr.mean())

    # Frequency encoding from training fold ONLY
    config_freq = X_tr['ProductConfigID'].value_counts().to_dict()
    X_tr['ProductConfigID_Freq'] = X_tr['ProductConfigID'].map(config_freq).fillna(0)
    X_vl['ProductConfigID_Freq'] = X_vl['ProductConfigID'].map(config_freq).fillna(0)
    X_te['ProductConfigID_Freq'] = X_te['ProductConfigID'].map(config_freq).fillna(0)

    inv_freq = X_tr['InventoryGroupDescription'].value_counts().to_dict()
    X_tr['InvGroup_Freq'] = X_tr['InventoryGroupDescription'].map(inv_freq).fillna(0)
    X_vl['InvGroup_Freq'] = X_vl['InventoryGroupDescription'].map(inv_freq).fillna(0)
    X_te['InvGroup_Freq'] = X_te['InventoryGroupDescription'].map(inv_freq).fillna(0)

    # Group-mean price from training fold ONLY
    config_means = X_tr.groupby('ProductConfigID').apply(
        lambda grp: y_tr[grp.index].mean() if len(grp) > 0 else global_mean
    )
    X_tr['Config_MeanPrice'] = X_tr['ProductConfigID'].map(config_means).fillna(global_mean)
    X_vl['Config_MeanPrice'] = X_vl['ProductConfigID'].map(config_means).fillna(global_mean)
    X_te['Config_MeanPrice'] = X_te['ProductConfigID'].map(config_means).fillna(global_mean)

    return X_tr, X_vl, X_te
```

### Step 3 — Updated CV Loop
```python
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_all), 1):
    X_tr_raw, X_vl_raw = X_all.iloc[tr_idx].copy(), X_all.iloc[vl_idx].copy()
    y_tr, y_vl = y_all[tr_idx], y_all[vl_idx]
    X_tr, X_vl, X_te = add_oof_target_encoding(
        X_tr_raw, y_tr, X_vl_raw, X_test_final.copy(), TE_COLS, alpha=15.0)
    X_tr, X_vl, X_te = add_fold_group_features(X_tr, y_tr, X_vl, X_te)
    # Train LightGBM, XGBoost, CatBoost as usual
```
**Blend weights:** LGB=0.70, XGB=0.00, CAT=0.30 | **Blended OOF RMSLE:** **0.19542**
**Kaggle Score: 0.18964**

---

# IMPROVEMENT #5: Expanded Target Encoding & Dual LightGBM Ensemble
## File: 22f2001153improve_15.ipynb | Kaggle Score: **0.18847** | OOF RMSLE: 0.19415

---

## The Story

v14 achieved score **0.18964**. In v15, we realized we were leaving predictive power on the table in three areas:
1. **Under-utilized categories:** We had only target encoded 4 features. There were other categorical features with massive impact on price.
2. **Regional pricing differences:** Equipment prices are not the same nationwide. We needed regional interaction features.
3. **Dead Weight Model:** XGBoost was consistently getting **0% weight** in our blended predictions because its errors were identical to LightGBM. We needed a model that would actually help LightGBM.
**Key innovations in v15:**
1. **Expanded Target Encoding (7 cols vs 4 in v14):** Added `Spec_SubClass` (eta²=0.24), `Spec_VariantModifier` (eta²=0.22), and `RegionCode` (53 regions).
2. **Interaction Target Encoding:** Grouped `InventoryGroupDescription` x `RegionCode` (OOF TE, alpha=20) to capture regional premiums per machine type.
3. **Replaced XGBoost with LGB2:** Replaced XGBoost with a second LightGBM model (`seed=123`) with slightly different parameters (different random subsampling and regularization) to introduce diverse predictions.
4. **Tuned LightGBM parameters:** Slower learning rate (`lr=0.010` vs `0.015`) and deeper trees (`num_leaves=511` vs `255`) for more precise, complex rules.
5. **Broader Group Stats:** Added `VendorPartnerID` and `RegionCode` to the grouping columns for fold-level group statistics.

---

## Step-by-Step Walkthrough

### Step 1 — Replacing XGBoost (Dual LGB Setup)
In v14, our blend weights were LGB=0.70, XGB=0.00, CAT=0.30. XGBoost was useless. In v15, we defined two distinct LightGBM configurations:
- **LGB1 (seed=42):** primary model, `subsample=0.75`, `colsample_bytree=0.70`, `reg_lambda=3.0` L2 regularization.
- **LGB2 (seed=123):** diversity model, `subsample=0.80`, `colsample_bytree=0.65`, `reg_alpha=0.2` L1, `reg_lambda=2.0` L2 regularization.
**Why this works:** Changing the random seed and structure parameters forces the model to construct different trees, producing distinct errors. When blended, their errors cancel out!

### Step 2 — Expanded OOF Target Encoding (7 columns)
```python
TE_COLS = [
    'Spec_FullDescriptor', 'Spec_BaseClass', 'ProductConfigID',
    'FunctionalClassification',
    'Spec_SubClass',           # NEW in v15! eta^2 = 0.24
    'Spec_VariantModifier',     # NEW in v15! eta^2 = 0.22
    'RegionCode'                # NEW in v15! 53 regions
]
```

### Step 3 — InventoryGroupDescription x RegionCode Interaction TE
```python
def add_interaction_target_encoding(X_tr, y_tr, X_vl, X_te, col1, col2, alpha=20):
    # Combine two columns into a single temporary key
    key_tr = X_tr[col1].astype(str) + '_' + X_tr[col2].astype(str)
    key_vl = X_vl[col1].astype(str) + '_' + X_vl[col2].astype(str)
    key_te = X_te[col1].astype(str) + '_' + X_te[col2].astype(str)
    # ... (OOF mapping logic) ...
    return X_tr, X_vl, X_te
```

### Step 4 — Results
**OOF scores:** LGB1=0.19609 | LGB2=0.19585 | CAT=0.20481
**Best blend weights:** LGB1=30% | LGB2=40% | CAT=30% | **Blended OOF RMSLE:** **0.19415** (v14 was 0.19542)
**Kaggle Score: 0.18847** — Massive improvement of 0.00117 points!

---

# IMPROVEMENT #6: Cyclical Date Encoding & Multi-Interaction Ensemble
## File: 22f2001153improve_16.ipynb | Kaggle Score: **0.18763** | OOF RMSLE: **0.19359**

---

## The Story

v15 achieved score **0.18847**. To squeeze out the last bit of performance, we performed feature importance analysis on v15. We made two surprising discoveries:
1. **Seasonal boundaries:** `SaleDayOfYear` (1 to 365) was one of the most important features. But raw integers represent December 31 as 365 and January 1 as 1. The model sees them as 364 days apart, when in reality they are consecutive days! We needed a representation that wraps around.
2. **Non-linear depreciation:** Heavy equipment depreciates much faster in the first 2-3 years, and then the price stabilizes. Giving them mathematical transformations of age (`Age²` and `LogAge`) makes their splits far more accurate.
3. **Power of interactions:** The single interaction feature in v15 was so successful (ranked #8 in importance) that we knew adding more high-value interactions would boost performance.
**Key innovations in v16:**
1. **Cyclical Date Encoding:** Encoded day-of-year, month, and day-of-week using **sine and cosine transformations**. This maps date boundaries into a continuous circle.
2. **Three Interaction Target Encodings:** Added `FunctionalClassification` x `RegionCode` and `Spec_SubClass` x `RegionCode` interactions (OOF TE, alpha=22) to model how specific subclasses and classifications perform across regions.
3. **Non-Linear Age Features:** Created `MachineAge_sq` (exponential depreciation) and `LogAge` (smoothed tail depreciation).
4. **Binary Asset History Flag:** Added `Has_Asset_History` (1 if prior sales count > 0, 0 if 0). Only 14.4% of machines have price history — this flag explicitly highlights this group to the model.
5. **FunctionalClassification Group Stats:** Added `FunctionalClassification` (eta²=0.352 — the strongest categorical feature) as a grouping column for fold-level group statistics.

---

## Step-by-Step Walkthrough

### Step 1 — Cyclical Date Encoding (sin/cos)
```python
df['SaleDayOfYear_sin'] = np.sin(2 * np.pi * df['SaleDayOfYear'] / 365)
df['SaleDayOfYear_cos'] = np.cos(2 * np.pi * df['SaleDayOfYear'] / 365)
df['SaleMonth_sin']     = np.sin(2 * np.pi * df['SaleMonth'] / 12)
df['SaleMonth_cos']     = np.cos(2 * np.pi * df['SaleMonth'] / 12)
df['SaleDayOfWeek_sin'] = np.sin(2 * np.pi * df['SaleDayOfWeek'] / 7)
df['SaleDayOfWeek_cos'] = np.cos(2 * np.pi * df['SaleDayOfWeek'] / 7)
```
**How this helps:** On a unit circle, the sine/cosine values for Day 365 (Dec 31) are almost identical to Day 1 (Jan 1). The model now understands that December 31 and January 1 represent the same winter season, correcting a major tree-split limitation.

### Step 2 — Non-Linear Depreciation Features
```python
df['MachineAge']    = (df['SaleYear'] - df['ManufactureYear']).clip(lower=0)
df['MachineAge_sq'] = df['MachineAge'] ** 2
df['LogAge']        = np.log1p(df['MachineAge'])
```
A machine loses massive value in its first few years (`MachineAge_sq` accelerates this difference). For old machines, a difference between 20 years and 25 years is minimal (`LogAge` compresses this difference). This math aligns perfectly with real-world depreciation.

### Step 3 — Results
**OOF scores:** LGB1=0.19540 | LGB2=0.19521 | CAT=0.20438
**Best blend weights:** LGB1=30% | LGB2=45% | CAT=25% | **Blended OOF RMSLE:** **0.19359** (v15 was 0.19415)
**Kaggle Score: 0.18763** — Our best score at that point!

---

# IMPROVEMENT #7: SaleDay Cyclical Encoding & Time-Based Interaction TEs
## File: 22f2001153@improve_17.ipynb | Kaggle Score: **0.18757** | OOF RMSLE: **0.19359**

---

## The Story

v16 achieved score **0.18763**. In v17, we took a data-driven approach to identify further gains by inspecting feature importance and statistical relations:
1. **Missed cyclical feature:** `SaleDay` (day of month: 1–31) ranked #3 in importance in v16! But it was treated as a raw integer. December 31 and January 1 are consecutive, yet the model saw 31 and 1 as far apart. We needed to apply sin/cos to the day of the month.
2. **Logical Interactions:** A young machine with high hours is very different from an old machine with the same hours. We added a `LogHours_x_LogAge` interaction to capture usage intensity in context.
3. **Macro Time Trend:** Captured general price trends since 2000 using a clipped `Years_Since_2000` variable to capture inflation/market conditions.
4. **Year-by-Year Equipment Pricing (Time Interaction TEs):** Equipment values fluctuate over years. We created 3 time-based interactions: `ProductConfigID` x `SaleYear` (eta²=0.8443 — extremely strong!), `Spec_BaseClass` x `SaleYear`, and `FunctionalClassification` x `SaleYear` to capture this.
5. **Outlier Mitigation:** Clipped `OperationalHoursMeter` at the 99th percentile to prevent extreme values from distorting splits.

---

## Step-by-Step Walkthrough

### Step 1 — SaleDay sin/cos Cyclical Encoding
```python
df['SaleDay_sin'] = np.sin(2 * np.pi * df['SaleDay'] / 31)
df['SaleDay_cos'] = np.cos(2 * np.pi * df['SaleDay'] / 31)
```
This maps the day of the month onto a circle, so Day 31 and Day 1 map to nearly the same coordinates, preserving the boundary relation.

### Step 2 — LogHours x LogAge Interaction
```python
df['LogHours_x_LogAge'] = df['LogHours'] * df['LogAge'].fillna(0)
```
This multiplies log-hours and log-age, giving a clear continuous representation of total usage relative to machine age.

### Step 3 — Time-Based Interaction Target Encodings
We added 3 time interactions computed OOF with high smoothing:
- `ProductConfigID` x `SaleYear` (eta²=0.8443, alpha=25)
- `Spec_BaseClass` x `SaleYear` (eta²=0.5240, alpha=25)
- `FunctionalClassification` x `SaleYear` (eta²=0.3914, alpha=20)
This allowed the model to adjust predictions dynamically based on how specific configurations priced in specific sale years.

### Step 4 — Results
**OOF scores:** LGB1=0.19542 | LGB2=0.19527 | CAT=0.20332 (CAT OOF improved by +0.001!)
**Best blend weights:** LGB1=30% | LGB2=40% | CAT=30% | **Blended OOF RMSLE:** **0.19359**
**Kaggle Score: 0.18757** — Beat v16 on the leaderboard!

---

# IMPROVEMENT #8: Density-Checked Interaction TEs & AgeBucket Categorical
## File: 22f2001153@improve_18.ipynb | Kaggle Score: **0.18732** | OOF RMSLE: **0.19326**

---

## The Story

v17 achieved score **0.18757**. In v18, we discovered a core mathematical weakness in v17's time-based interaction target encodings: **extreme sparsity**.

**The Sparsity Problem:** `ProductConfigID` x `SaleYear` has 16,316 potential combination cells. Dividing 138,701 training rows by 16,316 gives an average of **only 8 samples per cell**. Most cells only had 0 or 1 row! A target encoding computed on 1 row is pure noise. Despite target smoothing, the model was learning noise, wasting trees.

**The Density-Check Philosophy (v18 Innovation):** We defined a rule: only use interaction target encodings where the average cell density is **greater than 30 (preferably >100) samples per cell**.

**Key innovations in v18:**
1. **AgeBucket Categorical Feature:** Created a 5-bin categorical age grouping (`New`, `Young`, `Mid`, `Old`, `Ancient`).
2. **Dense Time Interactions:** Replaced the sparse v17 time interactions with 4 high-density combinations: `AgeBucket` x `SaleYear` (1,981 samples/cell!), `InventoryGroupCategory` x `SaleYear` (124 samples/cell), `RegionCode` x `SaleYear` (187 samples/cell), and `Spec_SubClass` x `SaleYear` (67 samples/cell).
3. **Increased Model Capacity:** Increased LightGBM 1 `num_leaves` from 511 to **1023** to allow trees to capture these rich, dense interaction features.

---

## Step-by-Step Walkthrough

### Step 1 — AgeBucket Categorical Feature Creation
```python
age_bins   = [-1, 2, 5, 10, 20, 200]
age_labels = ['New', 'Young', 'Mid', 'Old', 'Ancient']
df['AgeBucket'] = pd.cut(df['MachineAge'], bins=age_bins, labels=age_labels)
df['AgeBucket'] = df['AgeBucket'].astype(str).fillna('Unknown').astype('category')
```
This groups ages into 5 categories, enabling a highly dense cross with `SaleYear` (5 x 14 = 70 combination cells).

### Step 2 — Density-First Interaction Design
We dropped the sparse ProductConfigID x SaleYear (8 samples/cell) and only kept interactions passing our density check (>30 samples/cell). This ensured target encodings were calculated on stable, representative sample sizes.

### Step 3 — Increasing LGB1 num_leaves to 1023
Since we added dense, high-value interaction features, we doubled LGB1's maximum leaf nodes (`num_leaves=1023`) to allow the model to learn complex joint splits of these interactions.

### Step 4 — Results
**OOF scores:** LGB1=0.19509 | LGB2=0.19491 | CAT=0.20323 (all three models improved!)
**Best blend weights:** LGB1=25% | LGB2=45% | CAT=30% | **Blended OOF RMSLE:** **0.19326** (improved by 0.00033)
**Kaggle Score: 0.18732** — Our absolute best score!

---

# COMPARATIVE ANALYSIS: All Eight Improvements

---

## Score Journey

```
BASELINE (v1)     : RMSLE ~0.2050  (XGBoost + RandomizedSearchCV)
IMPROVEMENT #2 (v6) : RMSLE 0.19086  (LGB+XGB+CAT ensemble)
IMPROVEMENT #3 (v10): RMSLE 0.19020  (Leak-free history + OOF TE)
IMPROVEMENT #4 (v14): RMSLE 0.18964  (Fold-level group features)
IMPROVEMENT #5 (v15): RMSLE 0.18847  (Expanded TE (7 cols) + Dual LGB)
IMPROVEMENT #6 (v16): RMSLE 0.18763  (Cyclical Dates + Non-Linear Age)
IMPROVEMENT #7 (v17): RMSLE 0.18757  (SaleDay Cyclical + Time TEs)
IMPROVEMENT #8 (v18): RMSLE 0.18732  <-- BEST KAGGLE SCORE!
```

## Feature Engineering Progression

| Feature | v1 | v6 | v10 | v14 | v15 | v16 | v17 | v18 |
|---|---|---|---|---|---|---|---|---|
| Date (Year/Month/Quarter) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Extended date (Week/DayOfWeek) | No | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Calendar flags (IsMonthEnd/Weekend) | No | No | Yes | Yes | Yes | Yes | Yes | Yes |
| MachineAge | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| UsagePerYear | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Missing value flags (_present) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **Asset History (4 features)** | No | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **OOF Target Encoding (TE)** | No | No | Yes (4) | Yes (4) | Yes (7) | Yes (7) | Yes (7) | Yes (7) |
| **Fold-level Freq & Group stats** | No | No | No | Yes | Yes | Yes | Yes | Yes |
| **LGB2 Diversity Model** | No | No | No | No | Yes | Yes | Yes | Yes |
| **Interaction TEs (OOF)** | No | No | No | No | Yes (1) | Yes (3) | Yes (6) | Yes (7) |
| **Cyclical Date (sin/cos)** | No | No | No | No | No | Yes (3) | Yes (4) | Yes (4) |
| **Non-Linear Age (Sq/Log)** | No | No | No | No | No | Yes | Yes | Yes |
| **LogHours x LogAge Interaction** | No | No | No | No | No | No | Yes | Yes |
| **Years_Since_2000 Trend** | No | No | No | No | No | No | Yes | Yes |
| **Hours Outlier Clipping** | No | No | No | No | No | No | Yes | Yes |
| **AgeBucket Categorical** | No | No | No | No | No | No | No | Yes |
| **Density-Checked Interactions** | No | No | No | No | No | No | No | Yes |
| Feature count (to model) | 65 | 62 | 72 | 68+f | 71+f | 80+f | 84+f | 85+f |

## Model Architecture Progression

| Component | v1 | v6 | v10 | v14 | v15 | v16 | v17 | v18 |
|---|---|---|---|---|---|---|---|---|
| Models | 1 (XGB) | 3 | 3 | 3 | 3 (Dual LGB) | 3 | 3 | 3 |
| Validation | 80/20 | 5-Fold | 5-Fold | 5-Fold | 5-Fold | 5-Fold | 5-Fold | 5-Fold |
| LGB1 weight | N/A | 0.65 | 0.75 | 0.70 | 0.30 | 0.30 | 0.30 | 0.25 |
| LGB2 weight | N/A | N/A | N/A | N/A | 0.40 | 0.45 | 0.40 | 0.45 |
| XGB weight | 1.00 | 0.05 | 0.00 | 0.00 | N/A | N/A | N/A | N/A |
| CAT weight | N/A | 0.30 | 0.25 | 0.30 | 0.30 | 0.25 | 0.30 | 0.30 |
| Price clipping | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| LGB1 num_leaves | N/A | 511 | 511 | 255 | 511 | 511 | 511 | **1023** |

## Leakage Prevention Timeline

```
v1:  OK -- standard train/val split, no encoding
v6:  OK -- 5-Fold CV, no encoding
v8:  BUG -- Naive Target Encoding -> WORSENED to 0.19996 (leakage!)
v10: FIXED -- OOF Target Encoding -> 0.19020
v12: BUG -- Global frequency/group-mean -> scale mismatch + leakage
v13: BUG -- Same issue, still poor (~0.191)
v14: FIXED -- Fold-level group features -> 0.18964
v15: OK -- All encodings kept Out-of-Fold -> 0.18847
v16: OK -- Safe CV loop, cyclical dates, multiple interactions -> 0.18763
v17: OK -- Time interactions added but sparse -> 0.18757
v18: OK -- Density-checking on interactions, increased leaves -> 0.18732
```

---

# MASTER VIVA Q&A: Complete Reference

---

## Story-Type Questions

**Q: Walk me through the progression from your first notebook to your best score.**
I started with a baseline XGBoost model scoring ~0.2050. I then made seven key improvements:

**Improvement #1 (v6, 0.19086):** Switched from single XGBoost to a three-model ensemble (LightGBM + XGBoost + CatBoost) with 5-Fold CV. Every row now contributes to training; fold-averaging gives stable test predictions; blend weight optimization found LGB=65%, XGB=5%, CAT=30%.

**Improvement #2 (v10, 0.19020):** Discovered 21.4% of test machines had previously appeared in training data (same AssetID). Added Leak-Free Asset History Features using temporal ordering (cumcount/shift). Also added OOF Target Encoding for 40 high-cardinality categorical columns — after learning that naive encoding in v8 caused the score to WORSEN to 0.19996.

**Improvement #3 (v14, 0.18964):** Fixed a leakage bug from v12/v13 where frequency encoding and group-mean features were computed globally (all 138K rows). In v14, these are computed INSIDE the CV loop using only training fold data. This prevents both scale mismatch (9x train/test ratio) and validation leakage.

**Improvement #4 (v15, 0.18847):** Expanded Target Encoding from 4 to 7 columns (adding SubClass, VariantModifier, and RegionCode based on eta² analysis). Added an interaction target encoding (InventoryGroupDescription x RegionCode) to capture regional premiums. Replaced the useless XGBoost (0% weight) with a second LightGBM model (LGB2, seed=123) to introduce prediction diversity.

**Improvement #5 (v16, 0.18763):** Encoded dates cyclically (sin/cos of day, month, day of week) so year boundaries align. Added two new interaction target encodings, non-linear age features (`MachineAge_sq` and `LogAge`), and a binary `Has_Asset_History` flag.

**Improvement #6 (v17, 0.18757):** Applied cyclical encoding to `SaleDay` (day of month), added `LogHours_x_LogAge` to model age-relative usage intensity, added `Years_Since_2000` trend, clipped hour outliers at the 99th percentile, and added 3 time-based interactions.

**Improvement #7 (v18, 0.18732):** Addressed v17's sparsity issues by implementing a **Density-Check philosophy** (>30 average samples per combination cell). Created an `AgeBucket` categorical feature and constructed 4 dense time-based interactions. Doubled LGB1 capacity (`num_leaves=1023`) to capture these dense interactions, yielding our best score of 0.18732.

**Q: What was the single most important improvement in your project?**
The Leak-Free Asset History Features in v10 were the most impactful discovery. Finding that 21.4% of test machines had price history in training data was a major breakthrough. For these resold machines, the model essentially performs a lookup of past auction prices instead of predicting them from scratch. This single innovation drove our largest jump in accuracy.

---

## Theory Questions

**Q: What is Bagging vs Boosting?**
Bagging (e.g. Random Forest): Trees are trained independently in parallel on random bootstrap samples of data, and their outputs are averaged. Goal: reduce variance. Example: Random Forest scored 0.2895 in our project.
Boosting (e.g. LightGBM, CatBoost): Trees are trained sequentially, where each new tree is built to correct the residual errors made by the previous trees. Goal: reduce bias. Example: LightGBM scored 0.197. Boosting performs far better on our dataset because equipment price prediction requires learning complex non-linear combinations of features, which boosting does iteratively.

**Q: What is L1 vs L2 regularization?**
L1 Regularization (Lasso): Adds the absolute sum of weights `lambda * sum(|w|)` to the loss function. This drives weights of non-informative features to exactly zero, performing automatic feature selection.
L2 Regularization (Ridge): Adds the squared sum of weights `lambda * sum(w^2)` to the loss function. This shrinks weights of correlated features toward zero but keeps them in the model, promoting stability.
Our LightGBM uses both: `reg_alpha=0.1` (L1) and `reg_lambda=3.0` (L2) to prevent overfitting on our 85+ engineered features.
---

## Code Templates for Live Coding

**Template 1: RMSLE from scratch**
```python
import numpy as np
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # Clip negatives
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
```

**Template 2: Cyclical Encoding**
```python
import numpy as np
def encode_cyclical(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df
```

**Template 3: 5-Fold CV skeleton**
```python
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
    y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
    model.fit(X_tr, y_tr)
    oof_preds[vl_idx] = model.predict(X_vl)      # OOF
    test_preds += model.predict(X_test) / 5      # Fold-average
```

---

## Quick Reference Card

| Topic | Answer |
|---|---|
| Problem type | Regression (predicting continuous auction price) |
| Metric | RMSLE (lower is better) |
| Our best score | **0.18732** (v18) |
| Cutoff | < 0.20 RMSLE |
| Training rows | 138,701 |
| Test rows | 15,000 |
| Raw features | 50 |
| Number of notebooks | 8 (v1, v6, v10, v14, v15, v16, v17, v18) |
| Final models | LightGBM 1 (25%) + LightGBM 2 (45%) + CatBoost (30%) |
| Why Dual LightGBM | LGB2 with seed=123 replaces useless XGBoost (got 0% weight) |
| Test-train overlap | 21.4% of test machines appeared in training (same AssetID) |
| Innovation #1 | Leak-Free Asset History Features (v10) |
| Innovation #2 | OOF Target Encoding (7 columns) (v15) |
| Innovation #3 | Density-Checking for Interaction TEs (>30 samples/cell) (v18) |
| Innovation #4 | AgeBucket categorical crossing with SaleYear (v18) |
| Innovation #5 | Increased capacity (LGB1 `num_leaves=1023`) (v18) |
| Why log1p? | Right-skewed prices; RMSLE is log-scale metric |
| CV type | 5-Fold KFold (shuffle=True, random_state=42) |
| Price clipping | Q0.1% to Q99.9% = [$7,500, $137,500] |

## Feature Importance (v18 Final Model)

| Rank | Feature | Why Important |
|---|---|---|
| 1 | RegionCode__x__SaleYear_TE | Dense time interaction TE — captures regional year trends |
| 2 | MachineAge_rel_ProductConfigID | Relative age of machine compared to config median |
| 3 | InventoryGroupCategory__x__SaleYear_TE | Dense time interaction TE — captures type trends |
| 4 | RegionCode | Baseline regional price level |
| 5 | AgeBucket__x__SaleYear_TE | Dense time interaction TE — captures age-bracket year trends |
| 6 | SaleDayOfYear_cos / _sin | Cyclical year date features — map winter boundaries correctly |
| 7 | ProductConfigID | Raw machine configuration |
| 8 | Spec_SubClass__x__SaleYear_TE | Dense time interaction TE — captures subclass year trends |
| 9 | OperationalHoursMeter | Machine total hours usage (wear signal) |
| 10 | Config_MeanPrice | Fold-level group mean price — config price baseline |

---

## Level 1 and Level 2 Viva Checklist

**Level 1 Requirements (all must be met):**
- [x] Problem statement understood: regression on heavy equipment auction prices
- [x] Dataset loaded correctly: train (138,701 rows, 50 cols), test (15,000 rows, 49 cols)
- [x] EDA: univariate (histograms, box plots), bivariate (scatter, correlation)
- [x] Missing values: detected, handled (median imputation, 'Missing' fill, _present flags)
- [x] Scaling: StandardScaler in v1 pipeline (not needed for trees — documented)
- [x] Encoding: OrdinalEncoder + OOF Target Encoding for categoricals
- [x] Feature engineering: MachineAge, UsagePerYear, date decomposition, asset history
- [x] Multiple models: Ridge, Random Forest, HistGB, XGBoost, LightGBM, CatBoost
- [x] Hyperparameter tuning: RandomizedSearchCV (v1) + early stopping (v6/v10/v14/v15/v16/v17/v18)
- [x] Performance analysis: residuals, percentage error, scatter, feature importance
- [x] Score comparison: all models compared, RMSLE documented
- [x] Final submission: best model (Dual LGB + CAT ensemble, RMSLE=0.18732)

**Level 2 Preparation (conceptual + mathematical + coding):**
- [x] Can explain every model used: data, model, loss function, optimization, evaluation
- [x] Can explain L1/L2 regularization with formulas
- [x] Can write RMSLE, 5-Fold CV, RandomizedSearchCV, Cyclical Encoding from scratch
- [x] Can explain data leakage and all 3 instances in our project (naive TE, global freq, fold leakage)
- [x] Can explain OOF target encoding step by step
- [x] Can explain asset history feature construction
- [x] Can compare Bagging vs Boosting with examples
- [x] Can explain why Kaggle score (0.18732) is better than OOF RMSLE (0.19326)

---

## ⚡ Token-Saving Resume Instructions
If Antigravity stops due to token limits or resets, run the following command in the chat:

```
Read the D:\Download\AI MLP Project\viva_progress.md file.
We are preparing for the MLP Level 1 & 2 Viva on Kaggle Heavy Equipment Price Prediction.
Student ID: 22f2001153. Best score: 0.18732 (v18 notebook).
Identify the next task from the NEXT TASKS section and continue from there.
```

*Document End — Version 3 Complete*
