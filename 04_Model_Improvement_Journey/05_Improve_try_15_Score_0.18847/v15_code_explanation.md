# 📖 v15 Code Explanation — Easy Story Guide
## `22f2001153improve_15.ipynb`
### Heavy Equipment Price Prediction | Score Journey: 0.19020 (v10) → 0.18964 (v14) → **0.18847 (v15)**

---

> **How to read this guide:** Imagine you are trying to predict the price of a used bulldozer or excavator at an auction. You have data about thousands of past auctions. Your goal is to build a "smart calculator" that uses this historical data to predict prices for new auctions.

---

## 🚀 What Changed from v14 to v15?

v14 achieved Kaggle score **0.18964**. v15 improves upon it with 5 key changes:

| # | Improvement | Why It Helps |
|---|---|---|
| 1 | **Expanded Target Encoding** (4 cols → 7 cols) | Added Spec_SubClass (eta²=0.24), Spec_VariantModifier (eta²=0.22), RegionCode (53 unique values) |
| 2 | **Interaction Target Encoding** (InventoryGroupDescription x RegionCode) | Captures regional pricing premiums per machine type |
| 3 | **Tuned LightGBM** (lr=0.010, num_leaves=511) | Deeper more accurate trees vs v14 (lr=0.015, num_leaves=255) |
| 4 | **Replaced XGBoost with LGB2** | XGBoost got 0% blend weight in v14. Added second LightGBM (seed=123) for diversity |
| 5 | **Broader Group Statistics** | Added VendorPartnerID and RegionCode as grouping columns |

---

## 📊 Competition Overview

| Item | Value |
|---|---|
| Train rows | 138,701 |
| Test rows | 15,000 |
| Target | Price of heavy equipment at auction |
| Metric | RMSLE (lower is better) |
| v14 Kaggle score | 0.18964 |
| **v15 Kaggle score** | **0.18847** |

---

## 🔧 CELL 1 — Default Kaggle Boilerplate (Skipped)

Standard Kaggle template — lists competition files. No changes needed.

**Output:**
```
/kaggle/input/competitions/heavy-equipment.../sample_submission.csv
/kaggle/input/competitions/heavy-equipment.../train.csv
...
```

---

## 📦 CELL 2 — Load All Libraries

**What:** Opens our toolbox of Python packages.

| Library | Purpose |
|---|---|
| numpy | Fast math on arrays |
| pandas | Table/dataframe operations |
| matplotlib | Charts and graphs |
| sklearn.KFold | Splits data for cross-validation |
| lightgbm | Our strongest gradient boosting model |
| CatBoostRegressor | Model that handles categories natively |
| SEED = 42 | Fixed seed for reproducibility |

**Key Change from v14:** XGBoost REMOVED. Replaced by a second LightGBM (LGB2, seed=123).

**Output:**
```
ALL LIBRARIES LOADED OK.
NOTE: XGBoost removed in v15 (was getting 0 percent blend weight).
```

---

## 🛠️ CELL 3 — Helper Functions

**`rmsle(y_true, y_pred)`** — The competition metric.
- Uses log(1+x) so that errors on cheap machines are weighted equally to errors on expensive machines.
- Lower RMSLE = better predictions.

**`make_submission(preds, ids, filename)`** — Saves predictions to CSV for Kaggle.

**Output:** `Helper functions ready.`

---

## 📂 CELL 4 — Load the Data

**What:** Reads competition CSV files into memory.

| Dataset | Rows | Columns | Notes |
|---|---|---|---|
| Train | 138,701 | 50 | Has TargetValue (price) |
| Test | 15,000 | 49 | No TargetValue - we predict this |

**Price range:** $7,500 – $142,000

**Output:**
```
TRAIN :  138,701 rows x 50 cols
TEST  :   15,000 rows x 49 cols
Price range: $7,500 - $142,000
```

---

## ⏳ CELL 5 — Leak-Free Asset History Features

**Story:** A bulldozer might be sold 3-4 times over its life. Its previous sale price helps predict the current price.

**What:** Combines train+test, sorts by TransactionID (time order), and computes 4 history features per machine:

| Feature | Meaning |
|---|---|
| Asset_Prev_Sales_Count | How many times was this machine sold before? |
| Asset_Prev_Mean_Price | Average price across all previous sales |
| Asset_Prev_Last_Price | Price at its most recent sale |
| Asset_Days_Since_Last_Sale | Days since it was last sold |

**Why "Leak-Free"?** Uses cumcount() and shift(1) to look ONLY at PAST sales, never the current one.

**Output:**
```
Asset history features added. Train: (138701, 54) | Test: (15000, 54)
```

---

## 🔨 CELL 6 — Base Feature Engineering (preprocess_df)

**What:** Transforms raw data into useful numbers for the models.

### Date Features
Splits TransactionDate into year, month, day, day-of-week, day-of-year, quarter, week, plus flags for month-end, year-end, etc.

### Data Cleaning
- ManufactureYear <= 1920 → NaN (bad data)
- OperationalHoursMeter == 0 → NaN (meter not recorded)

### Machine Age
`MachineAge = SaleYear - ManufactureYear` — older machines worth less.

### Ordinal Encoding
Text columns with natural order:
- UtilizationTier: Low=0, Medium=1, High=2
- AssetScaleFactor: Mini=0 → Large=5

### Hours Engineering
- LogHours = log(1 + hours) — smooths extreme values
- UsagePerYear = hours / (age + 1) — usage intensity
- HoursPresent = 1 if meter recorded, 0 if not

### Age Flags
- IsNewMachine (age <= 2), IsOldMachine (age >= 20), AgeDecade

### Text Cleaning
- "None or Unspecified" → NaN
- All text columns → category dtype

**Output:**
```
Train preprocessed: (138701, 71) | Test preprocessed: (15000, 71)
```

---

## 📐 CELL 7 — Prepare Feature Matrices

**What:** Separates data into inputs (X) and target (y).

- `y_all = log1p(price)` — log-scale makes prices easier for models to learn
- `X_all` = 68 training features (minus ID and price columns)
- `X_test_final` = 68 test features
- `cat_cols` = 40 categorical feature columns

**Output:**
```
Feature matrix : (138701, 68)
Test matrix    : (15000, 68)
Cat features   : 40
Num features   : 28
```

---

## ⭐ CELL 8 — Fold-Level Feature Engineering Functions

Three functions that run INSIDE the CV loop (critical for preventing data leakage).

### Function 1: add_oof_target_encoding

**Story:** A "Caterpillar 330C" excavator typically sells for $75,000. Telling the model this average helps it predict prices.

**What:** Replaces each category value with its average target price (Target Encoding).

**Anti-leakage trick:** Uses K-Fold OOF encoding:
- Training rows: TE computed from OTHER folds (never its own)
- Validation/Test rows: TE computed from full training fold
- Smoothing alpha=15.0 blends rare categories toward global mean

**v15 TE Columns (7 total — was 4 in v14):**

| Column | Unique Values | Why Added |
|---|---|---|
| Spec_FullDescriptor | 3,505 | Very high cardinality, strong signal |
| Spec_BaseClass | 1,249 | High cardinality |
| ProductConfigID | 3,594 | Very high cardinality |
| FunctionalClassification | 65 | Already in v14 |
| Spec_SubClass | 147 | **NEW! eta²=0.24 — strongest new signal** |
| Spec_VariantModifier | 132 | **NEW! eta²=0.22** |
| RegionCode | 53 | **NEW! 53 regions, each has pricing signal** |

> **eta²** measures how strongly a column explains price variance. Spec_SubClass with eta²=0.24 explains 24% of price variance!

### Function 2: add_interaction_target_encoding

**Story:** A "Wheel Loader" in California may be worth $50K, but the same machine in Alaska might be $65K. The combination of machine type + region is more informative than either alone.

**What:** Creates a combined feature (InventoryGroupDescription × RegionCode) and applies OOF Target Encoding.

**Why higher alpha=20?** Interaction cells have fewer observations, so more smoothing is needed.

### Function 3: add_fold_group_features

**What:** Computes group statistics using ONLY the training fold, then maps to validation/test.

1. **Frequency Encoding:** How common is this specific value in the training fold?
2. **Group Mean Features:** Average hours/age for each category group
3. **Relative Features:** This machine's hours / group average (is it more or less used than typical?)

**v15 adds VendorPartnerID and RegionCode as grouping columns** (on top of v14's InventoryGroupCategory, Spec_BaseClass, ProductConfigID).

**Output:**
```
All fold-level feature engineering functions compiled. Ready for CV.
Target encoding columns: 7 (v14 had 4)
```

---

## 🔄 CELL 9 — 5-Fold Cross Validation Loop (Main Training)

**What:** Trains all 3 models using 5-fold cross validation.

### The 3 Models and Their Key Parameters:

#### 🟢 LightGBM 1 (Primary, seed=42)

| Parameter | Value | Meaning |
|---|---|---|
| n_estimators | 15,000 | Max trees (early stopping stops earlier) |
| num_leaves | 511 | Max leaves per tree (double v14's 255!) |
| learning_rate | 0.010 | Slower than v14's 0.015 = more careful |
| subsample | 0.75 | Train each tree on 75% of rows |
| colsample_bytree | 0.70 | Use 70% of features per tree |
| min_child_samples | 30 | Min 30 samples per leaf (prevents overfitting) |
| reg_lambda | 3.0 | L2 regularization |
| early_stopping | 300 | Stop if no improvement for 300 rounds |

#### 🔵 LightGBM 2 (Diversity, seed=123)

Same architecture as LGB1 but with:
- Different random seed (123 vs 42)
- subsample=0.80 vs 0.75
- colsample_bytree=0.65 vs 0.70
- Different regularization (reg_alpha=0.2, reg_lambda=2.0)

**Why?** Different seeds + slightly different hyperparameters = different errors = better ensemble!

#### 🟡 CatBoost

| Parameter | Value | Meaning |
|---|---|---|
| iterations | 15,000 | Max trees |
| learning_rate | 0.010 | Same as LGB |
| l2_leaf_reg | 3.0 | L2 regularization |
| bagging_temperature | 0.5 | Randomness level |
| grow_policy | Lossguide | Leaf-first growth (like LightGBM) |
| max_leaves | 255 | Max leaves per tree |
| cat_features | cat_fold | Which columns are categorical |
| early_stopping_rounds | 300 | Stop if no improvement for 300 rounds |

CatBoost handles categorical features natively — no need for manual encoding.

### Fold-by-Fold Results:

| Fold | LGB1 (iter) | LGB2 (iter) | CAT |
|---|---|---|---|
| 1 | 0.19421 (4474) | 0.19399 (4391) | 0.20306 |
| 2 | 0.19660 (5257) | 0.19627 (5234) | 0.20537 |
| 3 | 0.19706 (4420) | 0.19687 (3925) | 0.20555 |
| 4 | 0.19686 (4657) | 0.19659 (4410) | 0.20670 |
| 5 | 0.19570 (4123) | 0.19553 (4240) | 0.20335 |

**Overall OOF Scores:**
```
LGB1 OOF RMSLE : 0.19609  (v14 LGB was 0.19733 — improved!)
LGB2 OOF RMSLE : 0.19585
CAT  OOF RMSLE : 0.20481
```

---

## ⚖️ CELL 10 — Finding the Best Blend Weights

**Story:** You have 3 expert appraisers. How much should you trust each one?

**What:** Tests all possible weight combinations (in 5% increments) and finds the blend with lowest OOF RMSLE.

**Optimal Weights Found:**
| Model | Weight |
|---|---|
| LGB1 | 30% |
| LGB2 | 40% |
| CAT | 30% |

**Blended OOF RMSLE:** 0.19415 (v14 was 0.19542 — improved!)

**Key Insight:** LGB2 gets the highest weight (40%)! Because LGB1 and LGB2 make different errors, their combination is more powerful than either alone.

**Output:**
```
Best blend    : LGB1=0.30  LGB2=0.40  CAT=0.30
Blended RMSLE : 0.19415  (v14 blended was 0.19542)
Kaggle v14 score : 0.18964  ->  [KEEP GOING] Still above 0.18964...
```

> The OOF score "KEEP GOING" warning is expected — OOF RMSLE is always higher than Kaggle score. We proceed to submit anyway.

---

## 📤 CELL 11 — Building the Final Submission

**What:**
1. Average 5-fold predictions for each model
2. Blend using optimal weights (LGB1=30%, LGB2=40%, CAT=30%)
3. Reverse log transformation: `np.expm1()` → actual dollar prices
4. Clip to training price range ($7,500 – $137,500)
5. Save as submission.csv

**Output:**
```
Price clipped to [$7,500, $137,500]
Saved submission.csv: 15,000 rows | min=$7,532 median=$34,638 max=$137,500

   TransactionID  TargetValue
0        1139307     69355.55
1        1139419     80544.98
...
```

---

## 📈 CELL 12 — Diagnostics and Feature Importance

**What:** Creates diagnostic charts and prints feature importance.

### How to Read the Charts:

**Chart 1 — Actual vs. Predicted Scatter:**
- X-axis: actual price (log scale), Y-axis: predicted price (log scale)
- Red diagonal = perfect prediction. Points close to line = good predictions.
- Want: tight cluster along the diagonal.

**Chart 2 — Residuals Distribution:**
- X-axis: error (actual - predicted), Y-axis: count
- Should be bell-shaped and centered at 0.
- Bias toward negative = model overpredicts. Bias toward positive = underpredicts.

**Chart 3 — Percentage Error Distribution:**
- Shows what % error we make (clipped ±40%)
- Tight bell around 0% = consistently accurate.

### Top Features (by LightGBM Importance):

| Rank | Feature | Importance |
|---|---|---|
| 1 | Spec_FullDescriptor_TE | Strongest predictor |
| 2 | Asset_Prev_Mean_Price | History of past prices |
| 3 | SaleDayOfYear | Seasonal pricing |
| 4 | MachineAge | Age of machine |
| ~7 | Spec_SubClass_TE | **NEW in v15! eta²=0.24** |
| ~8 | InventoryGroupDesc x RegionCode_TE | **NEW in v15! Interaction feature** |

---

## 🏆 Score Journey

| Version | Key Strategy | OOF RMSLE | Kaggle Score |
|---|---|---|---|
| v6 | Baseline Tuned Trees | 0.19627 | 0.19086 |
| v8 | Target Encoding (Leaked) | 0.18120 | 0.19996 ❌ |
| v10 | Leak-Free TE + Asset History | 0.19604 | 0.19020 |
| v12 | Global Freq + Group Stats (Leaked) | 0.19230 | 0.22521 ❌ |
| v13 | Fixed Asset History, Kept Freq (Leaked) | 0.19532 | 0.22379 ❌ |
| v14 | Fold-Safe Freq + Group Stats (Fixed) | 0.19542 | 0.18964 |
| **v15** | **Expanded TE (7 cols) + Interaction TE + Dual LGB** | **0.19415** | **0.18847** |

---

## 📚 Glossary

| Term | Simple Explanation |
|---|---|
| **RMSLE** | Root Mean Squared Log Error — competition metric (lower = better) |
| **OOF Predictions** | Out-of-Fold predictions — model predicts on data it never saw during training |
| **Target Encoding (TE)** | Replacing a category with its average price |
| **Interaction TE** | TE on a combination of two columns (e.g., machine type x region) |
| **Data Leakage** | When future/secret information sneaks into training features — causes inflated scores |
| **K-Fold CV** | Splits data into K parts and trains K times, each time using a different part for validation |
| **Early Stopping** | Stop adding trees when the model stops improving on validation data |
| **Ensemble/Blending** | Combining predictions from multiple models for better accuracy |
| **Smoothing (alpha)** | Blending rare category averages toward global mean to reduce overfitting |
| **num_leaves** | Max leaf nodes per tree — higher = more expressive = more complex patterns captured |
| **learning_rate** | How much each new tree corrects previous errors — smaller = more careful |
| **subsample** | Fraction of rows used per tree — prevents overfitting |
| **colsample_bytree** | Fraction of features used per tree — prevents over-reliance on any one feature |
| **reg_lambda (L2)** | Regularization that penalizes large weights — prevents overfitting |
| **eta²** | Measures how strongly a category column correlates with price |
| **Frequency Encoding** | Replacing a category with how often it appears in the training fold |
| **Gradient Boosting** | ML technique building trees sequentially, each fixing previous errors |
