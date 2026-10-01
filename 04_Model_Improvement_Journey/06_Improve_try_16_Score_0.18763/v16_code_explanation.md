# 📖 v16 Code Explanation — Easy Story Guide
## `22f2001153improve_16.ipynb`
### Heavy Equipment Price Prediction | Score Journey: 0.18964 (v14) → 0.18847 (v15) → **0.18763 (v16)**

---

> **How to read this guide:** Imagine you are trying to predict the price of a used bulldozer or excavator at an auction. You have data about thousands of past auctions. Your goal is to build a "smart calculator" that uses this historical data to predict prices for new auctions. Every section below explains one part of the notebook step by step in simple language.

---

## ⚠️ IMPORTANT: How to Submit on Kaggle

1. Go to Kaggle → Competition → "Code" → **"+ New Notebook"**
2. Click **"File" → "Import Notebook"** → Upload `22f2001153improve_16.ipynb`
3. Click **"Add Data"** → Add competition dataset
4. Click **"Run All"** (takes ~4–5 hours on Kaggle)
5. Submit the generated `submission.csv`

> ⚠️ Always upload as a **fresh new notebook**. Never copy-paste code into an old notebook.

---

## 🚀 What Changed from v15 to v16?

v15 achieved Kaggle score **0.18847**. v16 improves upon it with 5 data-driven improvements based on v15 feature importance analysis:

| # | Improvement | Why It Helps |
|---|---|---|
| 1 | **Cyclical Date Encoding** (SaleDayOfYear was #1 feature in v15!) | sin/cos tells model that Dec 31 and Jan 1 are adjacent (raw integers cannot do this) |
| 2 | **Two New Interaction TEs** (FC x Region, SubClass x Region) | v15 showed InventoryGroupDesc x Region jumped to #8 in importance — more interactions = more power |
| 3 | **Non-Linear Age Features** (MachineAge_sq, LogAge) | Physical depreciation is NOT linear — value drops faster in early years |
| 4 | **Asset History Flag** (Has_Asset_History) | Only 14.4% of rows have prior history — model needs to explicitly know this |
| 5 | **FunctionalClassification in Group Stats** | eta²=0.352 — the STRONGEST unused grouping variable in v15 |

---

## 📊 Competition Overview

| Item | Value |
|---|---|
| Train rows | 138,701 |
| Test rows | 15,000 |
| Target | Price of heavy equipment at auction |
| Metric | RMSLE (lower is better) |
| v15 Kaggle score | 0.18847 |
| **v16 Kaggle score** | **0.18763** |

---

## 🔧 CELL 1 — Default Kaggle Boilerplate (Skipped)

Standard Kaggle template cell — lists competition files. No changes needed.

---

## 📦 CELL 2 — Load All Libraries

**What:** Opens our toolbox of Python packages.

| Library | Purpose |
|---|---|
| numpy | Fast math on arrays |
| pandas | Table/dataframe operations |
| matplotlib | Charts and graphs |
| sklearn.KFold | Splits data for cross-validation |
| lightgbm | Our primary gradient boosting model |
| CatBoostRegressor | Model that handles categories natively |
| SEED = 42 | Fixed seed for reproducibility |

**Output:**
```
ALL LIBRARIES LOADED OK.
NOTE: XGBoost removed in v15 (was getting 0 percent blend weight).
```

---

## 🛠️ CELL 3 — Helper Functions

**`rmsle(y_true, y_pred)`** — Competition metric. Lower = better.

**`make_submission(preds, ids, filename)`** — Saves predictions to CSV.

**Output:** `Helper functions ready.`

---

## 📂 CELL 4 — Load the Data

**What:** Reads competition CSV files.

| Dataset | Rows | Columns |
|---|---|---|
| Train | 138,701 | 50 |
| Test | 15,000 | 49 |

**Price range:** $7,500 – $142,000

**Output:**
```
TRAIN :  138,701 rows x 50 cols
TEST  :   15,000 rows x 49 cols
Price range: $7,500 - $142,000
```

---

## ⏳ CELL 5 — Leak-Free Asset History Features

**Story:** A bulldozer might be sold multiple times. Its previous sale price helps predict the current price.

**What:** Combines train+test, sorts by TransactionID (time order), and computes history features.

**v16 adds one new feature: `Has_Asset_History`**

| Feature | Meaning |
|---|---|
| Asset_Prev_Sales_Count | How many times was this machine sold before? |
| **Has_Asset_History** | **NEW! Binary: 1 if machine has ANY prior sales, 0 if first sale** |
| Asset_Prev_Mean_Price | Average price across all previous sales |
| Asset_Prev_Last_Price | Price at its most recent sale |
| Asset_Days_Since_Last_Sale | Days since it was last sold |

**Why `Has_Asset_History`?**
Only 14.4% of training rows have prior history. The model needs to explicitly know whether it has useful history features or whether they are all NaN (missing). A binary flag captures this "missing history" pattern directly.

**Output:**
```
Building leak-free asset history features...
Asset history features added. Train: (138701, 55) | Test: (15000, 55)
Rows with prior history: 19,969 / 138,701 (14.4%)
```

---

## 🔨 CELL 6 — Base Feature Engineering (preprocess_df)

**What:** Transforms raw data into useful numbers for the models.

v16 adds several new features to this step:

### Date Features (same as v15)
Splits TransactionDate into year, month, day, day-of-week, day-of-year, quarter, week, plus flags for month-end, year-end, etc.

### 🆕 NEW v16: Cyclical Date Encoding

```python
df['SaleDayOfYear_sin'] = np.sin(2 * np.pi * df['SaleDayOfYear'] / 365)
df['SaleDayOfYear_cos'] = np.cos(2 * np.pi * df['SaleDayOfYear'] / 365)
df['SaleMonth_sin']     = np.sin(2 * np.pi * df['SaleMonth'] / 12)
df['SaleMonth_cos']     = np.cos(2 * np.pi * df['SaleMonth'] / 12)
df['SaleDayOfWeek_sin'] = np.sin(2 * np.pi * df['SaleDayOfWeek'] / 7)
df['SaleDayOfWeek_cos'] = np.cos(2 * np.pi * df['SaleDayOfWeek'] / 7)
```

**Why cyclical encoding?**

SaleDayOfYear was the #1 most important feature in v15! But there's a problem with raw integers:

- Day 365 (Dec 31) = 365
- Day 1 (Jan 1) = 1

The difference is 364, but these days are actually adjacent (only 1 day apart in real life)! A tree model sees them as far apart and splits incorrectly.

**Solution: sin/cos encoding wraps the year into a circle:**
- sin/cos of Day 365 ≈ sin/cos of Day 1 (close values!)
- The model can now correctly understand that winter prices are similar across years

This same logic applies to `SaleMonth` (December close to January) and `SaleDayOfWeek` (Sunday close to Monday).

### Data Cleaning (same as v15)
- ManufactureYear <= 1920 → NaN
- OperationalHoursMeter == 0 → NaN

### Machine Age + 🆕 Non-Linear Age Features

```python
df['MachineAge']    = (df['SaleYear'] - df['ManufactureYear']).clip(lower=0)
df['MachineAge_sq'] = (df['MachineAge'] ** 2)    # NEW in v16!
df['LogAge']        = np.log1p(df['MachineAge'])  # NEW in v16!
```

**Why non-linear?**
Physical depreciation is NOT linear:
- A machine drops more from year 1→2 than from year 10→11
- `MachineAge_sq` captures the accelerating early depreciation
- `LogAge` smooths out the depreciation curve for older machines

Example: A 2-year-old machine vs a 1-year-old machine: MachineAge difference = 1, MachineAge_sq difference = 3 (4 vs 1). The model learns that early depreciation is more significant.

### Ordinal Encoding (same as v15)
- UtilizationTier: Low=0, Medium=1, High=2
- AssetScaleFactor: Mini=0 → Large=5

### Hours Engineering (same as v15)
- LogHours, UsagePerYear, HoursPresent, IsNewMachine, IsOldMachine, AgeDecade

### Text Cleaning + Dropping Noisy Columns
- "None or Unspecified" → NaN, all text → category dtype
- Dropped: col18, col19, UtilizationTier, AssetScaleFactor

**Output:**
```
Preprocessing features (no leaky group features)...
Train preprocessed: (138701, 80) | Test preprocessed: (15000, 80)
```
(80 columns = more than v15's 71, due to new cyclical + age features)

---

## 📐 CELL 7 — Prepare Feature Matrices

**What:** Separates data into inputs (X) and target (y).

- `y_all = log1p(price)` — log-scale prices
- `X_all` = 77 training features
- `X_test_final` = 77 test features
- `cat_cols` = 40 categorical features

**Output:**
```
Feature matrix : (138701, 77)
Test matrix    : (15000, 77)
Cat features   : 40
Num features   : 37
```
(37 numerical features vs v15's 28 — 9 new features added!)

---

## ⭐ CELL 8 — Fold-Level Feature Engineering Functions

Three functions that run INSIDE the CV loop (to prevent data leakage).

### Function 1: add_oof_target_encoding (same as v15)

**What:** Replaces categories with their average target price (OOF Target Encoding).
- 7 columns (same as v15): Spec_FullDescriptor, Spec_BaseClass, ProductConfigID, FunctionalClassification, Spec_SubClass, Spec_VariantModifier, RegionCode
- alpha=15.0 smoothing

---

### Function 2: add_interaction_target_encoding (expanded in v16!)

**What:** Creates combined features (column A × column B) and applies OOF Target Encoding.

**v16 now has 3 interaction TEs (v15 had 1):**

| Interaction | Why |
|---|---|
| InventoryGroupDescription × RegionCode | **Carried from v15** — proved to be #8 most important feature! |
| **FunctionalClassification × RegionCode** | **NEW! eta²=0.352 × strong region signal** |
| **Spec_SubClass × RegionCode** | **NEW! eta²=0.24 × strong region signal** |

**Why higher alpha=22?** Interaction cells are sparser than single columns, so more smoothing is needed to prevent overfitting rare combinations.

**Example:** "Excavator (SubClass) in Mountain Region" might consistently sell for $10K more than average — this pattern is captured by the SubClass × Region interaction.

---

### Function 3: add_fold_group_features (expanded in v16!)

**What:** Computes group statistics using ONLY training fold, maps to validation/test.

1. **Frequency Encoding** (same 4 columns as v15)
2. **Group Mean + Relative Features:**

v16 adds **FunctionalClassification** as a grouping column:

| Grouping Column | v15 | v16 | Why Added |
|---|---|---|---|
| InventoryGroupCategory | ✅ | ✅ | — |
| Spec_BaseClass | ✅ | ✅ | — |
| ProductConfigID | ✅ | ✅ | — |
| VendorPartnerID | ✅ | ✅ | Added in v15 |
| RegionCode | ✅ | ✅ | Added in v15 |
| **FunctionalClassification** | ❌ | **✅** | **NEW! eta²=0.352 — strongest unused grouper** |

For each grouping column × each target feature (OperationalHoursMeter, MachineAge):
- `{group}_{target}_mean`: What is the typical hours/age for this group?
- `{target}_rel_{group}`: How does this machine compare to its group average?

**Output:**
```
All fold-level feature engineering functions compiled. Ready for CV.
TE columns        : 7
Interaction TEs   : 3 (InventoryGroupDesc x Region, FC x Region, SubClass x Region)
Group stat cols   : 6 (added FunctionalClassification in v16)
```

---

## 🔄 CELL 9 — 5-Fold Cross Validation Loop (Main Training)

**What:** Trains all 3 models using 5-fold cross validation.

### Inside each fold:
1. OOF Target Encoding (7 columns)
2. Interaction TE for 3 combinations (InventoryGroupDesc×Region, FC×Region, SubClass×Region)
3. Frequency + Group Statistics
4. Train LGB1, LGB2, CatBoost

### The 3 Models:

#### 🟢 LightGBM 1 (Primary, seed=42)

| Parameter | Value | Meaning |
|---|---|---|
| n_estimators | 15,000 | Max trees |
| num_leaves | 511 | Max leaves per tree |
| learning_rate | 0.010 | Careful, slow learning |
| subsample | 0.75 | 75% of rows per tree |
| colsample_bytree | 0.70 | 70% of features per tree |
| min_child_samples | 30 | Min samples per leaf |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 3.0 | L2 regularization |
| early_stopping | 300 | Stop if no improvement |

#### 🔵 LightGBM 2 (Diversity, seed=123)

Same architecture, different hyperparameters:
- subsample=0.80 (vs 0.75)
- colsample_bytree=0.65 (vs 0.70)
- reg_alpha=0.2, reg_lambda=2.0 (different regularization)
- Different random seed → different error patterns → better ensemble!

#### 🟡 CatBoost

| Parameter | Value | Meaning |
|---|---|---|
| iterations | 15,000 | Max trees |
| learning_rate | 0.010 | Careful learning |
| l2_leaf_reg | 3.0 | L2 regularization |
| bagging_temperature | 0.5 | Randomness level |
| grow_policy | Lossguide | Leaf-first growth |
| max_leaves | 255 | Max leaves per tree |
| cat_features | cat_fold | Which columns are categorical |
| early_stopping_rounds | 300 | Stop if no improvement |

CatBoost handles categorical features natively without manual encoding.

### Fold-by-Fold Results:

| Fold | LGB1 (iter) | LGB2 (iter) | CAT |
|---|---|---|---|
| 1 | 0.19333 (5467) | 0.19308 (4637) | 0.20246 |
| 2 | 0.19578 (5304) | 0.19580 (5205) | 0.20567 |
| 3 | 0.19647 (4576) | 0.19631 (4376) | 0.20472 |
| 4 | 0.19630 (4933) | 0.19601 (4472) | 0.20582 |
| 5 | 0.19509 (3922) | 0.19481 (4389) | 0.20319 |

**Overall OOF Scores:**

| Model | v16 OOF | v15 OOF | Improvement |
|---|---|---|---|
| LGB1 | **0.19540** | 0.19609 | ✅ Better |
| LGB2 | **0.19521** | 0.19585 | ✅ Better |
| CAT | **0.20438** | 0.20481 | ✅ Better |

**Output:**
```
LGB1 OOF RMSLE : 0.19540  (v15 was 0.19609, target < 0.19609)
LGB2 OOF RMSLE : 0.19521  (v15 was 0.19585)
CAT  OOF RMSLE : 0.20438  (v15 was 0.20481)
```

---

## ⚖️ CELL 10 — Finding the Best Blend Weights

**Story:** 3 expert appraisers. How much to trust each one?

**What:** Tests all weight combinations (5% increments), finds lowest blended RMSLE.

**Optimal Weights Found:**

| Model | Weight |
|---|---|
| LGB1 | 30% |
| LGB2 | 45% |
| CAT | 25% |

**Blended RMSLE:** 0.19359 (v15 was 0.19415 — improved!)

**Key Insight:** LGB2's weight increased from 40% (v15) to 45% (v16)! This suggests LGB2's diverse predictions are even more complementary to LGB1 in v16.

**Output:**
```
Best blend    : LGB1=0.30  LGB2=0.45  CAT=0.25
Blended RMSLE : 0.19359  (v15 blended was 0.19415)
Kaggle v15 score : 0.18847  ->  [KEEP GOING] Still above 0.18847...
```

> The "KEEP GOING" warning is expected — OOF RMSLE is always higher than Kaggle score. We still proceed to submit.

---

## 📤 CELL 11 — Building the Final Submission

**What:**
1. Average 5-fold predictions for each model
2. Blend using optimal weights (LGB1=30%, LGB2=45%, CAT=25%)
3. Reverse log transformation: `np.expm1()` → actual dollar prices
4. Clip to training price range ($7,500 – $137,500)
5. Save as submission.csv

**Output:**
```
Price clipped to [$7,500, $137,500]
Saved submission.csv: 15,000 rows | min=$7,763 median=$34,616 max=$137,500

   TransactionID  TargetValue
0        1139307     69602.14
1        1139419     80375.45
2        1139482     31860.57
3        1139522     20327.60
4        1139684     12909.15
```

---

## 📈 CELL 12 — Diagnostics and Feature Importance

**What:** Creates diagnostic charts and prints feature importance rankings.

### How to Read the Charts:

**Chart 1 — Actual vs. Predicted Scatter:**
- X-axis: actual price (log scale), Y-axis: predicted price (log scale)
- Red diagonal = perfect prediction
- Points close to line = accurate. Points far = large errors.
- Want: tight cluster along the diagonal, no fan-shaped spread.

**Chart 2 — Residuals Distribution:**
- X-axis: error (actual - predicted), Y-axis: count
- Should be bell-shaped, centered at 0.
- Bias toward negative = model overpredicts. Positive bias = underpredicts.

**Chart 3 — Percentage Error Distribution:**
- Shows % error made (clipped ±40%)
- Tight bell around 0% = consistently accurate predictions.

### Top Features (by LightGBM Importance):

v16 expected top features (based on v15 analysis + new features):

| Rank | Feature | Notes |
|---|---|---|
| 1 | Spec_FullDescriptor_TE | Strongest predictor |
| 2 | SaleDayOfYear_sin / _cos | **NEW! Cyclical encoding now more informative** |
| 3 | Asset_Prev_Mean_Price | History of past prices |
| 4 | MachineAge | Machine age |
| ~5 | FunctionalClassification_TE | eta²=0.352 — very strong |
| ~7 | FC x RegionCode_TE | **NEW! Interaction TE |
| ~8 | SubClass x RegionCode_TE | **NEW! Interaction TE** |

> **Key insight:** Cyclical date features likely rank higher than raw `SaleDayOfYear` from v15 because they correctly capture seasonal patterns at year boundaries.

---

## 🏆 Score Journey

| Version | Key Strategy | OOF RMSLE | Kaggle Score |
|---|---|---|---|
| v6 | Baseline Tuned Trees | 0.19627 | 0.19086 |
| v10 | Leak-Free TE + Asset History | 0.19604 | 0.19020 |
| v14 | Fold-Safe Freq + Group Stats | 0.19542 | 0.18964 |
| v15 | Expanded TE (7) + Interaction TE + Dual LGB | 0.19415 | 0.18847 |
| **v16** | **Cyclical Dates + 3 Interaction TEs + Non-Linear Age + FunctionalClassification Groups** | **0.19359** | **0.18763** |

---

## 💡 Key Lessons from v14 → v15 → v16

1. **Always check feature importance.** v15's top feature was `SaleDayOfYear` → v16 improved it with cyclical encoding.
2. **Interaction features are powerful.** `InventoryGroupDesc × Region` proved so useful in v15 that we added 2 more interactions in v16.
3. **Non-linear transformations matter.** Machine depreciation is exponential, not linear — `MachineAge_sq` and `LogAge` capture this.
4. **Binary flags for missingness.** Only 14.4% of rows have asset history — `Has_Asset_History` tells the model to handle these differently.
5. **Feature importance guides improvements.** Every improvement in v16 came directly from analyzing what v15 found most useful.

---

## 📚 Glossary

| Term | Simple Explanation |
|---|---|
| **RMSLE** | Root Mean Squared Log Error — competition metric (lower = better) |
| **OOF Predictions** | Out-of-Fold predictions — model predicts on data it never saw during training |
| **Target Encoding (TE)** | Replacing a category with its average price |
| **Interaction TE** | TE on a combination of two columns (e.g., machine type x region) |
| **Cyclical Encoding** | sin/cos transformation that wraps periodic features into a circle |
| **Data Leakage** | When future/secret information sneaks into training features — causes inflated scores |
| **K-Fold CV** | Splits data into K parts and trains K times, each time using a different part for validation |
| **Early Stopping** | Stop adding trees when the model stops improving on validation data |
| **Ensemble/Blending** | Combining predictions from multiple models for better accuracy |
| **Smoothing (alpha)** | Blending rare category averages toward global mean to reduce overfitting |
| **num_leaves** | Max leaf nodes per tree — higher = more expressive |
| **learning_rate** | How much each new tree corrects previous errors — smaller = more careful |
| **subsample** | Fraction of rows used per tree — prevents overfitting |
| **colsample_bytree** | Fraction of features used per tree |
| **reg_lambda (L2)** | Regularization that penalizes large weights — prevents overfitting |
| **eta²** | Measures how strongly a category column correlates with price |
| **Frequency Encoding** | Replacing a category with how often it appears in training |
| **Gradient Boosting** | ML technique building trees sequentially, each fixing previous errors |
| **MachineAge_sq** | Machine age squared — captures accelerating early depreciation |
| **LogAge** | log(1 + age) — smoothed age for older machines |
| **Has_Asset_History** | Binary flag: does this machine have any previous sales history? |
