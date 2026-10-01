# 📖 v18 Code Explanation — Easy Story Guide
## `22f2001153@improve_18.ipynb`
### Heavy Equipment Price Prediction | Score Journey: 0.18757 (v17) → **0.18732 (v18)**

---

> **How to read this guide:** Imagine you are trying to predict the price of a used bulldozer or excavator at an auction. You have data about thousands of past auctions. Every section below explains one part of the notebook step by step, in simple language.

---

## 🚀 What Changed from v17 to v18?

v17 achieved Kaggle score **0.18757**. v18 improves upon it with **4 data-driven changes** guided by v17/v18 feature importance analysis:

| # | Improvement | Why It Helps |
|---|---|---|
| 1 | **AgeBucket categorical feature** (5 bins: New/Young/Mid/Old/Ancient) | A structured age grouping that allows dense, reliable target encoding combinations with SaleYear |
| 2 | **4 Dense Time-based Interaction TEs** (AgeBucket×SaleYear, InventoryGroupCategory×SaleYear, RegionCode×SaleYear, Spec_SubClass×SaleYear) | **Density-checked** interactions with avg>100 samples/cell — reliable TE estimates, not noise |
| 3 | **Increased LGB capacity**: num_leaves 511 → 1023 | More expressive trees to capture the richer feature set from 6+ interaction TEs |
| 4 | **Density-first interaction design philosophy** | Every new interaction TE is checked for cell density BEFORE adding — sparse TEs (< 30/cell) are excluded to avoid noise |

> **Key innovation in v18:** The concept of **density-checking** interaction TEs. In v17, interactions like ProductConfigID×SaleYear (16,316 cells!) were added without checking how sparse they are. Many cells had fewer than 10 examples — meaning the TE estimate was basically just noise + global mean. v18 only adds interactions where the average cell size > 100 observations.

---

## 📊 Competition Overview

| Item | Value |
|---|---|
| Train rows | 138,701 |
| Test rows | 15,000 |
| Target | Price of heavy equipment at auction |
| Metric | RMSLE (lower is better) |
| v17 Kaggle score | 0.18757 |
| **v18 Kaggle score** | **0.18732** |

---

## ⚠️ IMPORTANT: How to Submit on Kaggle

1. Go to Kaggle → Competition → "Code" → **"+ New Notebook"**
2. Click **"File" → "Import Notebook"** → Upload `22f2001153@improve_18.ipynb`
3. Click **"Add Data"** → Add competition dataset
4. Click **"Run All"** (takes ~4–5 hours on Kaggle)
5. Submit the generated `submission.csv`

---

## 🔧 CELL 1 — Default Kaggle Boilerplate (Skipped)

Standard Kaggle template cell — lists competition files.

---

## 📦 CELL 2 — Load All Libraries

**What:** Opens our toolbox. Same libraries as v17.

**Expected output:**
```
ALL LIBRARIES LOADED OK.
NOTE: XGBoost removed in v15 (was getting 0 percent blend weight).
```

---

## 🛠️ CELL 3 — Helper Functions

**What:** Defines `rmsle()` and `make_submission()`. Same as all previous versions.

**Expected output:** `Helper functions ready.`

---

## 📂 CELL 4 — Load Competition Data

**What:** Reads train.csv and test.csv.

**Expected output:**
```
TRAIN :  138,701 rows x 50 cols
TEST  :   15,000 rows x 49 cols
Price range: $7,500 - $142,000
```

---

## ⏳ CELL 5 — Leak-Free Asset History Features

**What:** Computes machine sale history features. Same 5 features as v16/v17.

**Expected output:**
```
Building leak-free asset history features...
Asset history features added. Train: (138701, 55) | Test: (15000, 55)
Rows with prior history: 19,969 / 138,701 (14.4%)
```

---

## 🔨 CELL 6 — Base Feature Engineering (`preprocess_df`)

**What:** Transforms raw data into useful features. v18 carries all v17 features AND adds one critical new feature.

### All v17 features carried forward:
- SaleDayOfYear_sin/cos, SaleMonth_sin/cos, SaleDayOfWeek_sin/cos (from v16)
- **SaleDay_sin/cos** (from v17)
- MachineAge, MachineAge_sq, LogAge (from v16)
- LogHours_x_LogAge (from v17)
- Years_Since_2000 (from v17)
- Hours clipping at 99th percentile (from v17)
- Has_Asset_History binary flag (from v16)

---

### 🆕 NEW in v18: AgeBucket Categorical Feature

```python
age_bins   = [-1, 2, 5, 10, 20, 200]
age_labels = ['New', 'Young', 'Mid', 'Old', 'Ancient']
df['AgeBucket'] = pd.cut(df['MachineAge'], bins=age_bins, labels=age_labels)
df['AgeBucket'] = df['AgeBucket'].astype(str).fillna('Unknown').astype('category')
```

**Why?** This creates a new categorical column that places each machine into one of 5 age groups:

| Bucket | MachineAge | What it means |
|---|---|---|
| New | 0–2 years | Nearly new — minimal depreciation |
| Young | 3–5 years | Light depreciation, good condition |
| Mid | 6–10 years | Moderate depreciation, normal wear |
| Old | 11–20 years | Heavy depreciation, significant wear |
| Ancient | 20+ years | Very old, likely low value |

**Why is AgeBucket better than MachineAge alone?**
`AgeBucket` has only 5 unique values. When crossed with `SaleYear` (14 unique values), we get 5×14 = 70 combinations — an average of **1,981 samples per cell** (= 138,701 / 70). This is extremely dense, meaning the target encoding for each (age_group, year) combination is very reliable!

Compare to ProductConfigID×SaleYear in v17: ~16,000 combinations → ~8 samples per cell on average → very noisy!

**Expected output:**
```
Preprocessing features...
Train: (138701, 85) | Test: (15000, 85)
```
(85 columns = v17's 84 + AgeBucket)

---

## 📐 CELL 7 — Prepare Feature Matrices

**What:** Separates data into inputs (X) and target (y).

**v18 has 41 categorical features** (vs v17's 40) — AgeBucket is categorical!

**Expected output:**
```
Feature matrix : (138701, 82)
Test matrix    : (15000, 82)
Cat features   : 41
```

---

## ⭐ CELL 8 — Fold-Level Feature Engineering Functions

**What:** Functions that run INSIDE the CV loop.

### The Density-Check Philosophy (NEW in v18)

Before adding any interaction TE, v18 checks how many samples each cell has on average:

```python
# Density check formula:
avg_samples_per_cell = len(training_data) / (unique_values_A * unique_values_B)
```

**Decision rules:**
- avg > 100 samples/cell → **SAFE** (alpha=10-15, reliable TE)
- avg 30-100/cell → **OK** (alpha=15-25, with more smoothing)
- avg < 30/cell → **DO NOT USE** (noise only — TE collapses to global mean anyway)

**v17's problem:** ProductConfigID×SaleYear had ~16,316 combinations → avg = 138,701/16,316 ≈ **8 samples/cell**. Most cells had 0-2 samples. The TE was basically random noise!

---

### Function 2: Interaction TEs in v18

**Interactions from v16 (Region-based, unchanged):**
- InventoryGroupDescription × RegionCode (alpha=20)
- FunctionalClassification × RegionCode (alpha=22)
- Spec_SubClass × RegionCode (alpha=22)

**NEW Time-based interactions in v18 (density-checked!):**

| Interaction | Combinations | Avg/cell | Alpha | Status |
|---|---|---|---|---|
| **AgeBucket × SaleYear** | 5 × 14 = 70 | **1,981/cell** | 10 | ✅ VERY SAFE |
| **InventoryGroupCategory × SaleYear** | ~80 × 14 = 1,120 | **124/cell** | 12 | ✅ SAFE |
| **RegionCode × SaleYear** | ~53 × 14 = 742 | **187/cell** | 12 | ✅ SAFE |
| **Spec_SubClass × SaleYear** | ~147 × 14 = 2,058 | **~67/cell** | 15 | ✅ OK |

Notice: ProductConfigID×SaleYear, Spec_BaseClass×SaleYear from v17 are **DROPPED** in v18 because they are too sparse!

**Expected output:**
```
Feature engineering functions compiled.
Density-checked interaction TEs for v18:
  SAFE (avg>100/cell): AgeBucket x SaleYear, InventoryGroupCategory x SaleYear, RegionCode x SaleYear
  OK   (avg~112/cell): Spec_SubClass x SaleYear
```

---

## 🔄 CELL 9 — 5-Fold Cross Validation Loop

**What:** Trains 3 models using 5-fold CV. v18 makes one key model change: **LGB1 num_leaves increased to 1023**.

### Model Changes vs v17:

| Model | v17 num_leaves | v18 num_leaves | Other |
|---|---|---|---|
| LGB1 | 511 | **1023** | ← Increased capacity! |
| LGB2 | 511 | 511 | Unchanged |
| CatBoost | 255 | 255 | Unchanged |

**Why increase LGB1's num_leaves to 1023?**
With 4 additional dense interaction TEs and the new AgeBucket feature, LGB1 needs more capacity to capture the richer feature interactions. `num_leaves=1023` allows trees with up to 1,023 leaf nodes — twice as expressive as 511!

**⚠️ Warning:** Larger num_leaves = longer training time per fold. Each fold may take ~20-30 minutes more.

### Inside each fold, the steps are:
1. OOF TE for 7 single columns
2. Region interactions: InventoryGroupDesc×Region, FC×Region, SubClass×Region (from v16)
3. Time interactions: AgeBucket×Year, InvGrpCat×Year, Region×Year, SubClass×Year (NEW in v18)
4. Frequency + Group Statistics

### Fold-by-Fold Results:

| Fold | LGB1 (iter) | LGB2 (iter) | CAT |
|---|---|---|---|
| 1 | 0.19265 (3992) | 0.19234 (4107) | 0.20089 |
| 2 | 0.19562 (3815) | 0.19534 (4145) | 0.20391 |
| 3 | 0.19594 (2950) | 0.19579 (3485) | 0.20455 |
| 4 | 0.19630 (3410) | 0.19616 (3324) | 0.20458 |
| 5 | 0.19492 (2748) | 0.19488 (3121) | 0.20219 |

**OOF Results vs v17:**

| Model | v17 OOF | v18 OOF | Change |
|---|---|---|---|
| LGB1 | 0.19542 | **0.19509** | ✅ Improved! |
| LGB2 | 0.19527 | **0.19491** | ✅ Improved! |
| CAT | 0.20332 | **0.20323** | ✅ Improved! |

All three models improved — the density-checked TEs + AgeBucket + larger LGB1 capacity all contributed.

**Expected output:**
```
Starting 5-Fold CV - v18 (Dense Interactions + More Capacity)
======================================================================
Fold 1: LGB1=0.19265(iter=3992) | LGB2=0.19234(iter=4107) | CAT=0.20089
...
LGB1 OOF RMSLE : 0.19509  (v17 was 0.19542, need improvement)
LGB2 OOF RMSLE : 0.19491  (v17 was 0.19527)
CAT  OOF RMSLE : 0.20323  (v17 was 0.20332)
```

---

## ⚖️ CELL 10 — Finding the Best Blend Weights

**What:** Grid search over weight combinations to minimize blended RMSLE.

**Optimal Weights:**
| Model | v17 Weight | v18 Weight |
|---|---|---|
| LGB1 | 30% | **25%** |
| LGB2 | 40% | **45%** |
| CAT | 30% | **30%** |

LGB2 gets the highest weight (45%) — it benefits from the dense interactions in a slightly different way than LGB1 (different seed, different row/feature sampling).

**Blended RMSLE:** 0.19326 (v17 was 0.19359 — **improved by 0.00033**!)

**Expected output:**
```
Best blend    : LGB1=0.25  LGB2=0.45  CAT=0.30
Blended RMSLE : 0.19326  (v17 blended was 0.19359)
Kaggle v17 score : 0.18757  ->  [BEAT v17!]
```

> **"[BEAT v17!]"** — The OOF blended RMSLE (0.19326) beat v17's blended RMSLE (0.19359), and this is confirmed by the actual Kaggle score improvement (0.18732 vs 0.18757).

---

## 📤 CELL 11 — Building the Final Submission

**What:** Blends predictions and saves `submission.csv`.

- Weights: LGB1=25%, LGB2=45%, CAT=30%
- Reverse log: `np.expm1()`
- Clip: [$7,500, $137,500]

**Expected output:**
```
Price clipped to [$7,500, $137,500]
Saved submission.csv: 15,000 rows | min=$7,728 median=$34,615 max=$137,500

   TransactionID  TargetValue
0        1139307     69296.52
1        1139419     81322.36
2        1139482     30853.24
3        1139522     20357.37
4        1139684     12913.14
```

---

## 📈 CELL 12 — Diagnostics & Feature Importance

**What:** 3 diagnostic charts + top-40 feature importance from LGB1 last fold.

**How to read the 3 charts:**
- **Chart 1 (Actual vs Predicted):** Points should cluster tightly on the red diagonal. Fan shape = more errors at extremes.
- **Chart 2 (Residuals):** Bell curve centered at 0 = unbiased predictions.
- **Chart 3 (% Error):** Tight bell around 0% = consistently accurate.

**Actual OOF metrics from v18 output:**
| Metric | v18 Value |
|---|---|
| Blended OOF RMSLE | 0.19326 |
| R² | 0.9161 |
| MAE | $5,042 |
| Median abs % error | 9.7% |

**Top features from actual v18 output:**

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | RegionCode__x__SaleYear_TE | 116,945 | NEW dense time interaction |
| 2 | MachineAge_rel_ProductConfigID | 93,290 | Group relative |
| 3 | InventoryGroupCategory__x__SaleYear_TE | 92,179 | NEW dense time interaction |
| 4 | RegionCode | 90,712 | Raw feature |
| 5 | AgeBucket__x__SaleYear_TE | 87,030 | NEW AgeBucket interaction |
| 6 | SaleDayOfYear_cos | 83,645 | Cyclical date |
| 7 | SaleDayOfYear_sin | 82,577 | Cyclical date |
| 8 | ProductConfigID | 78,914 | Raw feature |
| 9 | SaleDayOfYear | 78,433 | Raw date |
| 10 | Spec_SubClass__x__SaleYear_TE | 76,306 | NEW dense time interaction |

> **Key validation:** The NEW dense time interactions rank #1, #3, #5, #10 in the top 10! The density-checked approach clearly produces high-quality features. `RegionCode×SaleYear` is the strongest single feature — 12% more important than anything else.

---

## 🏆 Score Journey

| Version | Key Strategy | OOF RMSLE | Kaggle Score |
|---|---|---|---|
| v1 | Baseline XGBoost | — | ~0.20845 |
| v6 | Tuned Trees | 0.19627 | 0.19086 |
| v10 | Leak-Free TE + Asset History | 0.19604 | 0.19020 |
| v14 | Fold-Safe Freq + Group Stats | 0.19542 | 0.18964 |
| v15 | Expanded TE (7 cols) + Interaction TE + Dual LGB | 0.19415 | 0.18847 |
| v16 | Cyclical Dates + 3 Interaction TEs + Non-Linear Age | 0.19359 | 0.18763 |
| v17 | SaleDay Cyclical + LogHours×LogAge + 3 Time Interaction TEs | 0.19359 | 0.18757 |
| **v18** | **AgeBucket + 4 Dense Interaction TEs + LGB1 num_leaves=1023** | **0.19326** | **0.18732** |

---

## 💡 Key Lessons from v17 → v18

1. **Density matters for interaction TEs.** v17 added high-eta² interactions (ProductConfigID×SaleYear) that were too sparse to be reliable. v18 replaced them with lower-eta² but DENSE interactions that are actually more useful.
2. **AgeBucket is a smarter way to cross with SaleYear** than raw MachineAge. 70 cells × 1,981 samples each = very reliable estimates.
3. **Model capacity must grow with feature richness.** More interaction TEs → need more num_leaves to capture them.
4. **Feature importance guides improvements.** The top-40 feature list from v17/v18 directly told us which interactions were most useful.

---

## 📚 Glossary

| Term | Simple Explanation |
|---|---|
| **RMSLE** | Root Mean Squared Log Error — competition metric (lower = better) |
| **OOF** | Out-of-Fold — predictions made when that row was NOT in training |
| **Density Check** | Verifying an interaction TE has enough samples per cell to be reliable |
| **AgeBucket** | Categorical age grouping (New/Young/Mid/Old/Ancient) — allows dense TEs |
| **Cell density** | Average number of training samples in each (column_A, column_B) combination |
| **Dense TE** | Interaction TE with >100 samples/cell — reliable, low-noise estimate |
| **Sparse TE** | Interaction TE with <30 samples/cell — noisy, collapses to global mean |
| **num_leaves=1023** | Max 1,023 leaf nodes per tree — twice as expressive as 511 |
| **Cyclical Encoding** | sin/cos transformation that wraps periodic features into a circle |
| **Interaction TE** | Target encoding on a combination of two columns (e.g., age group × year) |
| **eta²** | Measures how strongly a category explains price variance |
| **Smoothing alpha** | Blends rare category estimates toward global mean — higher = more shrinkage |
| **Gradient Boosting** | ML technique building trees sequentially, each fixing previous errors |
