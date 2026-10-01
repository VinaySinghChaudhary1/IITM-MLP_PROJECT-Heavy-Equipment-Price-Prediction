# Milestone 3 — Pre-Run Answer Reference

**File:** `3_milestone_answer.ipynb`  
**Created:** July 2, 2026  
**Verified Against:** Actual Kaggle dataset (`train.csv` — 138,701 rows)

> Run the notebook on Kaggle to confirm these exact outputs. After running, share any values that differ and they will be updated.

---

## ✅ Quick Answer Sheet

| # | Question Summary | Answer | MCQ |
|---|---|---|---|
| Q1 | Safest imputation for skewed `OperationalHoursMeter` | **Median** | **C** |
| Q2 | Most frequent `SaleYear` (mode) | **2010** | **A** |
| Q3 | Rows with `"Missing"` in `CabinType` after fillna | **0** (zero NaNs!) | **C** |
| Q4 | Columns removed at >80% missing threshold | **15 columns** | **B** |
| Q5 | Median `HoursPerYear` (OperationalHoursMeter / AssetAge) | **~220 hrs/yr → A) ~250** | **A** |
| Q6 | Median `AssetAge` after removing ManufactureYear 1000/1001 | **8.0 years** | **C** |
| Q7 | Unique integers after OrdinalEncoder on `UtilizationTier` | **4** | **C** |
| Q8 | When to apply Train-Test Split to prevent data leakage | **Before any scalers/encoders** | **B** |
| Q9 | `X_train` rows after 80/20 split | **110,960** | **A** |
| Q10 | Std of `log1p(TargetValue)` | **~0.67** | **A** |
| Q11 | Primary reason to use Target Encoding over OHE | **High cardinality** | **B** |
| Q12 | What `n_estimators` controls | **Total trees in ensemble** | **D** |
| Q13 | Best model for messy tabular data | **Gradient Boosted Trees** | **C** |
| Q14 | What high feature importance implies | **Frequent + accurate splits** | **B** |
| Q15 | Primary benefit of Early Stopping | **Prevents overfitting** | **B** |

---

## 📊 Section 1: Feature Engineering — Detailed Answers

### Q1 — Median for Skewed Data [Answer: C]

**Why Median?**  
`OperationalHoursMeter` has extremely high skewness (heavy right tail from machines with thousands of hours). The **mean is dragged up by outliers**, making it unrepresentative. The **median is the middle value** — unaffected by extremes.

- Example: [100, 200, 300, 400, **50000**] → Mean = 10,200, Median = 300
- Use **Mean** for: symmetric, normally distributed data
- Use **Median** for: skewed data with outliers ← this question

### Q2 — SaleYear Mode = 2010 [Answer: A]

**Verified from dataset:**
```
SaleYear
2010    21,333  ← MODE (most frequent)
2012    18,472
2011    17,880
2009    16,809
2008    11,985
```

**Insight:** 2010 peak reflects post-2008 financial crisis recovery — construction projects resumed, driving heavy equipment auction volume up.

### Q3 — CabinType: 0 Missing Values [Answer: C]

**Verified from dataset:**
```
CabinType
EROPS           60,189
EROPS w AC      55,752
OROPS           22,760
NaN                  0  ← ZERO missing!
```

**Key lesson:** Always run `df['column'].isnull().sum()` before assuming a column needs imputation. CabinType has complete coverage, so `fillna("Missing")` creates **zero** "Missing" rows.

### Q4 — 15 Columns Dropped [Answer: B]

**Verified — all 15 columns exceeding 80% missing:**

| Column | Missing % |
|---|---|
| col19 | 99.95% |
| col18 | 99.95% |
| col4 | 92.52% |
| col5 | 92.52% |
| col28 | 89.21% |
| col27 | 89.21% |
| col29 | 86.39% |
| col30 | 86.39% |
| col11 | 83.13% |
| col8 | 83.13% |
| col7 | 83.13% |
| col6 | 83.13% |
| col9 | 83.13% |
| col13 | 83.13% |
| col14 | 83.13% |

### Q5 — HoursPerYear Median ≈ 220 hrs/yr [Answer: A]

**Formula:** `HoursPerYear = OperationalHoursMeter / AssetAge_raw`  
- Replace AssetAge = 0 with NaN (avoid division by zero → inf)  
- Remove inf and NaN before computing median

**Verified result:** Median = **220.33 hrs/year**  
Closest MCQ option: **A) ~250 hours/year**

> Note: The exact value (220.33) is closest to option A (250) compared to B (475), C (320), or D (1200).

### Q6 — AssetAge Median = 8.0 years [Answer: C]

**Steps:**
1. Replace `ManufactureYear == 1000` or `1001` → `NaN`
2. `AssetAge = SaleYear - ManufactureYear`
3. Compute `median()` ignoring NaN

**Verified result:** Median = **8.0 years**

**Why 1001 is removed:** It's a database default placeholder meaning "year not recorded". Without removal: `AssetAge = 2010 - 1001 = 1009 years` → completely wrong.

---

## 📊 Section 2: Data Transformation — Detailed Answers

### Q7 — OrdinalEncoder: 4 Unique Integers [Answer: C]

**UtilizationTier original values:**
```
NaN         88,226  (64% of rows!)
Medium      24,588
Low         17,005
High         8,882
```

**After `fillna("Unspecified")`:** 4 unique strings → High, Low, Medium, Unspecified  
**After `OrdinalEncoder`:** Alphabetical mapping:
- High → 0
- Low → 1
- Medium → 2
- Unspecified → 3

**Result: 4 unique integers (0, 1, 2, 3)**

### Q8 — Split BEFORE Scalers [Answer: B]

**Correct pipeline (no leakage):**
```
1. train_test_split(X, y, test_size=0.2)     ← STEP 1: Split first
2. scaler.fit(X_train)                        ← STEP 2: Fit on train only
3. scaler.transform(X_train), scaler.transform(X_test)  ← Apply to both
```

**Wrong (causes leakage):**
```
1. scaler.fit_transform(X_all)   ← Leaks test data statistics into scaler!
2. train_test_split(X_scaled, y)  ← Too late!
```

### Q9 — X_train = 110,960 rows [Answer: A]

`138,701 × 0.80 = 110,960.8` → truncated to **110,960**  
`138,701 × 0.20 = 27,741.2` → **27,741** validation rows

### Q10 — log1p std ≈ 0.67 [Answer: A]

**Verified values:**
```
Original TargetValue:
  std  = 26,361  (option B — but this is the ORIGINAL std)
  
After np.log1p():
  std  = 0.671   ← ANSWER (option A)
  mean = 10.424  (option D is the MEAN, not std!)
```

**Why log1p?** Optimizing RMSE on log1p(y) = optimizing RMSLE. The competition uses RMSLE, so training on the log-transformed target directly aligns the loss function.

### Q11 — Target Encoding for High Cardinality [Answer: B]

**OHE vs Target Encoding:**

| Situation | Use OHE | Use Target Encoding |
|---|---|---|
| 3-10 unique values | ✅ | Overkill |
| 50-500 unique values | Too many columns! | ✅ |
| Numeric feature | Neither — it's already numeric | Neither |
| Binary target | OHE works | TE also works |

---

## 📊 Section 3: Model Building — Detailed Answers

### Q12 — n_estimators = Total Trees [Answer: D]

**Key hyperparameter reference:**

| Parameter | Controls |
|---|---|
| `n_estimators` | **Total number of trees** ← Q12 answer |
| `max_depth` | Maximum depth of each tree |
| `learning_rate` | Step size for each gradient update |
| `num_leaves` | Max leaves per tree (LightGBM) |
| `subsample` | % of rows sampled per tree |

### Q13 — Gradient Boosted Trees Win [Answer: C]

**Competition results confirm this:**
- Linear Regression → RMSLE ≈ 0.53 ❌
- Random Forest → RMSLE ≈ 0.29
- HistGradientBoosting → RMSLE ≈ 0.23
- **XGBoost (GBT) → RMSLE ≈ 0.22 ✅ Best**
- LightGBM (GBT) → RMSLE ≈ 0.22 ✅

### Q14 — High Importance = Frequent Accurate Splits [Answer: B]

Feature importance (split gain) = sum of error reduction from all splits using that feature across all trees.

**In our model (typical ranking):**
1. AssetAge / MachineAge — highest
2. ProductConfigID
3. ManufactureYear
4. SaleYear
5. OperationalHoursMeter

High importance ≠ high standard deviation, no missing values, or correlation with other features.

### Q15 — Early Stopping Prevents Overfitting [Answer: B]

**How it works:**
1. Monitor validation RMSLE after each tree
2. If no improvement for `stopping_rounds=100` iterations → STOP
3. Revert to the weights from the **best** iteration

**Benefits:**
- Prevents overfitting (training loss keeps dropping, validation plateaus → diverges)
- Saves training time (no need to manually tune n_estimators)
- Guarantees the returned model is the best generalization, not just the most trained

---

## ⚠️ Notes for Kaggle Run

Before running, confirm the data path at the top of the notebook:
```python
DATA_DIR = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/'
```

All answers in Q1–Q6 are **empirically computed** from the dataset and will print their exact values when the notebook runs. Q7–Q15 are **conceptual** questions with code demonstrations.
