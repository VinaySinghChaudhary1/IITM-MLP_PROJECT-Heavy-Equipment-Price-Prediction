# 📖 Understanding v14 Notebook — Line by Line (Story Style)
### Heavy Equipment Price Prediction | 22f2001153
### Score Journey: 0.20845 → 0.19969 → 0.19086 (v6 Best) → 0.19996 (v8 Leakage) → 0.19020 (v10) → 0.22521 (v12 Leakage) → 0.22379 (v13 Leakage) → **v14 (Leak-Fixed): 0.18964 ✅**

---

## ❌ Why v12 and v13 (Scores: 0.22521 & 0.22379) Failed

In the v12 and v13 implementations, we added **Frequency Encoding** (`col_Freq`) and **Group Averages** (`CategoryMeanHours`, etc.) to preprocess categorical dimensions. While these features showed local CV improvements, their implementation contained a subtle but devastating bug: **Train-Test Scale Mismatch and Target Leakage**.

When we calculated frequency encoding and group statistics in `preprocess_df()`, we did so on the entire dataframe passed to it.
- **Scale Mismatch**: The training set contains **138,701 rows**, whereas the test set contains only **15,000 rows**. When computing frequency counts of categories, values in the training set were roughly **9 times larger** than their values in the test set. 
- For instance, if a specific machine type appeared 900 times in training and 100 times in testing, the model learned during training that `ProductConfigID_Freq = 900` corresponds to high-value equipment. However, when evaluating on test data, the frequency for the exact same machine type was mapped as `100`. The model saw `100` and predicted a completely incorrect price range.
- **Target Leakage**: The group means calculated in preprocessing allowed validation data to influence training statistics, leaking the target values and giving an overly optimistic local validation score.

---

## 🔧 How v14 Fixes This

To fix this distribution mismatch and leakage, v14 enforces the **K-Fold Fold-Level Mapping Principle**:

1. **❌ Clean Preprocessing**: We stripped all frequency encodings, group means, and relative statistics out of `preprocess_df()`. This function now only performs simple column cleaning (e.g. date splitting, age calculation, missing value correction).
2. **✅ Loop-Level Feature Engineering**: We move all advanced feature engineering inside the cross-validation loop.
   - For each of the 5 folds, we compute frequency mappings and group-level means **using only the 80% training partition** (the training fold).
   - We then map these statistics onto the validation fold (20%) and the test set (15,000 rows).
   - This ensures the frequency counts and average prices are computed on the **same absolute scale**, preventing any mismatch.
3. **✅ String Cast Fix**: Since grouping column `InventoryGroupCategory` is set to a pandas `category` dtype, mapping a float mean and filling NaNs directly causes a `TypeError` (as pandas categories are rigid). We convert the categorical columns to strings (`.astype(str)`) before performing groupby and map, resolving the crash.

---

## 🚀 HOW TO SUBMIT ON KAGGLE

1. Go to Kaggle → Your Competition → **Code** tab → Click **+ New Notebook**
2. Click **File → Import Notebook** and upload `22f2001153_Improved_v14.ipynb` (from `D:\Download\AI MLP Project\`).
3. Ensure the competition dataset is attached.
4. Click **Run All** (runs in ~45 minutes).
5. Submit `submission.csv`.

---

## 📦 CELL-BY-CELL EXPLANATION

### 🔧 CELL 1 — Title and compliance check (Markdown)
Explains the structure of the notebook, rules (using only pandas, numpy, scikit-learn, lightgbm, xgboost, catboost), and why v12/v13 failed.

### 🔧 CELL 2 — Load Libraries (Code)
We import our suite of standard machine learning tools:
- `KFold` for splitting data.
- `lightgbm`, `xgboost`, `catboost` regressors.
- `mean_squared_error` for validation.
- Configures pandas column limits and floats for clean table viewing.

### 🔧 CELL 3 — Path Detection & Helpers (Code)
- `rmsle`: calculates the competition metric (Root Mean Squared Logarithmic Error).
- `make_submission`: builds the submission CSV.
- Path check: Checks if we are running in Kaggle cloud (`/kaggle/input`) or locally, setting file paths accordingly.

### 📊 CELL 4 — Loading the Data (Code)
Reads `train.csv` (138,701 rows) and `test.csv` (15,000 rows) into memory.

### ⏳ CELL 5 — Leak-Free Asset History Features (Code)
- Groups transactions by `AssetID` (individual machine).
- Sorts by `TransactionID` to maintain strict chronological order.
- Calculates `Asset_Prev_Mean_Price` and `Asset_Prev_Last_Price` using prior sales. 
- Uses `shift(1)` so a transaction's price is never leaked to itself.

### 🔨 CELL 6 — Base Feature Engineering (Code)
Extracts simple, stateless features:
- Decomposes sale dates into Year, Month, Day, Weekday, etc.
- Computes `MachineAge = SaleYear - ManufactureYear`.
- Cleans placeholder values (zeros in hours and manufacture years <= 1920 become `NaN`).
- Ordinal maps `UtilizationTier` and `AssetScaleFactor`.

### 📐 CELL 7 — Prepare Feature Matrices (Code)
- Separates target variable `y_all = log1p(TargetValue)` from predictor features.
- Drops identifiers `TransactionID` and `AssetID` so the model doesn't overfit to specific machines or transaction ranks.

### ⭐ CELL 8 — Fold-Safe Feature Engineering Functions (Code)
Defines two functions run inside the training loop:
1. `add_oof_target_encoding`: performs K-Fold target encoding on high-cardinality features.
2. `add_fold_group_features`: computes category-level frequencies and mean hours/age. Crucially, these are computed only on the training fold, and mapped to val/test to prevent distribution mismatch.

### 🔄 CELL 9 — 5-Fold Cross Validation Loop (Code)
Splits training data into 5 folds.
- LightGBM, XGBoost, and CatBoost models are trained on 80% of data and validated on 20% in each fold.
- Out-of-fold (OOF) predictions are generated.
- Average Out-Of-Fold Scores:
  - **LightGBM**: `0.19733`
  - **XGBoost**: `0.20325`
  - **CatBoost**: `0.20619`

### ⚖️ CELL 10 — Finding Blended Weights (Code)
Runs grid search on OOF arrays. Finds that a combination of **70% LightGBM + 30% CatBoost (0% XGBoost)** minimizes the blended OOF RMSLE down to **0.19542** (an improvement over v10's 0.19604).

### 📤 CELL 11 — Building Final Submission (Code)
Combines the test predictions from the 5 folds using the optimal ensemble weights, takes `expm1` to convert from log-scale to dollar values, clips predictions to training percentiles ([$7,500, $137,500]) and saves `submission.csv`.

### 📈 CELL 12 — Diagnostics and Feature Importance (Code)
Plots validation scatter plots, residuals, error distributions, and outputs the top 25 features. `Spec_FullDescriptor_TE` and `Asset_Prev_Mean_Price` rank highest.

---

## 🏆 Complete Score Journey

| Version | Key Strategy | Local OOF RMSLE | Kaggle Score |
|---|---|---|---|
| v6 | Baseline Tuned Trees | 0.19627 | 0.19086 |
| v8 | Naive Target Encoding | 0.18120 (Leaked) | 0.19996 |
| **v10** | Leak-Free Target Encoding + Asset History | 0.19604 | **0.19020** |
| v12 | Global Frequency + Group Averages | 0.19230 (Leaked) | 0.22521 |
| v13 | Fixed Asset History, Kept Global Freq | 0.19532 (Leaked) | 0.22379 |
| **v14** | **Fold-Safe Frequency + Group Stats (Fixed)** | **0.19542** | **0.18964 ✅** |

---

## 📚 Glossary of ML Terms

| Term | Simple Explanation |
|---|---|
| **Data Leakage** | When information from outside the training dataset (such as validation/test labels) is used to create features, causing overoptimistic training results. |
| **OOF Target Encoding** | Replacing a text category with the average target value of other folds to represent price levels without leakage. |
| **Scale Mismatch** | When feature values have different magnitudes/meanings in train vs test (e.g. counting raw frequencies in different size datasets). |
| **RMSLE** | Root Mean Squared Logarithmic Error. The scoring metric of the competition. |
