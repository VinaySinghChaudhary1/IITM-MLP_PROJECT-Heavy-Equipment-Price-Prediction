# Milestone 4

The following questions are part of Milestone 4 of the Machine Learning Practice Project course. Perform the tasks as instructed in the questions.

**Note:** The milestone is considered submitted only when you receive the response copy to your email ID. If there are any issues, please reach out to the instructor before the deadline.

**Deadline:** 14 July 2026

---

### Dataset Columns Used Across These Questions
Use the following columns and all the rows from the training dataset for the questions of this milestone unless specified otherwise:
- **TargetValue** — sale price (target)
- **UtilizationTier** — Low / Medium / High usage (categorical, has missing values)
- **DrivetrainType, CabinType, AssetScaleFactor** — categorical machine specs
- **ManufactureYear, OperationalHoursMeter** — numerical machine specs
- **Spec_FullDescriptor, RegionCode, FunctionalClassification** — extra descriptive columns (not required, but usable for extensions)

---

### Q1 — Stratified vs. Non-Stratified Split
Since `TargetValue` is a continuous price (not a class label), we first turn it into 5 price bands (quantile bins) so we can "stratify" the split, just like you would stratify on a class column.

#### Steps:
**A)** Fill missing values in `UtilizationTier` with the string `"Unknown"`. Create a new column `PriceBand` by cutting `TargetValue` into 5 quantile-based bins using `pd.qcut(..., q=5, labels=False)`. Split the data using `train_test_split` into 70% train / 30% validation, with `random_state=42`, stratifying on `PriceBand`. Compute the value counts of `PriceBand` in the train and validation sets (sorted by bin index ascending), and store them as NumPy arrays.

**B)** Repeat the exact same split (`test_size=0.3`, `random_state=42`) but with `stratify=None`. Again compute the sorted validation `PriceBand` counts as a NumPy array.

**C)** Convert both validation count arrays (from A and B) into proportions (divide by total validation size). For each bin, compute the absolute difference between the two proportions, and report the maximum absolute difference (rounded to 4 decimal places).

**Your Answer:** `____________________________________`

---

### Q2 — Preprocessing Pipeline
Use the stratified split (`X_train`, `X_valid`, `y_train`, `y_valid`) from Q1 for the rest of the notebook. All preprocessing objects must be fit only on `X_train` and used to transform both `X_train` and `X_valid`.

#### Steps:
- **Categorical columns:** `["UtilizationTier", "DrivetrainType", "CabinType", "AssetScaleFactor"]` — process with `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`.
- **Numerical columns:** `["ManufactureYear", "OperationalHoursMeter"]` — process with `SimpleImputer(strategy="mean")` then `StandardScaler()`.

Combine both using a `ColumnTransformer` with `remainder="drop"`. Fit on `X_train`, transform both `X_train` and `X_valid`. Store results as `X_train_proc`, `X_valid_proc`.

Report the sum of all values in `X_train_proc` (rounded to 3 decimal places).

**Your Answer:** `____________________________________`

---

### Q3 — Hyperparameter Tuning with RandomizedSearchCV
Use `X_train_proc` and `y_train` from Q2.

Tune a `RandomForestRegressor(random_state=42)` using `RandomizedSearchCV` over:
- `n_estimators`: `[50, 100, 200]`
- `max_depth`: `[5, 10, 15]`

**Configuration:** `n_iter=5`, `cv=3`, `random_state=42`, `n_jobs=-1`, default scoring.

Fit on `(X_train_proc, y_train)`, then report the best `n_estimators` found.

**Your Answer:** `____________________________________`

---

### Q4 — AdaBoost Estimator Error Variance
Train an `AdaBoostRegressor(n_estimators=50, random_state=42)` (all other parameters default) on `(X_train_proc, y_train)`.

After fitting, access `model.estimator_errors_` (length-50 array of each weak learner's error).

Compute `np.var(model.estimator_errors_)` and report it rounded to 4 decimal places.

**Your Answer:** `____________________________________`

---

### Q5 — Feature Importance
Train a `RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)` on `(X_train_proc, y_train)`.

After fitting, extract `feature_importances_` and report the index of the feature with the maximum importance. If there is a tie, return the smallest index.

**Your Answer:** `____________________________________`

---

### Q6 — Counting MLP Weights
Let $N =$ `X_train_proc.shape[1]` (number of input features after preprocessing).

An `MLPRegressor` with `hidden_layer_sizes=(128, 64, 32)` has this architecture:
- **Input layer:** $N$ neurons
- **Hidden Layer 1:** 128 neurons
- **Hidden Layer 2:** 64 neurons
- **Hidden Layer 3:** 32 neurons
- **Output layer:** 1 neuron (regression has a single output)

Compute the total number of connection weights (excluding all biases) using your actual value of $N$.

**Your Answer:** `____________________________________`

---

### Q7 — Training Loss After 5 Iterations
Train the same MLP architecture as Q6 (`hidden_layer_sizes=(128, 64, 32)`, `activation="relu"`) with: `solver="adam"`, `max_iter=5`, `batch_size=32`, `random_state=42` (all other params default).

Fit on `(X_train_proc, y_train)`. After training, access `model.loss_` and report it rounded to 4 decimal places.

**Your Answer:** `____________________________________`

---

### Q8 — Effect of Regularization (alpha)
Train two `MLPRegressor` models on `(X_train_proc, y_train)` with `hidden_layer_sizes=(100,)`, `max_iter=5`, `random_state=42` (defaults otherwise):
- **Model A:** `alpha=0.0001` (default — light regularization)
- **Model B:** `alpha=1.0` (strong regularization)

Compute the Mean Absolute Error (MAE) of each model's predictions on the training set. Report the absolute difference between the two MAE values, rounded to 4 decimal places.

**Your Answer:** `____________________________________`

---

### Q9 — Validation Performance
Using the MLP model trained in Q7, generate predictions on `X_valid_proc` and compare them to `y_valid`. Compute the Mean Absolute Error (MAE) on the validation set, rounded to 4 decimal places.

**Your Answer:** `____________________________________`
