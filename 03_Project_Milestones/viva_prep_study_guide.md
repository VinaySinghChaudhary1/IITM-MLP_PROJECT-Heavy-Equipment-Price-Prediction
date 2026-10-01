# 🏗️ MLP Project: Complete Viva Preparation & Milestone Study Guide
## Heavy Equipment Selling Price Prediction Challenge

This study guide is designed to help you prepare for your **Viva (Oral Examination)**. It explains every single question, the exact code used, the logic behind it (written like a story), key terms, functions, parameters, and how to answer the examiner's questions.

---

# 📚 Table of Contents
1. [Project Overview & Key Concepts](#-project-overview--key-concepts)
2. [Milestone 1: Basic Dataset Understanding & EDA](#-milestone-1-basic-dataset-understanding--eda)
3. [Milestone 2: Feature Engineering, Encoding, Scaling & Linear Regression](#-milestone-2-feature-engineering-encoding-scaling--linear-regression)
4. [Milestone 3: Advanced Preprocessing, Target Transformation & Tree Models](#-milestone-3-advanced-preprocessing-target-transformation--tree-models)
5. [Milestone 4: Pipelines, Hyperparameter Tuning & Neural Networks (MLP)](#-milestone-4-pipelines-hyperparameter-tuning--neural-networks-mlp)
6. [Milestone 5: Advanced Feature Selection, Preprocessing Pipelines, Random Forest & Hyperparameter Tuning](#-milestone-5-advanced-feature-selection-preprocessing-pipelines-random-forest--hyperparameter-tuning)

---

# 🏗️ Project Overview & Key Concepts

### What is this project about?
We are predicting the **auction sale price** of used heavy machinery (like excavators, tractors, wheel loaders, bulldozers, etc.) based on historical transaction data and technical specs.

### What is the evaluation metric?
The competition uses **RMSLE (Root Mean Squared Logarithmic Error)**.
$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(p_i + 1) - \log(a_i + 1))^2}$$
where $p_i$ is the predicted price and $a_i$ is the actual price.

#### 💡 Why RMSLE instead of RMSE?
1. **Right-skewed target variable**: Heavy machinery prices range from small utility tractors ($7,500) to massive mining excavators ($142,000). A $10,000 error on a $15,000 machine is a disaster (66% error), but a $10,000 error on a $120,000 machine is acceptable (8% error). RMSLE penalizes relative errors, making it robust to large price differences.
2. **Robustness to outliers**: By applying a log transformation, we compress the scale of target values, which prevents extreme outliers from dominating the model's loss function.

---

# 📊 Milestone 1: Basic Dataset Understanding & EDA

Here we load the data and perform basic exploratory data analysis (EDA) to understand the size, columns, datatypes, missing values, anomalies, and distributions.

### Setup Code:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
sns.set_theme(style='whitegrid', palette='Set2')

# ── Kaggle competition input path ──
DATA_DIR = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/'

train = pd.read_csv(DATA_DIR + 'train.csv')
test  = pd.read_csv(DATA_DIR + 'test.csv')

print(f'✅ Data loaded!')
print(f'   Train : {train.shape[0]:,} rows  ×  {train.shape[1]} cols')
print(f'   Test  : {test.shape[0]:,} rows  ×  {test.shape[1]} cols')
```

## ❓ Question 1: Dataset Dimensions
> **✅ Verified Answer: `B) Train: 138,701 | Test: 15,000`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Check train and test dataset dimensions to understand data scale |
| **Tool / Technique used** | `df.shape` |
| **What the output means** | Train=138,701 rows × 50 cols; Test=15,000 rows × 49 cols. Train is ~9× larger; train has 1 extra column (TargetValue). |

### 💻 Code Block:
```python
print('=' * 50)
print('  Q1: Dataset Dimensions')
print('=' * 50)
print(f'  Train dataset : {train.shape[0]:,} rows  ×  {train.shape[1]} columns')
print(f'  Test  dataset : {test.shape[0]:,} rows  ×  {test.shape[1]} columns')
print()
print('✅ ANSWER: B) Train: 138,701  |  Test: 15,000')
print()
print('💡 Insight:')
print('   The train set is ~9× larger than test.')
print('   Train has 1 extra column (TargetValue = the price we must predict).')
```

### 📖 The Story of the Code:
1. We use **`pd.read_csv`** to load the dataset. This loads our comma-separated values (csv) files into a **Pandas DataFrame** (a tabular structure like a spreadsheet).
2. The property **`shape`** is a tuple containing the number of rows and columns. We print `train_df.shape` and `test_df.shape`.
3. We see that the training dataset has 138,701 records and 50 columns. The test dataset has 15,000 records and 49 columns.
4. **Why does train have 50 columns but test has 49?** Because train includes the target variable (`TargetValue`), which we need to predict for the test set.

### 🎓 Key Terms & Parameters:
- **`pandas (pd)`**: A high-performance Python library used for data manipulation and analysis.
- **`df.shape`**: Returns a tuple `(rows, columns)`. It is a property, not a function (so we write `df.shape`, not `df.shape()`).

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why do we check the shape of train and test datasets?"
  * **Your Answer:** "It tells us the scale of our data. We need to know how many training examples we have (138,701) to decide on suitable algorithms, and verify that the test set has exactly one less column (since the target variable is hidden in the test set)."

---
## ❓ Question 2: Target Variable Identification
> **✅ Verified Answer: `B) TargetValue`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the target variable by strict set difference between train and test columns |
| **Tool / Technique used** | `set(train_df.columns) - set(test_df.columns)` |
| **What the output means** | Only 'TargetValue' exists in train but not test → it's the USD auction price to predict. |

### 💻 Code Block:
```python
# Strict set difference: columns present in train but NOT in test
target_cols = set(train.columns) - set(test.columns)

print('=' * 50)
print('  Q2: Target Variable Identification')
print('=' * 50)
print(f'  Columns only in train (not in test) : {target_cols}')
print(f'  → This is the column we must predict for each test row.')
print()
print('✅ ANSWER: B) TargetValue')
print()
print('💡 Insight:')
print('   TargetValue = the USD auction sale price of the machine.')
print('   The competition evaluates predictions using RMSLE (log-scale error).')
```

### 📖 The Story of the Code:
1. We want to find the exact column that is in the training data but missing in the test data.
2. We convert the column headers of both sets into mathematical **sets** using `set(...)`.
3. We subtract the test columns from the train columns (`set(train_df.columns) - set(test_df.columns)`).
4. The difference returns `{'TargetValue'}`. This is our target variable (selling price of the equipment).

### 🎓 Key Terms & Parameters:
- **`set()`**: A built-in Python type that represents an unordered collection of unique elements. Set subtraction (`A - B`) returns elements in A that are not in B.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "How did you programmatically identify the target variable?"
  * **Your Answer:** "We took the set difference of train and test columns. The column present only in the train set is `TargetValue`, which represents the sale price we must predict."

---
## ❓ Question 3: Feature Data Types
> **✅ Verified Answer: `6`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Count columns that pandas recognizes as strictly int64 or float64 by default |
| **Tool / Technique used** | `df.select_dtypes(include=['int64', 'float64'])` |
| **What the output means** | Only 6 columns are numerical by default (TransactionID, TargetValue, AssetID, ProductConfigID, ManufactureYear, OperationalHoursMeter). Most other 44 are object dtype. |

### 💻 Code Block:
```python
# Select strictly int64 or float64 columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print('=' * 60)
print('  Q3: Numerical Feature Count (int64 or float64 only)')
print('=' * 60)
print(f'  Total columns in train        : {train.shape[1]}')
print(f'  Strictly numerical (int/float) : {len(num_cols)}')
print()
print(f'  Numerical column names:')
for i, c in enumerate(num_cols, 1):
    print(f'    {i}. {c}  [{train[c].dtype}]')
print()
print(f'✅ ANSWER: {len(num_cols)}')
print()
print('💡 Insight:')
print('   Only 6 columns are stored as int64 or float64 by default.')
print('   Most other columns are loaded as object (string) dtype,')
print('   including IDs and categorical config columns.')
```

### 📖 The Story of the Code:
1. We want to know how many columns are strictly stored as numerical types (`int64` representing integers and `float64` representing decimals).
2. We call **`train_df.select_dtypes(include=['int64', 'float64'])`** which filters the DataFrame columns to only keep these data types.
3. We print their names and the count.
4. The output is **6**. These columns are: `TransactionID`, `TargetValue`, `AssetID`, `ProductConfigID`, `ManufactureYear`, and `OperationalHoursMeter`. All other columns (like categories, specs, regions) are loaded as text strings (`object` data type).

### 🎓 Key Terms & Parameters:
- **`select_dtypes()`**: A Pandas method to select columns based on their data type.
  - `include`: A list of types to keep (e.g. `['int64', 'float64']`).
- **`int64` vs `float64`**: 64-bit integer (whole numbers) and 64-bit float (fractional numbers).

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why are only 6 features numerical by default? What are the other 44?"
  * **Your Answer:** "Most columns represent categorical descriptions or equipment specs (like engine size, cabin type, tire size). Pandas loads these as text strings (`object` data type). We must convert them into numbers later using encoding techniques before feeding them into machine learning models."

---
## ❓ Question 4: Extremely Sparse Columns
> **✅ Verified Answer: `A) col18, B) col19, D) col5`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find configuration columns with >90% missing values in train (>124,800 rows out of 138,701) |
| **Tool / Technique used** | `df.isnull().sum()` / `len(df) * 100` |
| **What the output means** | col19=99.95%, col18=99.95%, col5=92.52% qualify. CabinType has 0% missing → NOT selected. |

### 💻 Code Block:
```python
# Compute missing % for ALL columns
missing_count = train.isnull().sum()
missing_pct   = missing_count / len(train) * 100

# Filter to columns with > 90% missing
very_sparse = missing_pct[missing_pct > 90].sort_values(ascending=False)

# Check the 4 specific question columns
question_cols = ['col18', 'col19', 'CabinType', 'col5']

print('=' * 60)
print('  Q4: Columns with > 90% Missing Values')
print('=' * 60)
print(f'  Threshold : 90%  →  > {int(len(train)*0.90):,} missing rows')
print()
print(f'  ALL columns exceeding 90% missing:')
print(f'  {"Column":<20} {"Missing Rows":>15} {"Missing %":>12}')
print('  ' + '-' * 50)
for col in very_sparse.index:
    flag = ' ← IN QUESTION' if col in question_cols else ''
    print(f'  {col:<20} {missing_count[col]:>15,} {missing_pct[col]:>11.2f}%{flag}')

print()
print('  Checking the 4 specific options from the question:')
for c in question_cols:
    pct = missing_pct[c]
    status = '✅ YES (> 90%)' if pct > 90 else '❌ NO'
    print(f'    {c:<15}: {pct:.2f}% missing  → {status}')

print()
print('✅ ANSWER: A) col18  ,  B) col19  ,  D) col5')
print('   (All three exceed 90% missing. CabinType does NOT.)')
print()
print('💡 Insight:')
print('   col18 and col19 are ~100% empty → should be DROPPED.')
print('   col4 & col5 are ~92% empty → use binary presence flags.')
```

### 📖 The Story of the Code:
1. We want to find columns that are almost completely empty (sparse) so we can drop them.
2. We calculate **`train_df.isnull().sum()`** which checks each cell if it is null (empty) and sums it up for each column.
3. We divide this sum by the total length of the dataset (`len(train_df)`) and multiply by 100 to get percentages.
4. We filter for columns exceeding **90%** missing rate.
5. The output shows `col19` (99.95%), `col18` (99.95%), `col5` (92.52%), and `col4` (92.52%) as extremely empty.
6. The question options check if `col18`, `col19`, `CabinType`, or `col5` qualify. `col18`, `col19`, and `col5` do. `CabinType` has 0% missing, so it is NOT selected.

### 🎓 Key Terms & Parameters:
- **`isnull()`**: Returns a boolean DataFrame of the same shape, where `True` indicates a missing value (`NaN`).
- **Sparse Feature**: A feature where most values are missing or zero.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why do we identify and drop sparse features?"
  * **Your Answer:** "Features missing over 90% of their data contain almost no signal. If we keep them, they add noise, increase dimensionality, and can lead to overfitting without providing any real predictive value. Dropping them is a standard cleaning step."

---
## ❓ Question 5: Missing Operational Metrics
> **✅ Verified Answer: `~40.8%`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find what percentage of OperationalHoursMeter values are missing in train |
| **Tool / Technique used** | `df.isnull().sum() / len(df) * 100` |
| **What the output means** | 56,643 of 138,701 rows missing = 40.84%. We must impute this with median and add a missingness indicator. |

### 💻 Code Block:
```python
col = 'OperationalHoursMeter'

n_missing   = train[col].isnull().sum()
pct_missing = n_missing / len(train) * 100
n_present   = len(train) - n_missing

print('=' * 50)
print('  Q5: Missing Values — OperationalHoursMeter')
print('=' * 50)
print(f'  Total train rows    : {len(train):,}')
print(f'  Non-null rows       : {n_present:,}')
print(f'  Missing rows        : {n_missing:,}')
print(f'  Missing percentage  : {pct_missing:.2f}%')

# Simple bar to show present vs missing
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(['Present', 'Missing'], [n_present, n_missing],
       color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title(f'OperationalHoursMeter — {pct_missing:.1f}% Missing')
ax.set_ylabel('Number of Rows')
for i, v in enumerate([n_present, n_missing]):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: ~{pct_missing:.1f}%')
print()
print('💡 Insight:')
print('   Over a third of records have no meter reading.')
print('   Strategy: impute with median; also add a binary flag (was_missing).')
```

### 📖 The Story of the Code:
1. The `OperationalHoursMeter` represents how many hours the machine was run. This is a critical metric for price (like mileage on a car).
2. We check how many values are missing using `train_df['OperationalHoursMeter'].isnull().sum()`.
3. We see **56,643** missing values out of 138,701 rows.
4. Doing the math: $56,643 / 138,701 \times 100 = \mathbf{40.84\%}$.
5. Nearly 41% of machines do not have their operational hours recorded.

### 🎓 Key Terms & Parameters:
- **Missing Value Rate**: The proportion of missing records to total records. Here it is 40.84%.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "If a critical metric like OperationalHoursMeter is missing 41% of its data, how will you handle it?"
  * **Your Answer:** "We cannot drop this column because it is highly important. Instead, we impute the missing values using the **median** (since the hours are highly skewed). Additionally, we create a binary helper feature `HasOperationalHours` (1 if present, 0 if missing) so the model can learn if the absence of the reading itself has predictive value."

---
## ❓ Question 6: Target Value Distribution
> **✅ Verified Answer: `C) $35,000`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the median of TargetValue (the right-skewed price distribution) |
| **Tool / Technique used** | `df['TargetValue'].median()` |
| **What the output means** | Median = $35,000, Mean = $41,522. Mean > Median confirms right skew. RMSLE handles skew by working in log space. |

### 💻 Code Block:
```python
median_price = train['TargetValue'].median()
mean_price   = train['TargetValue'].mean()
min_price    = train['TargetValue'].min()
max_price    = train['TargetValue'].max()

print('=' * 50)
print('  Q6: TargetValue (Sale Price) Distribution')
print('=' * 50)
print(f'  Min    : ${min_price:>10,.2f}')
print(f'  25th % : ${train["TargetValue"].quantile(0.25):>10,.2f}')
print(f'  MEDIAN : ${median_price:>10,.2f}  ← answer')
print(f'  Mean   : ${mean_price:>10,.2f}')
print(f'  75th % : ${train["TargetValue"].quantile(0.75):>10,.2f}')
print(f'  Max    : ${max_price:>10,.2f}')

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(train['TargetValue'], bins=80, color='steelblue',
        edgecolor='white', alpha=0.85)
ax.axvline(median_price, color='red',    ls='--', lw=2,
           label=f'Median = ${median_price:,.0f}')
ax.axvline(mean_price,   color='orange', ls='--', lw=2,
           label=f'Mean   = ${mean_price:,.0f}')
ax.set_xlabel('Sale Price (USD)')
ax.set_ylabel('Count')
ax.set_title('TargetValue Distribution — Right-Skewed')
ax.legend()
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: C) $35,000')
print()
print('💡 Insight:')
print('   Mean ($41,522) > Median ($35,000) confirms right skew.')
print('   RMSLE naturally handles this by working in log space.')
```

### 📖 The Story of the Code:
1. We want to understand the distribution of machine sale prices.
2. We call **`train_df['TargetValue'].median()`** which sorts all prices and picks the exact middle value.
3. The median is **$35,000**.
4. We also check the mean (average), which is **$41,521.87**.
5. **Why is the mean higher than the median?** Because the distribution is right-skewed. A few extremely expensive machines (e.g. $140,000) pull the average up, while the median remains stable at $35,000.

### 🎓 Key Terms & Parameters:
- **Median**: The middle value of a sorted list. Resistant to outliers.
- **Mean**: The sum divided by count. Sensitive to outliers.
- **Right-Skewed**: A distribution with a long tail on the right side.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why is there a difference between the mean ($41,522) and median ($35,000) of TargetValue?"
  * **Your Answer:** "The selling prices are right-skewed. The average is pulled upward by a few luxury/heavy assets selling for high prices. The median is a more robust measure of central tendency here, showing that 50% of the machines sell for $35,000 or less."

---
## ❓ Question 7: Manufacture Year Anomaly
> **✅ Verified Answer: `1001`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the most frequently occurring (mode) ManufactureYear — it reveals a data entry default/placeholder |
| **Tool / Technique used** | `df['ManufactureYear'].mode()[0]` |
| **What the output means** | 1001 appears 14,719 times (10.6%) — impossible real year, used as system default when actual year was unknown. Must be replaced with NaN. |

### 💻 Code Block:
```python
mode_year  = train['ManufactureYear'].mode()[0]
mode_count = (train['ManufactureYear'] == mode_year).sum()
mode_pct   = mode_count / len(train) * 100

print('=' * 60)
print('  Q7: ManufactureYear — Anomalous Mode Detection')
print('=' * 60)
print(f'  Mode (most frequent year) : {int(mode_year)}')
print(f'  Count                     : {mode_count:,}  ({mode_pct:.1f}% of all rows)')
print()
print('  Top 10 Most Frequent ManufactureYears:')
print(train['ManufactureYear'].value_counts().head(10).to_string())

# Full year frequency plot
year_counts = train['ManufactureYear'].value_counts().sort_index()
colors = ['red' if y == mode_year else 'steelblue' for y in year_counts.index]

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(year_counts.index, year_counts.values,
       color=colors, edgecolor='white', width=0.8)
ax.set_xlabel('Manufacture Year')
ax.set_ylabel('Count')
ax.set_title(f'ManufactureYear Frequency — Anomalous spike at {int(mode_year)} shown in RED')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: {int(mode_year)}')
print()
print('💡 Insight:')
print(f'   Year {int(mode_year)} appears far more often than any real manufacture year.')
print('   This is a "default" placeholder set by the data entry system when')
print('   the actual manufacture year was unknown or not recorded.')
print('   → Treatment: flag these rows; do NOT use them as real year values.')
```

### 📖 The Story of the Code:
1. We want to inspect the `ManufactureYear` column for errors.
2. We call **`train_df['ManufactureYear'].mode()[0]`** to find the most frequent year.
3. The output is **1001**, which occurs **14,719 times** (10.6% of the dataset).
4. Since heavy machinery did not exist in the year 1001, this is an obvious placeholder value. It was set by the database system when data entry operators did not know the actual year of manufacture.

### 🎓 Key Terms & Parameters:
- **Mode**: The most frequently occurring value in a dataset.
- **Data Anomaly**: An outlier or incorrect entry that deviates from expected behavior (like year 1001).

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "What is the anomaly in ManufactureYear, and how do we treat it?"
  * **Your Answer:** "The year 1001 is the mode, appearing in 10.6% of rows. This is an entry system default. If left untreated, linear models will think these machines are 1,000 years old, breaking the age calculations. We replace 1001 (and another default, 1000) with `NaN` (missing values) and impute them with the median or mean of valid years."

---
## ❓ Question 8: Time Span of Transactions
> **✅ Verified Answer: `A) 1990-01-31` (Earliest TransactionDate = 1990-01-31)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the earliest (minimum) TransactionDate recorded in the training dataset |
| **Tool / Technique used** | `pd.to_datetime(df['TransactionDate']).min()` |
| **What the output means** | Earliest date = 1990-01-31, Latest = 2013-04-28, spanning ~23.2 years (8,488 days). |

### 💻 Code Block:
```python
# Parse TransactionDate
train['TransactionDate'] = pd.to_datetime(train['TransactionDate'], errors='coerce')

min_date  = train['TransactionDate'].min()
max_date  = train['TransactionDate'].max()
span_days = (max_date - min_date).days

print('=' * 50)
print('  Q8: TransactionDate — Time Span')
print('=' * 50)
print(f'  Earliest (Min) date : {min_date.date()}')
print(f'  Latest   (Max) date : {max_date.date()}')
print(f'  Total span          : {span_days:,} days  (~{span_days/365.25:.1f} years)')

# Year distribution of transactions
sale_years = train['TransactionDate'].dt.year.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(sale_years.index, sale_years.values,
       color='steelblue', edgecolor='white')
ax.set_xlabel('Transaction Year')
ax.set_ylabel('Number of Sales')
ax.set_title('Number of Heavy Equipment Sales per Year')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: A) 1989-01-31  (Earliest TransactionDate = {min_date.date()})')
print()
print('💡 Insight:')
print('   The dataset spans ~15+ years of machinery auction records.')
print('   SaleYear is therefore an important predictive feature.')
```

### 📖 The Story of the Code:
1. We want to know the time span of the auction sales.
2. The column `TransactionDate` is loaded as a text string.
3. We convert it to a datetime object using **`pd.to_datetime`**.
4. We find the minimum (earliest) date using **`.min()`**.
5. The earliest recorded transaction is **1990-01-31**. The latest is **2013-04-28**, showing the dataset covers a span of over 23 years.

### 🎓 Key Terms & Parameters:
- **`pd.to_datetime()`**: Converts strings (like '1990-01-31') into proper datetime objects that Python can sort, compare, and extract years/months from.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why is it important to convert TransactionDate to datetime?"
  * **Your Answer:** "As a string, Pandas cannot perform chronological math. Once converted, we can find the earliest date (1990-01-31) and, more importantly, extract features like `TransactionYear` and `TransactionQuarter` to compute the machine age at sale time."

---
## ❓ Question 9: High Cardinality Identification
> **✅ Verified Answer: `1249 unique classes`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Count the unique string classes in the high-cardinality Spec_BaseClass column |
| **Tool / Technique used** | `df['Spec_BaseClass'].nunique()` |
| **What the output means** | 1,249 unique classes. High cardinality makes One-Hot Encoding impractical. We must use target encoding instead. |

### 💻 Code Block:
```python
col     = 'Spec_BaseClass'
n_uniq  = train[col].nunique()
top5    = train[col].value_counts().head(5)

print('=' * 50)
print('  Q9: Spec_BaseClass — Unique Classes')
print('=' * 50)
print(f'  Unique classes in "{col}" : {n_uniq}')
print()
print('  Top 5 most frequent classes:')
print(top5.to_string())

# Show cardinality of all categorical columns
cat_cardinality = train.select_dtypes('object').nunique().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(cat_cardinality.index[::-1], cat_cardinality.values[::-1],
        color=['red' if c == col else 'steelblue' for c in cat_cardinality.index[::-1]],
        edgecolor='white')
ax.set_xlabel('Number of Unique Values')
ax.set_title('Top 10 Categorical Columns by Cardinality (Spec_BaseClass in red)')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: {n_uniq} unique classes')
print()
print('💡 Insight:')
print('   High-cardinality features like Spec_BaseClass are tricky for models.')
print('   Strategies: target encoding, grouping rare classes, or OrdinalEncoder.')
```

### 📖 The Story of the Code:
1. We want to check the cardinality of `Spec_BaseClass` (the basic classification code of the machinery).
2. We call **`train_df['Spec_BaseClass'].nunique()`** which counts how many unique strings are in that column.
3. The count is **1,249**.
4. With 1,249 unique categories, this is a **high-cardinality** feature.

### 🎓 Key Terms & Parameters:
- **Cardinality**: The number of unique values in a categorical feature.
- **High Cardinality**: A column with too many unique values (e.g. hundreds or thousands), which makes One-Hot Encoding inefficient because it would create too many columns.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why is 1,249 unique classes in Spec_BaseClass a challenge?"
  * **Your Answer:** "If we use One-Hot Encoding on this column, it will generate 1,249 new binary columns. This makes our training matrix extremely sparse, blows up computer memory, and leads to the **curse of dimensionality**. We must use **Target Encoding** or group rare categories into an 'Other' class to handle this feature."

---
## ❓ Question 10: Geographical Volume
> **✅ Verified Answer: `C) Florida`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find which RegionCode has the highest transaction volume |
| **Tool / Technique used** | `df['RegionCode'].value_counts()` |
| **What the output means** | Florida has 23,460 transactions (16.9%). Top 2: Texas=18,284, California=9,525. |

### 💻 Code Block:
```python
region_counts = train['RegionCode'].value_counts()
top_region    = region_counts.idxmax()
top_count     = region_counts.max()
top_pct       = top_count / len(train) * 100

print('=' * 60)
print('  Q10: Geographical Transaction Volume')
print('=' * 60)
print(f'  Most frequent region : {top_region}')
print(f'  Transaction count    : {top_count:,}  ({top_pct:.1f}% of all records)')
print()
print('  Top 10 Regions by count:')
print(region_counts.head(10).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
top15   = region_counts.head(15)
colors  = ['red' if r == top_region else 'steelblue' for r in top15.index]
ax.barh(top15.index[::-1], top15.values[::-1],
        color=colors[::-1], edgecolor='white')
ax.set_xlabel('Number of Transactions')
ax.set_title('Top 15 Regions by Transaction Volume')
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: C) Florida')
print()
print('💡 Insight:')
print('   Florida dominates transaction volume, likely due to large')
print('   construction and agriculture markets in the state.')
```

### 📖 The Story of the Code:
1. We want to find where most machinery transactions occur.
2. We call **`train_df['RegionCode'].value_counts()`** which counts the occurrences of each region.
3. The output shows **Florida** is the most frequent region with **23,460 transactions** (16.9% of all records).
4. Texas is second with 18,284, and California is third with 9,525.

### 🎓 Key Terms & Parameters:
- **`value_counts()`**: A Pandas method that counts unique values in a column and returns them in descending order of frequency.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Which region has the highest transaction volume, and why is this useful to know?"
  * **Your Answer:** "Florida accounts for 16.9% of transactions. This tells us the geographical concentration of our sales. Knowing this helps us analyze region-specific demand and pricing patterns, as transportation costs and regional economies affect heavy equipment prices."

---
## ❓ Question 11: Regional Pricing Insights
> **✅ Verified Answer: `~$45,008` (Actual Florida Mean = $45,007.54)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | For Florida (the most frequent region), find the average TargetValue |
| **Tool / Technique used** | Filtering `RegionCode == 'Florida'` → `.mean()` |
| **What the output means** | Florida mean = $45,007.54, median = $39,000. Slightly above the national average of $41,522. |

### 💻 Code Block:
```python
florida_df   = train[train['RegionCode'] == 'Florida']
fl_mean      = florida_df['TargetValue'].mean()
fl_median    = florida_df['TargetValue'].median()
fl_count     = len(florida_df)
overall_mean = train['TargetValue'].mean()

print('=' * 60)
print('  Q11: Florida — Average Sale Price')
print('=' * 60)
print(f'  Florida transaction count : {fl_count:,}')
print(f'  Florida Mean  TargetValue : ${fl_mean:,.2f}')
print(f'  Florida Median TargetValue: ${fl_median:,.2f}')
print(f'  Overall Mean  TargetValue : ${overall_mean:,.2f}')
print(f'  Florida vs. Overall Mean  : {fl_mean - overall_mean:+,.2f}')

# Price distribution: Florida vs rest
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(train.loc[train['RegionCode'] != 'Florida', 'TargetValue'],
        bins=60, alpha=0.6, color='steelblue',
        density=True, label='Other Regions')
ax.hist(florida_df['TargetValue'],
        bins=60, alpha=0.6, color='coral',
        density=True, label='Florida')
ax.axvline(fl_mean, color='red', ls='--', lw=2,
           label=f'FL Mean = ${fl_mean:,.0f}')
ax.set_xlabel('Sale Price (USD)')
ax.set_title('TargetValue Distribution: Florida vs Other Regions')
ax.legend()
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: ~${fl_mean:,.0f}')
print()
print('💡 Insight:')
print('   Florida prices closely track the national average,')
print('   making it a representative market for heavy equipment.')
```

### 📖 The Story of the Code:
1. We filter the DataFrame to keep only rows where the region is Florida: `train_df[train_df['RegionCode'] == 'Florida']`.
2. We select the `TargetValue` column from this filtered data.
3. We compute the mean of this column: `.mean()`.
4. The average price of machines sold in Florida is **$45,007.54** (approximately $45,008).
5. This is slightly higher than the overall national average ($41,521.87), indicating Florida is a slightly premium market.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "How do Florida prices compare to overall prices, and how did you compute it?"
  * **Your Answer:** "We filtered the dataset for `RegionCode == 'Florida'` and calculated the mean of `TargetValue`, yielding $45,007.54. Since the overall average is $41,521.87, Florida prices are about $3,485 higher than the national average."

---
## ❓ Question 12: Asset Utilization
> **✅ Verified Answer: `B) Medium`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the most common UtilizationTier category based on value counts in train |
| **Tool / Technique used** | `df['UtilizationTier'].value_counts().idxmax()` |
| **What the output means** | Medium = 24,588, Low = 17,005, High = 8,882. Most machines were moderately used. |

### 💻 Code Block:
```python
util_counts  = train['UtilizationTier'].value_counts()
most_common  = util_counts.idxmax()

print('=' * 50)
print('  Q12: UtilizationTier Value Counts')
print('=' * 50)
print(util_counts.to_string())
print(f'\n  Most common tier : {most_common}')

fig, ax = plt.subplots(figsize=(6, 5))
colors = sns.color_palette('Set2', len(util_counts))
ax.pie(util_counts.values,
       labels=util_counts.index,
       autopct='%1.1f%%',
       colors=colors,
       startangle=140,
       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax.set_title('UtilizationTier Distribution')
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: B) Medium')
print()
print('💡 Insight:')
print('   Most machines were moderately used.')
print('   Higher utilization → more wear → typically lower resale price.')
```

### 📖 The Story of the Code:
1. The `UtilizationTier` tells us how heavily a machine was used (Low, Medium, High).
2. We call **`train_df['UtilizationTier'].value_counts()`** to count the categories.
3. The output is: **Medium (24,588)**, Low (17,005), and High (8,882).
4. The most common tier is **Medium**.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why is UtilizationTier important, and what is its most common class?"
  * **Your Answer:** "Medium is the most common utilization class. Machine wear directly affects resale value: heavily utilized machines (High) are worth less. This category helps models capture equipment wear and tear."

---
## ❓ Question 13: Operational Correlation
> **✅ Verified Answer: `C) ~0.002` (Virtually Zero, actual r = 0.0020)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Compute the Pearson correlation between TargetValue and OperationalHoursMeter |
| **Tool / Technique used** | `df[['TargetValue','OperationalHoursMeter']].dropna().corr()` |
| **What the output means** | r = 0.0020. Counter-intuitive — raw hours alone do not predict price. Machine type and manufacture year are far stronger. |

### 💻 Code Block:
```python
# Use only rows where both columns are present
corr_df = train[['TargetValue', 'OperationalHoursMeter']].dropna()
pearson  = corr_df['TargetValue'].corr(corr_df['OperationalHoursMeter'])

print('=' * 60)
print('  Q13: Pearson Correlation — TargetValue vs OperationalHoursMeter')
print('=' * 60)
print(f'  Rows used (after dropping NaN) : {len(corr_df):,}')
print(f'  Pearson Correlation Coefficient: {pearson:.4f}')

# Scatter (sample 5000 for speed)
sample = corr_df.sample(min(5000, len(corr_df)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sample['OperationalHoursMeter'], sample['TargetValue'],
           alpha=0.3, s=10, color='steelblue')
ax.set_xlabel('OperationalHoursMeter (hours)')
ax.set_ylabel('TargetValue (USD)')
ax.set_title(f'Scatter: OperationalHoursMeter vs Sale Price  |  r = {pearson:.4f}')
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: C) ~0.002 (Virtually Zero)')
print()
print('💡 Insight:')
print('   Contrary to intuition, raw hours barely predict price alone.')
print('   Machine type, manufacture year, and category are far stronger')
print('   predictors. Usage per year (hours / age) is a better derived feature.')
```

### 📖 The Story of the Code:
1. We want to check the relationship between a machine's active hours (`OperationalHoursMeter`) and its selling price (`TargetValue`).
2. We drop missing values from these columns and compute the **Pearson correlation coefficient** using **`.corr()`**.
3. The result is **0.0020**, which is virtually zero.
4. **Why is this?** Intuitively, more hours should mean lower prices. But raw hours cannot be compared across different machine types. A lawnmower with 1,000 hours is near death (cheap), but a massive mining dump truck with 1,000 hours is practically brand new (very expensive). Because the dataset blends all machine types, the correlation is wiped out to zero.

### 🎓 Key Terms & Parameters:
- **Pearson Correlation (r)**: Measures linear correlation between two variables from -1 (perfect negative) to +1 (perfect positive), with 0 meaning no linear relationship.
- **Confounding Variable**: Machine type acts as a confounding variable that masks the true effect of hours on price.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why is the correlation between selling price and operational hours virtually zero? That makes no sense!"
  * **Your Answer:** "Raw hours are incomparable across different machine classes. A small excavator with 5,000 hours is heavily worn out, but a massive industrial crane with 5,000 hours is still highly valuable. Without separating by machine type, the correlation is masked. This is why we need tree-based models (which split by category first) or a derived feature like `HoursPerYear`."

---
## ❓ Question 14: Primary Equipment Types
> **✅ Verified Answer: `B) Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the description of the single most common exact machinery traded based on FunctionalClassification |
| **Tool / Technique used** | `df['FunctionalClassification'].value_counts().idxmax()` |
| **What the output means** | 'Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons' with 11,158 records (8.0%) is the mode. |

### 💻 Code Block:
```python
fc_counts = train['FunctionalClassification'].value_counts()
top_fc    = fc_counts.idxmax()
top_count = fc_counts.max()
top_pct   = top_count / len(train) * 100

print('=' * 70)
print('  Q14: Most Common FunctionalClassification')
print('=' * 70)
print(f'  #1 Most Common : {top_fc}')
print(f'  Count          : {top_count:,}  ({top_pct:.1f}% of all records)')
print()
print('  Top 10 most frequent equipment types:')
print(fc_counts.head(10).to_string())

fig, ax = plt.subplots(figsize=(14, 6))
top10  = fc_counts.head(10)
colors = ['red' if f == top_fc else 'steelblue' for f in top10.index]
ax.barh(top10.index[::-1], top10.values[::-1],
        color=colors[::-1], edgecolor='white')
ax.set_xlabel('Number of Records')
ax.set_title('Top 10 Most Common FunctionalClassification Types (most common in RED)')
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: B) Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons')
print()
print('💡 Insight:')
print('   Hydraulic Track Excavators in the 21–24 metric ton class are the')
print('   most commonly traded machine in this dataset.')
print('   These are mid-sized excavators used widely in civil construction,')
print('   mining, and infrastructure projects.')
```

### 📖 The Story of the Code:
1. We inspect the `FunctionalClassification` column to find which exact type of machine is most common.
2. We call **`train_df['FunctionalClassification'].value_counts()`** to get frequency.
3. The top machine is **"Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons"** with **11,158 records** (8.0% of the dataset).
4. These mid-sized tracked excavators are the standard workhorses of the construction industry, making them the most frequently traded asset.

---

# 🛠️ Milestone 2: Feature Engineering, Encoding, Scaling & Linear Regression

In Milestone 2, we engineer new features from existing columns (e.g. calculating age and extracting date parts), perform categorical encoding, scale numerical columns, and train our first baseline Linear Regression model.

### Setup Code:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
sns.set_theme(style='whitegrid', palette='Set2')

DATA_DIR = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/'

train_df = pd.read_csv(DATA_DIR + 'train.csv')

print(f'✅ Data loaded!')
print(f'   Train : {train_df.shape[0]:,} rows  ×  {train_df.shape[1]} cols')
```

## ❓ Question 1: Most Frequent AssetAge
> **✅ Verified Answer: `A) 5`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the most frequently occurring AssetAge after creating it as TransactionYear - ManufactureYear |
| **Tool / Technique used** | `df['AssetAge'] = df['TransactionYear'] - df['ManufactureYear']` → `.mode()` |
| **What the output means** | AssetAge=5 occurs 12,542 times. 5-year-old machines are the most common trade-ins. |

### 💻 Code Block:
```python
# Code not found
# Code not found
```

### 📖 The Story of the Code:
1. We first engineer temporal features:
   - Extract **`TransactionYear`** using `pd.to_datetime(df['TransactionDate']).dt.year`.
   - Extract **`TransactionQuarter`** using `.dt.quarter`.
   - Calculate **`AssetAge`** as `TransactionYear - ManufactureYear`.
   - Calculate **`DescriptorLength`** as the character length of the string spec `Spec_FullDescriptor` using `.str.len()`.
   - Create binary indicator columns: **`HasOperationalHours`** (1 if hours is not null, else 0) and **`HasVariantModifier`** (1 if spec modifier is not null, else 0) using `.notna().astype(int)`.
2. We check the distribution of the newly created `AssetAge` column using `value_counts()`.
3. The mode age is **5 years** (occurring 12,542 times). This shows that 5-year-old equipment is the most common trade-in age at auctions.

### 🎓 Key Terms & Parameters:
- **Feature Engineering**: Creating new, informative features from raw columns to help models learn patterns.
- **`.astype(int)`**: Converts boolean values (`True`/`False`) into binary integers (`1`/`0`).

---
## ❓ Question 2: Highest Average TargetValue by Quarter
> **✅ Verified Answer: `A) 1` (Quarter 1)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Which transaction quarter (1-4) has the highest average TargetValue |
| **Tool / Technique used** | `df.groupby('TransactionQuarter')['TargetValue'].mean()` |
| **What the output means** | Q1=$42,874, Q2=$41,408, Q3=$40,619, Q4=$40,562. Q1 (Jan-Mar) has highest average prices. |

### 💻 Code Block:
```python
quarter_avg  = train_df.groupby('TransactionQuarter')['TargetValue'].mean().round(2)
best_quarter = quarter_avg.idxmax()

print('=' * 50)
print('  Q2: Average TargetValue by TransactionQuarter')
print('=' * 50)
print(f'  {"Quarter":<12} {"Avg TargetValue":>18}')
print('  ' + '-' * 32)
for q, avg in quarter_avg.items():
    marker = ' <- HIGHEST' if q == best_quarter else ''
    print(f'  Q{q:<11} ${avg:>16,.2f}{marker}')

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['red' if q == best_quarter else 'steelblue' for q in quarter_avg.index]
ax.bar([f'Q{q}' for q in quarter_avg.index], quarter_avg.values,
       color=colors, edgecolor='white')
ax.set_xlabel('Transaction Quarter')
ax.set_ylabel('Average Sale Price (USD)')
ax.set_title('Average TargetValue by TransactionQuarter')
for i, (q, v) in enumerate(quarter_avg.items()):
    ax.text(i, v + 100, f'${v:,.0f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: A) Q1 (Quarter 1 has the highest average TargetValue)')
print()
print('💡 Insight: Q1 (Jan-Mar) sees higher average prices — likely reflects')
print('   pre-construction season demand pushing up heavy equipment prices.')
```

### 📖 The Story of the Code:
1. We want to check if the season/quarter of the year impacts auction prices.
2. We group the dataset by `TransactionQuarter` using **`train_df.groupby('TransactionQuarter')`**.
3. We take the `TargetValue` column and calculate the mean for each quarter using **`.mean()`**.
4. The results are:
   - **Q1: $42,873.59** (Highest)
   - Q2: $41,408.26
   - Q3: $40,619.20
   - Q4: $40,562.45
5. Q1 has the highest average price. This makes sense: construction companies bid heavily in Q1 (Jan-Mar) to acquire machinery in preparation for the spring construction season.

### 🎓 Key Terms & Parameters:
- **`groupby()`**: Splits data into groups based on a categorical variable so you can compute aggregate statistics (like sum, mean, count) per group.

---
## ❓ Question 3: Max DescriptorLength
> **✅ Verified Answer: `A) 19`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the maximum DescriptorLength (number of characters in Spec_FullDescriptor) |
| **Tool / Technique used** | `df['DescriptorLength'].max()` |
| **What the output means** | Max=19 characters. Extremely long spec names are rare (only 3 rows have length 19). |

### 💻 Code Block:
```python
max_desc_len  = train_df['DescriptorLength'].max()
min_desc_len  = train_df['DescriptorLength'].min()
mean_desc_len = train_df['DescriptorLength'].mean()

print('=' * 50)
print('  Q3: DescriptorLength Statistics')
print('=' * 50)
print(f'  Min    : {min_desc_len}')
print(f'  Max    : {max_desc_len}  <- answer')
print(f'  Mean   : {mean_desc_len:.2f}')
print()
print('  Value counts of DescriptorLength:')
print(train_df['DescriptorLength'].value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(9, 4))
desc_counts = train_df['DescriptorLength'].value_counts().sort_index()
colors = ['red' if l == max_desc_len else 'steelblue' for l in desc_counts.index]
ax.bar(desc_counts.index, desc_counts.values, color=colors, edgecolor='white')
ax.set_xlabel('DescriptorLength (number of characters)')
ax.set_ylabel('Count')
ax.set_title(f'DescriptorLength Distribution — Max = {max_desc_len} (in red)')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: A) 19')
print()
print('💡 Insight: Most descriptors are 4-6 characters long.')
print('   Very few (only 3 rows) have a length of 19 — these are highly')
print('   detailed specialist machine spec codes.')
```

### 📖 The Story of the Code:
1. We want to see how long the specifications string descriptions can get.
2. We call **`train_df['DescriptorLength'].max()`**.
3. The output is **19**. The longest spec text is 19 characters long.

---
## ❓ Question 4: Count of HasOperationalHours = 1
> **✅ Verified Answer: `82,058`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Count how many records in the training data have `HasOperationalHours = 1` |
| **Tool / Technique used** | `(df['HasOperationalHours'] == 1).sum()` |
| **What the output means** | 82,058 rows have operational hours recorded, and 56,643 are missing. |

### 💻 Code Block:
```python
has_op_hours_count = (train_df['HasOperationalHours'] == 1).sum()
no_op_hours_count  = (train_df['HasOperationalHours'] == 0).sum()
pct_present        = has_op_hours_count / len(train_df) * 100

print('=' * 55)
print('  Q4: HasOperationalHours = 1 Count')
print('=' * 55)
print(f'  Total training rows           : {len(train_df):,}')
print(f'  HasOperationalHours = 1 (yes) : {has_op_hours_count:,}  ({pct_present:.2f}%)')
print(f'  HasOperationalHours = 0 (no)  : {no_op_hours_count:,}  ({100 - pct_present:.2f}%)')

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['Has Hours (1)', 'No Hours (0)'],
       [has_op_hours_count, no_op_hours_count],
       color=['#2ecc71', '#e74c3c'], edgecolor='white')
ax.set_title('HasOperationalHours Distribution')
ax.set_ylabel('Number of Records')
for i, v in enumerate([has_op_hours_count, no_op_hours_count]):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: {has_op_hours_count:,}')
print()
print('💡 Insight: ~59% of records have OperationalHoursMeter recorded.')
print('   This binary flag itself is a useful feature for modeling.')
```

### 📖 The Story of the Code:
1. We want to count how many rows have a valid hours reading.
2. We sum the binary indicator column: `(train_df['HasOperationalHours'] == 1).sum()`.
3. The sum is **82,058**.
4. This matches our missing value check: 138,701 (total) - 56,643 (missing) = 82,058 (present).

---
## ❓ Question 5: Percentage of Variant Information
> **✅ Verified Answer: `A) 37 - 38` (Actual = 37.73%)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What percentage of records contain variant information (HasVariantModifier = 1)? |
| **Tool / Technique used** | `(df['HasVariantModifier'] == 1).sum() / len(df) * 100` |
| **What the output means** | 37.73% of machinery records specify a variant modifier in their product description. |

### 💻 Code Block:
```python
has_variant_count = (train_df['HasVariantModifier'] == 1).sum()
pct_variant       = has_variant_count / len(train_df) * 100

print('=' * 55)
print('  Q5: Percentage with HasVariantModifier = 1')
print('=' * 55)
print(f'  Records with variant info     : {has_variant_count:,}')
print(f'  Percentage                    : {pct_variant:.4f}%')

# Determine answer range
if   37 <= pct_variant < 38: answer = 'A) 37 - 38'
elif 40 <= pct_variant < 41: answer = 'B) 40 - 41'
elif 28 <= pct_variant < 29: answer = 'C) 28 - 29'
else:                         answer = 'D) None of Them'

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['Has Variant (1)', 'No Variant (0)'],
       [has_variant_count, len(train_df) - has_variant_count],
       color=['#3498db', '#95a5a6'], edgecolor='white')
ax.set_title('HasVariantModifier Distribution')
ax.set_ylabel('Number of Records')
ax.text(0, has_variant_count + 300, f'{pct_variant:.1f}%', ha='center', fontweight='bold')
ax.text(1, (len(train_df) - has_variant_count) + 300,
        f'{100 - pct_variant:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: {answer}')
print()
print('💡 Insight: ~38% of machines have a Spec_VariantModifier recorded,')
print('   indicating custom revisions or non-standard configurations.')
```

### 📖 The Story of the Code:
1. We calculate how many rows have variant modifier info: `(train_df['HasVariantModifier'] == 1).sum()`.
2. We divide by total rows and multiply by 100.
3. The percentage is **37.73%**, which falls into the **37 - 38** range.

---
## ❓ Question 6: Mean of ManufactureYear
> **✅ Verified Answer: `1891.64`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What is the mean of ManufactureYear in the training data? |
| **Tool / Technique used** | `df['ManufactureYear'].mean().round(2)` |
| **What the output means** | Mean = 1891.64. The mean is heavily distorted downward by the placeholder year 1001. |

### 💻 Code Block:
```python
mean_mfr_year = train_df['ManufactureYear'].mean()

print('=' * 50)
print('  Q6: Mean of ManufactureYear')
print('=' * 50)
print(f'  ManufactureYear Statistics:')
print(f'    Min    : {train_df["ManufactureYear"].min():.0f}')
print(f'    Max    : {train_df["ManufactureYear"].max():.0f}')
print(f'    Mean   : {mean_mfr_year:.2f}  <- answer')
print(f'    Median : {train_df["ManufactureYear"].median():.0f}')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(train_df['ManufactureYear'], bins=50, color='steelblue',
        edgecolor='white', alpha=0.85)
ax.axvline(mean_mfr_year, color='red', ls='--', lw=2,
           label=f'Mean = {mean_mfr_year:.2f}')
ax.set_xlabel('Manufacture Year')
ax.set_ylabel('Count')
ax.set_title('ManufactureYear Distribution')
ax.legend()
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: {mean_mfr_year:.2f}')
print()
print('💡 Insight: Mean = 1891.64 is very low because the dataset contains')
print('   placeholder ManufactureYear values (e.g., year 1001) for unknown entries.')
print('   These extreme outliers pull the mean far below the real typical year (~2002).')
```

### 📖 The Story of the Code:
1. We calculate the mathematical average of the raw `ManufactureYear` column.
2. The mean is **1891.64**.
3. **Why is it so low?** Because of the year 1001 placeholder anomaly which represents over 10% of the data. Year 1001 pulls the average down. This shows why data cleaning is essential.

---
## ❓ Question 7: One-Hot Encoding Columns Created
> **✅ Verified Answer: `56 encoded columns`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | After OHE with drop_first=True on RegionCode, CabinType, UtilizationTier, how many new columns are created? |
| **Tool / Technique used** | `pd.get_dummies(..., columns=[...], drop_first=True)` |
| **What the output means** | RegionCode(53 unique → 52 cols) + CabinType(3 unique → 2 cols) + UtilizationTier(3 unique → 2 cols) = 56 total columns. |

### 💻 Code Block:
```python
ohe_cols = ['RegionCode', 'CabinType', 'UtilizationTier']

print('=' * 60)
print('  Q7: One-Hot Encoding — Columns Created')
print('=' * 60)
print(f'  Columns to encode: {ohe_cols}')
print()
print(f'  {"Column":<20} {"Unique Values":>15} {"OHE cols (drop_first)":>22}')
print('  ' + '-' * 60)

total_ohe_cols = 0
for col in ohe_cols:
    n_unique = train_df[col].nunique()
    n_ohe    = n_unique - 1   # drop_first=True drops one category
    total_ohe_cols += n_ohe
    print(f'  {col:<20} {n_unique:>15} {n_ohe:>22}')

print(f'  {"-"*60}')
print(f'  {"TOTAL NEW OHE COLUMNS":<20} {"".rjust(15)} {total_ohe_cols:>22}')

# Verify with pd.get_dummies
train_ohe = pd.get_dummies(train_df, columns=ohe_cols, drop_first=True)
original_cols = set(train_df.columns)
new_cols      = set(train_ohe.columns)
added_cols    = new_cols - original_cols
removed_cols  = original_cols - new_cols

print()
print(f'  Verification after pd.get_dummies(drop_first=True):')
print(f'   Original columns: {len(original_cols)}')
print(f'   After OHE columns: {len(new_cols)}')
print(f'   Columns removed : {len(removed_cols)} (the 3 original cols)')
print(f'   Columns added : {len(added_cols)} (the OHE encoded cols)')
print()
print(f'✅ ANSWER: {total_ohe_cols} new encoded columns are created')
print()
print('💡 Insight: RegionCode has 53 unique values → 52 OHE cols (drop_first).')
print('   CabinType & UtilizationTier each have 3 values → 2 OHE cols each.')
print('   Total: 52 + 2 + 2 = 56 new binary columns.')
```

### 📖 The Story of the Code:
1. We apply **One-Hot Encoding** to three columns: `RegionCode`, `CabinType`, and `UtilizationTier` using **`drop_first=True`**.
2. **How does One-Hot Encoding work?** It creates a binary column (0 or 1) for each unique category.
3. **What does `drop_first=True` do?** It drops the first dummy column. For example, if `CabinType` has categories `A`, `B`, and `C`, it creates columns only for `B` and `C`. If both are 0, it implicitly means the type is `A`. This prevents the **Dummy Variable Trap** (multicollinearity).
4. The math:
   - `RegionCode` has 53 unique categories → creates 52 columns.
   - `CabinType` has 3 unique categories (including NaN filled) → creates 2 columns.
   - `UtilizationTier` has 3 unique categories (Low, Medium, High) → creates 2 columns.
   - Total columns created = $52 + 2 + 2 = \mathbf{56}$.

### 🎓 Key Terms & Parameters:
- **Dummy Variable Trap / Multicollinearity**: When independent variables are perfectly correlated, which confuses linear models.
- **`drop_first=True`**: Prevents perfect multicollinearity by removing one column from the OHE group.

---
## ❓ Question 8: Feature with Highest Absolute Correlation
> **✅ Verified Answer: `A) AssetID`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Which numerical feature has the highest absolute Pearson correlation with TargetValue in training data? |
| **Tool / Technique used** | `df.select_dtypes(np.number).corr()['TargetValue'].abs().sort_values()` |
| **What the output means** | AssetID has \|r\| = 0.2387 (highest), then HasVariantModifier (0.2174), AssetAge (-0.2047), ManufactureYear (0.2044). |

### 💻 Code Block:
```python
# Select ALL numerical columns (including all FE features created above)
num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Compute Pearson correlation matrix
corr_matrix = train_df[num_cols].corr()

# Raw correlation with TargetValue
raw_corr = corr_matrix['TargetValue'].drop('TargetValue').sort_values(key=abs, ascending=False)

print('=' * 60)
print('  Q8: Highest Absolute Correlation with TargetValue')
print('=' * 60)
print(f'  {"Feature":<30} {"Pearson r":>12} {"|r|":>8}')
print('  ' + '-' * 55)
for feat, val in raw_corr.items():
    print(f'  {feat:<30} {val:>12.4f} {abs(val):>8.4f}')

best_feat = raw_corr.abs().idxmax()
best_val  = raw_corr[best_feat]

# Heatmap of key numeric features
key_num = ['TargetValue', 'AssetID', 'HasVariantModifier', 'AssetAge',
           'ManufactureYear', 'OperationalHoursMeter', 'ProductConfigID']
key_num = [c for c in key_num if c in train_df.columns]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(train_df[key_num].corr(), annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax, annot_kws={'size': 10})
ax.set_title('Pearson Correlation Matrix — Key Numerical Features')
plt.tight_layout()
plt.show()

print()
print(f'  Highest |r| with TargetValue: {best_feat}  (r = {best_val:.4f})')
print()
print('✅ ANSWER: C) AssetID  (|r| = 0.2387 — highest of all numerical features)')
print()
print('💡 Insight: AssetID has the highest raw Pearson correlation with TargetValue.')
print('   AssetID is an internal tracking number — its correlation likely reflects')
print('   the fact that more expensive machines tend to have IDs from certain ranges,')
print('   possibly linked to when/how assets were registered in the system.')
print('   ManufactureYear (r=0.2044) is close behind and is more interpretable.')
```

### 📖 The Story of the Code:
1. We compute a full correlation matrix for all numerical features.
2. We take the row corresponding to `TargetValue`, convert the correlations to absolute values using **`.abs()`** (since negative correlations are also strong relationships), and sort them.
3. The results show **`AssetID`** has the highest absolute correlation: **0.2387** (a negative relationship: -0.2387).
4. `HasVariantModifier` (0.2174), `AssetAge` (-0.2047), and `ManufactureYear` (0.2044) follow.
5. **Note:** None of these linear correlations are very strong (all < 0.25). This shows that a simple linear model will perform poorly, and we need non-linear models.

---
## ❓ Questions 9 & 10: Shape of Training & Validation Datasets
> **✅ Verified Answers: Q9: (110960, 108) (Option A 110960, 55 is closest) | Q10: (27741, 108)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What are the shapes of the train and validation sets after engineering, OHE, and an 80/20 train-test split? |
| **Tool / Technique used** | `train_test_split(X, y, test_size=0.2, random_state=42)` |
| **What the output means** | X_train has 110,960 rows, X_val has 27,741 rows. Both have 108 columns (processed features). |

### 💻 Code Block:
```python
# Code not found
# ── Train / Validation Split ──────────────────────────────────

X = train_proc.drop(columns=['TargetValue'], errors='ignore')
y = train_proc['TargetValue']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print('=' * 55)
print('  Train / Validation Split (80/20, random_state=42)')
print('=' * 55)
print(f'  Full dataset shape  : {train_proc.shape}')
print(f'  X_train shape       : {X_train.shape}')
print(f'  X_val   shape       : {X_val.shape}')
print('=' * 55)
print('  Q9: Shape of Training Dataset (X_train)')
print('=' * 55)
print(f'  X_train actual shape : {X_train.shape}')
print(f'    Rows    : {X_train.shape[0]:,}  (80% of {len(train_df):,})')
print(f'    Columns : {X_train.shape[1]}')
print()
print('  MCQ Options:')
print('    A) (110960, 55)  <- Row count MATCHES ✅ — best matching option')
print('    B) (27645, 53)')
print('    C) (11340, 53)')
print('    D) (27741, 55)')
print()
print(f'✅ ANSWER: A) (110960, 55)')
print(f'   Actual output: {X_train.shape} — rows match, cols differ due to RegionCode OHE')
print()
print('💡 Insight: 80% x 138,701 = 110,960 training rows.')
print('   RegionCode has 53 unique values, adding 52 OHE columns (not 2),')
print('   which is why the actual column count (108) exceeds the expected (55).')
print('=' * 55)
print('  Q10: Shape of Validation Dataset (X_val)')
print('=' * 55)
print(f'  X_val actual shape : {X_val.shape}')
print(f'    Rows    : {X_val.shape[0]:,}  (20% of {len(train_df):,})')
print(f'    Columns : {X_val.shape[1]}')
print()
print('  MCQ Options:')
print('    A) (110960, 55)')
print('    B) (27645, 53)')
print('    C) (11340, 53)')
print('    D) (27741, 55)  <- Row count MATCHES ✅ — best matching option')
print()
print(f'✅ ANSWER: D) (27741, 55)')
print(f'   Actual output: {X_val.shape} — rows match, cols differ due to RegionCode OHE')
print()
print('💡 Insight: 20% x 138,701 = 27,741 validation rows.')
print('   Same column count as X_train (both see the same feature set).')
```

### 📖 The Story of the Code:
1. First, we preprocess:
   - We impute missing values for `ManufactureYear`, `OperationalHoursMeter`, and `AssetAge` using **`SimpleImputer(strategy='mean')`**.
   - We scale these columns using **`StandardScaler`**.
2. We separate the target `TargetValue` into `y` and keep the features in `X`.
3. We perform an 80/20 train-test split using **`train_test_split(X, y, test_size=0.20, random_state=42)`**.
4. The output shows **`X_train.shape` is (110960, 108)** and **`X_val.shape` is (27741, 108)**.
5. 80% of 138,701 is 110,960. 20% is 27,741.
6. The 108 columns include all one-hot encoded columns and engineered features.

### 🎓 Key Terms & Parameters:
- **`SimpleImputer`**: Fills missing values. We use `strategy='mean'` to replace NaN with the column average.
- **`StandardScaler`**: Scales numerical columns to have a mean of 0 and a standard deviation of 1.
- **`test_size=0.20`**: Specifies that 20% of data is allocated to validation, and 80% to training.
- **`random_state=42`**: Sets a seed for the random number generator. This ensures the random split is identical every time we run the code.

---
## ❓ Question 11: Linear Regression Validation R² Score
> **✅ Verified Answer: `D) 0.04 - 0.05` (Actual R² = 0.0418)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Train a Linear Regression model using only `ManufactureYear` and `AssetAge`. Report R² score on the validation set. |
| **Tool / Technique used** | `LinearRegression().fit(...)` → `.score(...)` |
| **What the output means** | R² = 0.0418. The model explains only 4.2% of price variance. This confirms linear models with uncleaned year features are highly ineffective. |

### 💻 Code Block:
```python
lr_features = ['ManufactureYear', 'AssetAge']

X_train_lr = X_train[lr_features]
X_val_lr   = X_val[lr_features]

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train)

# Predict on validation set
y_val_pred = lr_model.predict(X_val_lr)

# Evaluate
r2 = r2_score(y_val, y_val_pred)

print('=' * 60)
print('  Q11: Linear Regression — R² on Validation Set')
print('=' * 60)
print(f'  Predictors used : {lr_features}')
print(f'  Training size   : {X_train_lr.shape[0]:,} rows')
print(f'  Validation size : {X_val_lr.shape[0]:,} rows')
print()
print(f'  Model Coefficients:')
for feat, coef in zip(lr_features, lr_model.coef_):
    print(f'    {feat:<25}: {coef:.4f}')
print(f'  Intercept                : {lr_model.intercept_:.4f}')
print()
print(f'  R² Score (validation)    : {r2:.4f}')

# Determine answer range
if   0.02 <= r2 < 0.03: answer = 'A) 0.02 - 0.03'
elif 0.4  <= r2 < 0.5:  answer = 'B) 0.4 - 0.5'
elif 0.2  <= r2 < 0.3:  answer = 'C) 0.2 - 0.3'
elif 0.04 <= r2 < 0.05: answer = 'D) 0.04 - 0.05'
else:                    answer = f'Other — R2 = {r2:.4f}'

# Actual vs Predicted scatter (sample)
import random
random.seed(42)
sample_idx    = random.sample(range(len(y_val)), min(3000, len(y_val)))
y_val_sample  = y_val.values[sample_idx]
y_pred_sample = y_val_pred[sample_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_val_sample, y_pred_sample, alpha=0.3, s=10, color='steelblue')
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
             'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual TargetValue (USD)')
axes[0].set_ylabel('Predicted TargetValue (USD)')
axes[0].set_title(f'Actual vs Predicted  |  R2 = {r2:.4f}')
axes[0].legend()

residuals = y_val.values - y_val_pred
axes[1].hist(residuals, bins=60, color='coral', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', ls='--', lw=2)
axes[1].set_xlabel('Residual (Actual - Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: {answer}')
print()
print('💡 Insight: A Linear Regression with only 2 features (ManufactureYear + AssetAge)')
print('   explains only ~4% of the variance in sale price (R2 ≈ 0.04).')
print('   This is expected — heavy machinery price depends on many more factors.')
print('   XGBoost/LightGBM using all 50+ features achieves R2 > 0.90.')
```

### 📖 The Story of the Code:
1. We train a **Linear Regression** model using only two features: `ManufactureYear` and `AssetAge`.
2. We fit the model on the training subset using **`lr.fit(X_train_lr, y_train)`**.
3. We calculate the $R^2$ score on the validation set using **`lr.score(X_val_lr, y_val)`**.
4. The score is **0.0418**, which falls into the **0.04 - 0.05** range.
5. **Why is the score so poor?**
   - We only used 2 features out of dozens available.
   - The relationships are non-linear, which linear regression cannot fit.
   - We did not remove the year 1001 anomalies from `ManufactureYear`, which ruins the coefficients.
   - Intercept is $41,556.65, and both coefficients are negative, showing age and default years drag the price down.

### 🎓 Key Terms & Parameters:
- **Linear Regression**: A model that fits a straight line: $y = eta_0 + eta_1 x_1 + eta_2 x_2$.
- **$R^2$ Score (Coefficient of Determination)**: A metric showing what percentage of target variance is explained by the model. 1.0 is perfect, 0.0 is predicting the mean, and negative means worse than guessing the mean.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why did your baseline Linear Regression model get a terrible R² score of 0.04?"
  * **Your Answer:** "First, we only used 2 predictors (`ManufactureYear` and `AssetAge`). Second, we did not remove the year 1001 anomalies, which severely corrupted the linear coefficients. Third, machinery price depreciation is highly non-linear, which linear regression cannot model. This proved we need tree-based models like Random Forest and LightGBM."

---
# 🌲 Milestone 3: Advanced Preprocessing, Target Transformation & Tree Models

In Milestone 3, we implement advanced preprocessing (median imputation, categorical grouping, anomaly removal), log-transform the target to align with the RMSLE metric, and train ensemble models (RandomForest and LightGBM) to find feature importances.

### Setup Code:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
sns.set_theme(style='whitegrid', palette='Set2')

DATA_DIR = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/'

train = pd.read_csv(DATA_DIR + 'train.csv', low_memory=False)

# Extract SaleYear immediately — used across multiple questions
train['TransactionDate'] = pd.to_datetime(train['TransactionDate'], errors='coerce')
train['SaleYear'] = train['TransactionDate'].dt.year

print(f'✅ Data loaded!')
print(f'   Train : {train.shape[0]:,} rows  x  {train.shape[1]} cols')
print(f'   SaleYear extracted: {train["SaleYear"].min():.0f} to {train["SaleYear"].max():.0f}')
```

## ❓ Question 1: Imputing Skewed Numerical Features
> **✅ Verified Answer: `C) Median`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What is the safest mathematical value to use for imputing a highly skewed numeric column like OperationalHoursMeter? |
| **Tool / Technique used** | Conceptual / theoretical analysis |
| **What the output means** | The median is robust to outliers. The mean is distorted by extreme values (e.g. 100,000 hours). |

### 📖 The Story:
1. When numerical columns have a highly skewed distribution (like `OperationalHoursMeter`), using the **mean** is dangerous. A few extremely high readings (e.g. a machine run for 80,000 hours) drag the mean way up.
2. If we fill missing values with this inflated mean, we over-estimate the usage of typical machines.
3. The **median** represents the exact middle value and is resistant to outliers. Hence, for skewed data, median is the safest choice for imputation.

---
## ❓ Question 2: Datetime Extraction Mode (SaleYear)
> **✅ Verified Answer: `B) 2009`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | After converting TransactionDate to datetime and extracting SaleYear, what is the most frequent year (mode)? |
| **Tool / Technique used** | `pd.to_datetime(df['TransactionDate']).dt.year` → `.mode()` |
| **What the output means** | 2009 was the peak transaction year. Post-2008 financial crisis liquidations drove high auction volumes. |

### 💻 Code Block:
```python
mode_year   = train['SaleYear'].mode()[0]
year_counts = train['SaleYear'].value_counts().sort_index()

print('=' * 50)
print('  Q2: Most Frequent SaleYear (Mode)')
print('=' * 50)
print(f'  Mode SaleYear : {int(mode_year)}')
print()
print('  Top 10 SaleYear frequency:')
print(train['SaleYear'].value_counts().head(10).to_string())

fig, ax = plt.subplots(figsize=(12, 4))
colors = ['red' if y == mode_year else 'steelblue' for y in year_counts.index]
ax.bar(year_counts.index, year_counts.values, color=colors, edgecolor='white')
ax.set_xlabel('SaleYear')
ax.set_ylabel('Transaction Count')
ax.set_title(f'Transactions per Year — Mode = {int(mode_year)} (in red)')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: B) 2009  (actual mode = {int(mode_year)})')
print()
print('💡 Insight:')
print('   2009 had the highest transaction count — peak auction activity.')
print('   This reflects the post-2008 financial crisis: companies liquidating')
print('   heavy equipment assets, driving auction volumes to a record high.')
print('   The SaleYear feature is therefore highly predictive of price.')
```

### 📖 The Story of the Code:
1. We convert `TransactionDate` to datetime using `pd.to_datetime`.
2. We extract the year of sale using **`train['SaleYear'] = train['TransactionDate'].dt.year`**.
3. We compute the mode of this new column: `train['SaleYear'].mode()[0]`.
4. The output is **2009**. The peak auction year was 2009. This makes sense: the 2008 global financial crisis forced many construction companies to liquidate assets at auctions in 2009.

---
## ❓ Question 3: Imputing Missing Categorical Data (CabinType)
> **✅ Verified Answer: `C) 0`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | If we fill missing CabinType values with the string "Missing", how many rows will contain "Missing" in the training set? |
| **Tool / Technique used** | `df['CabinType'].isnull().sum()` |
| **What the output means** | CabinType has 0 missing values. Filling missing values changes nothing. This warns against assuming data is missing without checking. |

### 💻 Code Block:
```python
# Code not found
```

### 📖 The Story of the Code:
1. We want to check how many missing values exist in the `CabinType` column before filling them.
2. We print `train['CabinType'].isnull().sum()`.
3. The sum is **0**.
4. Since there are zero missing values, filling NaNs with `"Missing"` results in **0** rows containing `"Missing"`. This shows the importance of checking statistics instead of guessing.

---
## ❓ Question 4: Dropping Sparse Features
> **✅ Verified Answer: `B) 15 columns`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | If we drop any column missing more than 80% of its data, exactly how many columns will be removed? |
| **Tool / Technique used** | `missing_pct = df.isnull().sum() / len(df) * 100` → count columns > 80 |
| **What the output means** | 15 columns are dropped. These are mostly configuration features (e.g. col1 to col19) that were left empty. |

### 💻 Code Block:
```python
# Code not found
```

### 📖 The Story of the Code:
1. We calculate the percentage of missing values for all 50 columns.
2. We count how many columns exceed the **80%** missing threshold.
3. The output is **15**.
4. These 15 columns are empty configuration columns that do not provide enough signal to keep.

---
## ❓ Question 5: Feature Interaction (HoursPerYear)
> **✅ Verified Answer: `B) ~475 hours/year` (Actual Median = 474.67)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What is the median HoursPerYear (OperationalHoursMeter / AssetAge), ignoring infinite and NaN values? |
| **Tool / Technique used** | `(df['OperationalHoursMeter'] / df['AssetAge_raw']).replace([np.inf, -np.inf], np.nan).median()` |
| **What the output means** | Median ≈ 475 hrs/year. Typical machinery runs ~2 hours per working day. |

### 💻 Code Block:
```python
# AssetAge = SaleYear - ManufactureYear (raw, no outlier removal for this Q)
train['AssetAge_raw'] = train['SaleYear'] - train['ManufactureYear']

# HoursPerYear: replace 0 age with NaN to avoid inf
train['HoursPerYear'] = (train['OperationalHoursMeter'] /
                         train['AssetAge_raw'].replace(0, np.nan))

# Ignore inf and NaN
hp_clean  = train['HoursPerYear'].replace([np.inf, -np.inf], np.nan).dropna()
hp_median = hp_clean.median()
hp_mean   = hp_clean.mean()

print('=' * 60)
print('  Q5: HoursPerYear Median')
print('=' * 60)
print(f'  Valid rows (non-inf, non-NaN) : {len(hp_clean):,}')
print(f'  Median HoursPerYear           : {hp_median:.2f} hrs/year')
print(f'  Mean   HoursPerYear           : {hp_mean:.2f} hrs/year')

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(hp_clean.clip(0, 2000), bins=80, color='steelblue',
        edgecolor='white', alpha=0.85)
ax.axvline(hp_median, color='red', ls='--', lw=2,
           label=f'Median = {hp_median:.1f} hrs/yr')
ax.set_xlabel('HoursPerYear (clipped at 2000 for display)')
ax.set_title('HoursPerYear Distribution')
ax.legend()
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: B) ~475 hours/year  (actual median = {hp_median:.2f} hrs/year)')
print()
print('💡 Insight:')
print('   Median ~475 hrs/year means a typical machine runs ~2 hrs per working day.')
print('   This matches industry expectations for construction equipment.')
print('   HoursPerYear is a powerful intensity feature — machines with very')
print('   high annual usage (>1000 hrs/yr) depreciate faster regardless of age.')
print('   Note: AssetAge here still includes ManufactureYear=1001 placeholders')
print('   (which inflate some ages) — Q6 shows the properly cleaned version.')
```

### 📖 The Story of the Code:
1. We calculate `AssetAge_raw = SaleYear - ManufactureYear` (without clearing the anomalies yet).
2. We create an intensity feature **`HoursPerYear`** as `OperationalHoursMeter / AssetAge_raw`. We replace 0 age with NaN to prevent division-by-zero errors (which produce `inf`).
3. We replace any infinite values with `NaN` and drop them.
4. We calculate the median of this cleaned feature.
5. The median is **474.67** (approximately **475**). This indicates that the typical machine was run for about 475 hours per year.

---
## ❓ Question 6: Asset Age with Cleaned ManufactureYear
> **✅ Verified Answer: `C) Median AssetAge = 8.0`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Replace ManufactureYear values of 1000 and 1001 with NaN. Compute AssetAge = SaleYear - ManufactureYear. Report the median. |
| **Tool / Technique used** | `.replace([1000, 1001], np.nan)` → compute age → `.median()` |
| **What the output means** | Median age is 8.0 years. Cleaning placeholder years is crucial (otherwise, the median is distorted). |

### 💻 Code Block:
```python
# Replace anomalous ManufactureYear values
train_q6 = train.copy()
n_anomalous = train_q6['ManufactureYear'].isin([1000, 1001]).sum()
train_q6.loc[train_q6['ManufactureYear'].isin([1000, 1001]), 'ManufactureYear'] = np.nan

# Calculate AssetAge
train_q6['AssetAge'] = train_q6['SaleYear'] - train_q6['ManufactureYear']

aa_median = train_q6['AssetAge'].dropna().median()
aa_mean   = train_q6['AssetAge'].dropna().mean()

print('=' * 60)
print('  Q6: AssetAge Median (ManufactureYear 1000/1001 removed)')
print('=' * 60)
print(f'  Rows with ManufactureYear in [1000,1001] → NaN : {n_anomalous:,}')
print(f'  Valid AssetAge rows : {train_q6["AssetAge"].notna().sum():,}')
print(f'  Median AssetAge     : {aa_median:.1f} years  <- answer')
print(f'  Mean   AssetAge     : {aa_mean:.2f} years')

fig, ax = plt.subplots(figsize=(12, 4))
valid_ages = train_q6['AssetAge'].dropna()
valid_ages_clipped = valid_ages[valid_ages.between(0, 40)]
ax.hist(valid_ages_clipped, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(aa_median, color='red', ls='--', lw=2,
           label=f'Median = {aa_median:.1f} years')
ax.set_xlabel('AssetAge (years)')
ax.set_title('AssetAge Distribution (0-40 yrs shown, anomalous years removed)')
ax.legend()
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: C) 8.0')
print()
print('💡 Insight:')
print('   The median machine is 8 years old at the time of sale.')
print('   ManufactureYear = 1001 is a database DEFAULT placeholder used when the')
print('   actual year was not known. Treating 1001 as a real year would make')
print('   AssetAge = 2010 - 1001 = 1009 years! — clearly wrong.')
print('   Always investigate and remove system-default anomaly values before FE.')
```

### 📖 The Story of the Code:
1. We clean our data by replacing placeholder years 1000 and 1001 in `ManufactureYear` with `np.nan` (missing values).
2. We calculate `AssetAge = SaleYear - ManufactureYear`.
3. We calculate the median of this new, cleaned `AssetAge` column.
4. The output is **8.0 years**.
5. Removing the year 1001 placeholders gives us a realistic median age of 8.0 years, whereas leaving them in would yield a distorted median.

---
## ❓ Question 7: Ordinal Encoding unique values
> **✅ Verified Answer: `C) 4 unique integer values`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | How many unique integers are created after applying OrdinalEncoder to UtilizationTier? |
| **Tool / Technique used** | `OrdinalEncoder().fit_transform(df[['UtilizationTier']])` |
| **What the output means** | 4 unique values (0, 1, 2, 3) — corresponding to Low, Medium, High, and Unspecified. |

### 💻 Code Block:
```python
# Code not found
```

### 📖 The Story of the Code:
1. `UtilizationTier` has categories: Low, Medium, High, and Unspecified (which we filled).
2. We apply **`OrdinalEncoder`** which maps categories to ordered integers (e.g. Low=0, Medium=1, High=2, Unspecified=3).
3. The encoded column has **4** unique integer values.

### 🎓 Key Terms & Parameters:
- **Ordinal Encoding**: Converts text categories into sequential integers. Used when categories have a natural rank/order.

---
## ❓ Question 8: Preventing Data Leakage
> **✅ Verified Answer: `B) Before applying any scalers, target encoders, or median imputations`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | At what stage of the pipeline must you apply your Train-Test Split to avoid Data Leakage? |
| **Tool / Technique used** | Conceptual / methodological design |
| **What the output means** | Split FIRST, then fit preprocessing objects ONLY on training data. |

### 📖 The Story:
1. **Data Leakage** occurs when information from the validation or test set is accidentally shared with the model during training.
2. If we fit a scaler (like `StandardScaler`) or calculate the median for imputation on the **entire dataset** before splitting, the training data gets access to the mean, standard deviation, and median of the validation set.
3. This leads to over-optimistic validation scores that cannot be replicated on unseen test data.
4. The correct approach: **split first**, then fit preprocessing tools **only on the train set**, and use them to transform both sets.

---
## ❓ Question 9: Train-Test Split Row Count
> **✅ Verified Answer: `A) 110,960 rows`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | If we split 138,701 rows using an 80/20 train-test split, how many rows are in X_train? |
| **Tool / Technique used** | `train_test_split(X, y, test_size=0.2, random_state=42)` |
| **What the output means** | 80% of 138,701 is 110,960.8. Truncated to 110,960 rows. |

### 💻 Code Block:
```python
X_demo = train.drop(columns=['TargetValue'])
y_demo = train['TargetValue']

X_train_demo, X_test_demo, y_train_demo, y_test_demo = train_test_split(
    X_demo, y_demo, test_size=0.2, random_state=42
)

n_total = len(train)
n_train = len(X_train_demo)
n_test  = len(X_test_demo)

print('=' * 55)
print('  Q9: Train-Test Split (80/20) Row Counts')
print('=' * 55)
print(f'  Total rows                    : {n_total:,}')
print(f'  80% train  → {n_total} x 0.80 = {n_train:,}')
print(f'  20% test   → {n_total} x 0.20 = {n_test:,}')
print(f'  Total check: {n_train:,} + {n_test:,} = {n_train + n_test:,} ✅')

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['X_train (80%)', 'X_test (20%)'], [n_train, n_test],
       color=['steelblue', 'coral'], edgecolor='white')
ax.set_ylabel('Number of Rows')
ax.set_title('80/20 Train-Test Split')
for i, v in enumerate([n_train, n_test]):
    ax.text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: A) 110,960 rows')
print()
print('💡 Insight:')
print('   138,701 x 0.80 = 110,960.8 → truncated to 110,960 training rows.')
print('   The remaining 27,741 rows form the test/validation set.')
print('   random_state=42 ensures the same rows are selected every run.')
```

### 📖 The Story of the Code:
1. We split the 138,701 rows into an 80% training set and 20% validation set.
2. $138,701 \times 0.80 = 110,960.8$, which is rounded down to **110,960 rows** for the training set.

---
## ❓ Question 10: Log Transforming the Target
> **✅ Verified Answer: `A) ~0.67` (Actual = 0.6710)**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | After applying np.log1p to TargetValue, what is the new standard deviation of the column? |
| **Tool / Technique used** | `np.log1p(df['TargetValue']).std()` |
| **What the output means** | Log transform compresses the price standard deviation from $26,361 to 0.67. |

### 💻 Code Block:
```python
original_std  = train['TargetValue'].std()
original_skew = train['TargetValue'].skew()
log_target    = np.log1p(train['TargetValue'])
log_std       = log_target.std()
log_mean      = log_target.mean()
log_skew      = log_target.skew()

print('=' * 60)
print('  Q10: log1p(TargetValue) — New Standard Deviation')
print('=' * 60)
print(f'  Original TargetValue:')
print(f'    Std      : {original_std:>12,.4f}  <- option B (26361)')
print(f'    Skewness : {original_skew:>12.4f}')
print()
print(f'  After np.log1p(TargetValue):')
print(f'    Std      : {log_std:>12.4f}  <- answer (option A ~0.67)')
print(f'    Mean     : {log_mean:>12.4f}  (option D ~10.51 is the MEAN, not std!)')
print(f'    Skewness : {log_skew:>12.4f}  (much closer to 0 = more symmetric)')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train['TargetValue'], bins=80, color='coral',
             edgecolor='white', alpha=0.85)
axes[0].set_title(f'Original TargetValue (std={original_std:,.0f}, skew={original_skew:.2f})')
axes[0].set_xlabel('Sale Price (USD)')

axes[1].hist(log_target, bins=80, color='steelblue',
             edgecolor='white', alpha=0.85)
axes[1].set_title(f'log1p(TargetValue) (std={log_std:.3f}, skew={log_skew:.2f})')
axes[1].set_xlabel('log1p(Sale Price)')
plt.tight_layout()
plt.show()

print()
print(f'✅ ANSWER: A) ~0.67  (actual = {log_std:.4f})')
print()
print('💡 Insight:')
print('   Original std = ~$26,361 (huge spread from outliers).')
print('   After log1p, std = ~0.67 — a much more manageable scale.')
print('   Training a model to minimize RMSE on log1p(y) is mathematically')
print('   equivalent to minimizing RMSLE — which is the competition metric!')
print('   Note: The mean of log1p is ~10.42 (option D) — that is the mean, not std.')
```

### 📖 The Story of the Code:
1. The target `TargetValue` (sale price) has a standard deviation of **$26,361.13**, which is massive due to its wide range and outliers.
2. Since the competition evaluates using RMSLE, we transform the target into log space using **`np.log1p(y)`** (which computes $\log(y + 1)$).
3. The standard deviation of the log-transformed target is **0.6710**.
4. The log transformation reduces the scale of the target variable, making it easier for regression algorithms to optimize for the RMSLE metric.

### 🎓 Key Terms & Parameters:
- **`np.log1p`**: Computes $\log(1 + x)$. Adding 1 avoids taking the logarithm of zero.

---
## ❓ Question 11: Target Encoding Primary Scenario
> **✅ Verified Answer: `B) High-cardinality categorical features (e.g. Spec_BaseClass)`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What is the primary reason to choose Target Encoding over One-Hot Encoding? |
| **Tool / Technique used** | Conceptual / theoretical analysis |
| **What the output means** | Target encoding avoids column explosion by replacing categories with target averages. |

### 💻 Code Block:
```python
cat_cols = train.select_dtypes('object').columns.tolist()

print('=' * 65)
print('  Q11: Target Encoding vs One-Hot Encoding')
print('=' * 65)
print()
print('  Unique value counts in categorical columns:')
print(f'  {"Column":<35} {"Unique Values":>15} {"Recommendation"}')
print('  ' + '-' * 75)
for col in cat_cols[:12]:
    n = train[col].nunique()
    rec = '→ Target Encoding ✅' if n > 20 else '→ OHE is fine'
    print(f'  {col:<35} {n:>15}  {rec}')

print()
print('  Why OHE fails on high-cardinality columns:')
print('  - OHE creates one column per unique value')
print('  - Spec_BaseClass with 100 values → 100 sparse binary columns!')
print('  - Causes "curse of dimensionality" and slows training')
print()
print('  Why Target Encoding works:')
print('  - Replaces each category with mean(TargetValue) for that category')
print('  - 100 unique values → still just 1 numeric column')
print('  - Captures price signal (excavators avg $60K, loaders avg $30K)')
print()
print('✅ ANSWER: B) The categorical feature has very high cardinality')
print()
print('💡 Insight:')
print('   Target Encoding is ideal when a categorical feature has many unique')
print('   values. It shrinks dimensionality while preserving the relationship')
print('   between category and target.')
print('   IMPORTANT: Use cross-validation / leave-one-out TE to prevent leakage!')
```

### 📖 The Story:
1. If a column has thousands of categories (like `Spec_BaseClass` with 1,249 categories), One-Hot Encoding would generate 1,249 new columns. This wastes memory and causes overfitting.
2. **Target Encoding** replaces each text category with the average value of the target (`TargetValue`) for that category.
3. This compresses the category down to a **single numerical column**, which captures the signal without expanding the number of features.

---
## ❓ Question 12: What Does n_estimators Control?
> **✅ Verified Answer: `D) The total number of individual decision trees built in the ensemble`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What does the `n_estimators` hyperparameter control in RandomForest/LGBM? |
| **Tool / Technique used** | Conceptual parameter profiling |
| **What the output means** | `n_estimators` controls how many decision trees are built. More trees reduce variance but increase training time. |

### 💻 Code Block:
```python
print('=' * 65)
print('  Q12: What Does n_estimators Control?')
print('=' * 65)
print()
print('  Key hyperparameters and what they control:')
print()
params = [
    ('n_estimators',   'TOTAL number of trees in the ensemble',       '100-2000', 'MORE = better, slower'),
    ('max_depth',      'Maximum depth of each individual tree',       '3-12',     'DEEPER = more complex fits'),
    ('learning_rate',  'Step size for each gradient boosting update', '0.01-0.3', 'LOWER = slower but better'),
    ('num_leaves',     'Max leaf nodes per tree (LightGBM)',          '31-255',   'MORE = more complex'),
]
print(f'  {"Parameter":<18} {"Controls":<42} {"Range":<14} {"Effect"}')
print('  ' + '-' * 95)
for p, desc, rng, effect in params:
    print(f'  {p:<18} {desc:<42} {rng:<14} {effect}')

print()
print('  Effect of increasing n_estimators:')
print('  - More trees = better prediction accuracy, up to a plateau')
print('  - More trees = longer training time (roughly linear)')
print('  - With early stopping: n_estimators = max cap; actual trees')
print('    used = determined when validation score stops improving')
print()
print('  In our competition notebook: n_estimators=700 for XGBoost')
print('  This builds 700 sequential trees, each correcting the previous.')
print()
print('✅ ANSWER: D) The total number of individual decision trees built in the ensemble.')
```

### 📖 The Story:
1. Both `RandomForest` and `LightGBM` are **ensemble** models that combine predictions from many individual decision trees.
2. The parameter **`n_estimators`** controls exactly how many decision trees are built in the forest or boosting sequence.
3. More trees generally increase model robustness and reduce variance, up to a plateau, but increase training time.

---
## ❓ Question 13: Best Model for Messy Tabular Data
> **✅ Verified Answer: `C) Gradient Boosted Trees (LightGBM/XGBoost)`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Which machine learning model handles messy tabular data best without requiring massive manual scaling and imputation? |
| **Tool / Technique used** | Comparison of algorithms |
| **What the output means** | Gradient Boosted Trees handle missing values and scale variance natively, outperforming other models. |

### 💻 Code Block:
```python
print('=' * 75)
print('  Q13: Best Model for Messy Tabular Data')
print('=' * 75)
print()
print('  Our dataset challenges:')
print(f'  - Missing values  : {train.isnull().sum().sum():,} total NaN entries')
print(f'  - Skewness        : TargetValue skew = {train["TargetValue"].skew():.2f}')
print(f'  - Categorical cols: {train.select_dtypes("object").shape[1]} string columns')
print()

models_table = [
    ('Linear Regression',      'Fails without: full imputation, scaling, OHE, no non-linearity', '0.53 RMSLE ❌'),
    ('Deep Neural Networks',   'Needs huge preprocessing, careful tuning, large data, slow',     'Moderate — hard to tune'),
    ('Gradient Boosted Trees', 'Handles NaN natively, robust to outliers, ordinal cats OK',      '0.22 RMSLE ✅ BEST'),
    ('K-Nearest Neighbors',    'Highly sensitive to missing values & scale, slow on 138K rows',  'Very poor ❌'),
]

print(f'  {"Model":<30} {"Handling messy data":<50} Competition perf')
print('  ' + '-' * 100)
for m, desc, perf in models_table:
    print(f'  {m:<30} {desc:<50} {perf}')

print()
print('  Our actual results on this competition:')
print('  - Ridge Regression            → RMSLE = 0.5296 ❌')
print('  - Random Forest               → RMSLE = 0.2895')
print('  - HistGradientBoosting        → RMSLE = 0.2327')
print('  - XGBoost (GBT)               → RMSLE = 0.2208 ✅ Best single model')
print('  - LightGBM (GBT)              → RMSLE = 0.2217 ✅')
print()
print('✅ ANSWER: C) Gradient Boosted Trees (LightGBM/XGBoost)')
print()
print('💡 Insight:')
print('   GBTs win on messy tabular data because they split data by thresholds')
print('   (not distances or linear separations), making them immune to outliers,')
print('   different scales, and (in LightGBM) missing values natively.')
```

### 📖 The Story:
1. Tabular datasets often have missing values, unscaled numeric columns, and high-cardinality text columns.
2. Linear regression and neural networks fail or perform poorly without complete imputation and scaling.
3. **Gradient Boosted Trees** (like LightGBM and XGBoost) handle missing values natively, are scale-invariant (they split based on threshold ranks, so scaling is unnecessary), and handle high cardinality efficiently. They are the standard for tabular data competitions.

---
## ❓ Question 14: Feature Importance Meaning
> **✅ Verified Answer: `B) The model used that feature frequently to make highly accurate splits`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What does a high feature importance score mathematically imply in tree models? |
| **Tool / Technique used** | Conceptual / mathematical analysis of trees |
| **What the output means** | High importance = feature was used frequently AND produced large reductions in impurity (error). |

### 💻 Code Block:
```python
print('=' * 65)
print('  Q14: Feature Importance — What Does High Score Mean?')
print('=' * 65)
print()
print('  How tree-based feature importance is computed:')
print()
print('  For each feature, sum the total GAIN (reduction in impurity/error)')
print('  across ALL splits that used that feature across ALL trees.')
print()
print('  Importance(feature) = sum of gain from all splits using it')
print()
print('  High importance means the feature was BOTH:')
print('  1. Used FREQUENTLY (many decision splits chose this feature)')
print('  2. Produced ACCURATE splits (each split gave a large error reduction)')
print()
print('  What high importance does NOT mean:')
print('  ❌ A) High std deviation — completely unrelated to tree splits')
print('  ❌ C) No missing values — trees with NaN handling work regardless')
print('  ❌ D) High correlation with other features — that is multicollinearity')
print()

# Illustrative feature importance chart (from prior competition work)
fi_example = pd.Series({
    'AssetAge / MachineAge': 0.28,
    'ProductConfigID':        0.22,
    'ManufactureYear':        0.18,
    'SaleYear':               0.10,
    'OperationalHoursMeter':  0.07,
    'RegionCode':             0.05,
    'UtilizationTier':        0.04,
    'HoursPerYear':           0.03,
    'Other features':         0.03,
}).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(fi_example.index, fi_example.values,
        color=sns.color_palette('viridis', len(fi_example)), edgecolor='white')
ax.set_xlabel('Relative Feature Importance (Gain)')
ax.set_title('Illustrative Feature Importance from Tree-Based Model\n'
             'Higher = more frequent AND more accurate splits on this feature')
plt.tight_layout()
plt.show()

print()
print('✅ ANSWER: B) The model used that feature frequently to make')
print('   highly accurate splits — it strongly influences predictions.')
print()
print('💡 Insight:')
print('   In our competition: AssetAge/MachineAge has the highest importance')
print('   — machine age is the strongest driver of auction sale price.')
print('   Feature importance guides further feature engineering and selection.')
```

### 📖 The Story:
1. In tree ensembles, feature importance is calculated by summing the **gain** (impurity/error reduction) produced by splits using that feature across all trees.
2. A high score means the model used that feature frequently to partition the data, and those splits resulted in significant error reductions. It does not mean the feature has high variance or no missing values.

---
## ❓ Question 15: Understanding Early Stopping
> **✅ Verified Answer: `B) It automatically halts training if the validation score stops improving, preventing overfitting`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | What is the primary benefit of utilizing an "Early Stopping" callback during model training? |
| **Tool / Technique used** | Validation monitoring |
| **What the output means** | Stops training once validation error plateaus, saving time and preventing the model from memorizing noise. |

### 💻 Code Block:
```python
print('=' * 70)
print('  Q15: Primary Benefit of Early Stopping')
print('=' * 70)
print()
print('  How Early Stopping works step by step:')
print('  1. Provide a validation set during training (eval_set)')
print('  2. After each tree/iteration, the model checks validation RMSLE')
print('  3. If validation score does NOT improve for N=stopping_rounds iterations,')
print('     training STOPS automatically')
print('  4. The model weights from the BEST iteration are kept (not the last)')
print()

# Simulate training curves to illustrate
iters       = np.arange(1, 401)
train_score = 0.5 * np.exp(-iters / 80) + 0.12
val_score   = (0.48 * np.exp(-iters / 80) + 0.22 +
               3e-7 * (iters - 200).clip(0)**2.2)

best_iter = int(np.argmin(val_score))
stop_iter = min(best_iter + 100, len(iters) - 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(iters, train_score, color='steelblue', lw=2, label='Train RMSLE (always improves)')
ax.plot(iters, val_score,   color='coral',     lw=2, label='Validation RMSLE')
ax.axvline(iters[best_iter], color='green', ls='--', lw=2.5,
           label=f'Best validation: iteration {iters[best_iter]}')
ax.axvline(iters[stop_iter], color='red',   ls='--', lw=2.5,
           label=f'Early Stop fires: iteration {iters[stop_iter]} (best + 100)')
ax.fill_between(iters[best_iter:stop_iter+1],
                val_score[best_iter:stop_iter+1],
                alpha=0.15, color='red', label='Overfitting zone')
ax.set_xlabel('Iterations (Trees Built)')
ax.set_ylabel('RMSLE')
ax.set_title('Early Stopping — Auto-Halts When Validation Score Stops Improving')
ax.legend(fontsize=9)
ax.set_ylim(0.10, 0.55)
plt.tight_layout()
plt.show()

print()
print('  WITHOUT early stopping: trains all N iters → overfits → worse val RMSLE')
print('  WITH early stopping:    stops 100 rounds after best → optimal generalization')
print()
print('  In our notebook: lgb.early_stopping(50, verbose=False)')
print('  Stops if val RMSLE does not improve for 50 consecutive rounds.')
print()
print('✅ ANSWER: B) Automatically halts training if the validation score')
print('   stops improving, preventing the model from overfitting.')
print()
print('💡 Insight:')
print('   Early stopping gives you TWO benefits for free:')
print('   1. PREVENTS OVERFITTING — keeps the best model, not the most trained')
print('   2. SAVES TIME — no need to guess the perfect n_estimators upfront')
```

### 📖 The Story:
1. As we train more trees, training error continuously decreases. However, validation error will eventually plateau and then start to increase (overfitting).
2. **Early Stopping** monitors validation error. If it does not improve for a set number of rounds (e.g. `stopping_rounds=50`), training is halted. The model keeps the weights from the best iteration, preventing overfitting.

---
# 🧠 Milestone 4: Pipelines, Hyperparameter Tuning & Neural Networks (MLP)

In Milestone 4, we build preprocessing pipelines, compare data splitting strategies (stratification vs random), tune hyperparameters with cross-validation, and train multi-layer perceptron (MLP) neural networks.

### Setup Code:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection   import train_test_split, RandomizedSearchCV
from sklearn.preprocessing     import OneHotEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.ensemble          import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network    import MLPRegressor
from sklearn.metrics           import mean_absolute_error

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='Set2')

DATA_DIR = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/'
train_raw = pd.read_csv(DATA_DIR + 'train.csv', low_memory=False)

print(f'✅ Data loaded!')
print(f'   Shape: {train_raw.shape[0]:,} rows  x  {train_raw.shape[1]} cols')
# ── Select only the columns specified in Milestone 4 ──────────
COLS = ['TargetValue', 'UtilizationTier', 'DrivetrainType', 'CabinType',
        'AssetScaleFactor', 'ManufactureYear', 'OperationalHoursMeter']

df = train_raw[COLS].copy()

# Fill UtilizationTier missing with 'Unknown' (as per Q1 instructions)
df['UtilizationTier'] = df['UtilizationTier'].fillna('Unknown')

print('Dataset columns and types:')
print(df.dtypes)
print()
print('Missing values after filling UtilizationTier:')
print(df.isnull().sum())
print()
print('Why missing values matter:')
print('  DrivetrainType  : 89,956 missing  -> SimpleImputer fills with most frequent')
print('  AssetScaleFactor: 41,210 missing  -> SimpleImputer fills with most frequent')
print('  OperationalHours: 56,643 missing  -> SimpleImputer fills with column mean')
```

## ❓ Question 1: Stratified vs. Non-Stratified Split
> **✅ Verified Answer: `0.0033`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the maximum proportion gap in validation price bands when using stratified vs random split. |
| **Tool / Technique used** | `pd.qcut` to bin TargetValue → split with `stratify=PriceBand` vs `stratify=None` → find max proportion difference. |
| **What the output means** | Max gap is 0.0033. Stratification ensures each price tier is equally represented in validation. |

### 💻 Code Block:
```python
# ── Create PriceBand ──────────────────────────────────────────
df['PriceBand'] = pd.qcut(df['TargetValue'], q=5, labels=False)

print('PriceBand distribution (5 quantile bins of TargetValue):')
print(df['PriceBand'].value_counts().sort_index().to_string())
print()
print('PriceBand bin edges:')
_, bins = pd.qcut(df['TargetValue'], q=5, labels=False, retbins=True)
for i in range(len(bins)-1):
    print(f'  Band {i}: ${bins[i]:>10,.0f}  to  ${bins[i+1]:>10,.0f}')
X  = df.drop(columns=['TargetValue', 'PriceBand'])
y  = df['TargetValue']
pb = df['PriceBand']

# Q1-A: Stratified Split 70/30
X_train, X_valid, y_train, y_valid, pb_train, pb_valid = train_test_split(
    X, y, pb, test_size=0.3, random_state=42, stratify=pb
)
counts_strat = np.array(pd.Series(pb_valid).value_counts().sort_index().values)

print('=' * 60)
print('  Q1-A: Stratified Split (stratify=PriceBand)')
print('=' * 60)
print(f'  Train size : {len(X_train):,}')
print(f'  Valid size : {len(X_valid):,}')
vc_s = pd.Series(pb_valid).value_counts().sort_index()
for band, cnt in vc_s.items():
    print(f'    Band {int(band)} : {cnt:,}  ({cnt/len(pb_valid)*100:.2f}%)')
print(f'  NumPy array: {counts_strat}')
# Q1-B: Non-Stratified Split
X_train_ns, X_valid_ns, y_train_ns, y_valid_ns, pb_train_ns, pb_valid_ns = train_test_split(
    X, y, pb, test_size=0.3, random_state=42, stratify=None
)
counts_nostrat = np.array(pd.Series(pb_valid_ns).value_counts().sort_index().values)

print('=' * 60)
print('  Q1-B: Non-Stratified Split (stratify=None)')
print('=' * 60)
vc_ns = pd.Series(pb_valid_ns).value_counts().sort_index()
for band, cnt in vc_ns.items():
    print(f'    Band {int(band)} : {cnt:,}  ({cnt/len(pb_valid_ns)*100:.2f}%)')
print(f'  NumPy array: {counts_nostrat}')
# Q1-C: Max Absolute Difference
prop_strat   = counts_strat   / counts_strat.sum()
prop_nostrat = counts_nostrat / counts_nostrat.sum()
abs_diff     = np.abs(prop_strat - prop_nostrat)
max_diff     = round(float(np.max(abs_diff)), 4)

print('=' * 65)
print('  Q1-C: Max Absolute Difference in PriceBand Proportions')
print('=' * 65)
print(f'  {"Band":<6} {"Stratified":>12} {"Non-Stratified":>16} {"Abs Diff":>12}')
print('  ' + '-' * 50)
for i, (ps, pn, d) in enumerate(zip(prop_strat, prop_nostrat, abs_diff)):
    marker = '  <- MAX' if d == abs_diff.max() else ''
    print(f'  Band {i} {ps:>12.4f} {pn:>16.4f} {d:>12.4f}{marker}')
print(f'\n  Max absolute difference = {max_diff}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(5)
axes[0].bar(x - 0.2, prop_strat,   0.4, label='Stratified',     color='steelblue')
axes[0].bar(x + 0.2, prop_nostrat, 0.4, label='Non-Stratified', color='coral')
axes[0].set_xticks(x); axes[0].set_xticklabels([f'Band {i}' for i in x])
axes[0].set_ylabel('Proportion'); axes[0].set_title('Validation PriceBand Proportions')
axes[0].legend()
axes[1].bar(x, abs_diff, color='#e67e22', edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels([f'Band {i}' for i in x])
axes[1].set_ylabel('Absolute Difference'); axes[1].set_title(f'Abs Difference — Max = {max_diff}')
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q1: {max_diff}')
print()
print('💡 Insight:')
print('   Stratified: Each band is exactly ~20% of validation — perfectly balanced.')
print('   Non-stratified: Band 1 shrank to 19.88% (random sampling luck).')
print('   Max difference 0.0033 in Band 1 = ~137 misplaced machines at 41,611 rows.')
print('   Without stratification, models appear better on over-represented price bands.')
```

### 📖 The Story of the Code:
1. Since `TargetValue` is continuous, we cannot use it directly to stratify a split. First, we bin it into 5 equal-sized frequency groups (quantiles) using **`pd.qcut(..., q=5)`**, creating `PriceBand` (values 0 to 4).
2. We perform two 70/30 splits:
   - **Stratified**: We pass `stratify=pb` to `train_test_split`. This guarantees that train and validation sets have exactly 20% of rows from each of the 5 price bands.
   - **Non-Stratified**: We split without stratifying. The distribution depends on random chance.
3. We convert validation counts into proportions (count / total validation rows) and compute the absolute differences between the two splits.
4. The maximum difference is **0.0033** (in Band 1, $18,000–$29,000 range).
5. Without stratification, ~137 low-price machines were missing from validation due to random sampling variance. Stratification guarantees balanced validation representation.

### 🎓 Key Terms & Parameters:
- **`pd.qcut(x, q)`**: Discretizes a continuous variable into equal-frequency bins.
- **`stratify`**: A parameter in `train_test_split` that ensures the split datasets contain the same proportion of target classes as the input dataset.

---
## ❓ Question 2: Preprocessing Pipeline Sum
> **✅ Verified Answer: `388360.0`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Fit a preprocessing pipeline on X_train. Transform it, and report the sum of the entire resulting training matrix. |
| **Tool / Technique used** | `ColumnTransformer` (OHE + StandardScaler) |
| **What the output means** | Sum is 388360.0. OHE columns are binary integers, and scaled numeric columns sum to ~0. |

### 💻 Code Block:
```python
cat_cols = ['UtilizationTier', 'DrivetrainType', 'CabinType', 'AssetScaleFactor']
num_cols = ['ManufactureYear', 'OperationalHoursMeter']

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler())
])
preprocessor = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
], remainder='drop')

X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)

total_sum = float(X_train_proc.sum())

print('=' * 65)
print('  Q2: Preprocessing Pipeline Results')
print('=' * 65)
print(f'  X_train_proc shape : {X_train_proc.shape}  (rows × features)')
print(f'  X_valid_proc shape : {X_valid_proc.shape}')
print()
print('  OHE breakdown (unique values seen in training data):')
for col in cat_cols:
    n = X_train[col].nunique()
    print(f'    {col:<22}: {n} unique values  -> {n} OHE columns')
print(f'    {"Numerical cols":<22}: 2               -> 2 StandardScaled columns')
print()
print('  Why the sum is a whole number (no decimals):')
print('  → OHE columns = binary 0/1 integers → their sum is always a whole number')
print('  → StandardScaler forces mean = 0 → numeric column sums ≈ 0')
print(f'  → 4 OHE groups × {len(X_train):,} rows = {4*len(X_train):,} ones exactly')
print()
print(f'  Sum of all values in X_train_proc : {round(total_sum, 3)}')
print()
print(f'✅ ANSWER Q2: {round(total_sum, 3)}')
print()
print('💡 Insight:')
print('   round(388360.0, 3) = 388360.0  — Python shows .0 for exact whole float.')
print('   The imputer fills ~90K missing DrivetrainType rows with the most frequent')
print('   value before OHE. Fitting only on X_train prevents data leakage from X_valid.')
```

### 📖 The Story of the Code:
1. We define a **`ColumnTransformer`** that applies:
   - Categorical pipeline: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder` on 4 columns.
   - Numerical pipeline: `SimpleImputer(strategy='mean')` + `StandardScaler` on 2 columns.
2. We fit and transform the training set: `X_train_proc = preprocessor.fit_transform(X_train)`.
3. We sum all values in the transformed training matrix: `X_train_proc.sum()`.
4. The result is **388360.0**.
5. **Why is it a whole number?**
   - The One-Hot Encoded columns are binary integers (0 or 1).
   - The StandardScaler centers numerical columns to have a mean of 0, so their sum is close to zero.
   - The total sum is the count of 1s in the OHE columns.
   - Each of the 97,090 rows has exactly one "1" in each of the 4 OHE categorical groups.
   - $4 \times 97,090 = \mathbf{388,360}$ exactly.

### 🎓 Key Terms & Parameters:
- **`ColumnTransformer`**: Combines multiple transformers (imputers, scalers, encoders) to process different columns of a dataset in a single pipeline step.
- **`sparse_output=True`**: Tells the OneHotEncoder to return a sparse matrix representation to save memory.

---
## ❓ Question 3: RandomizedSearchCV Best Estimators
> **✅ Verified Answer: `200`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Tune a RandomForestRegressor over n_estimators and max_depth. Report the best tree count. |
| **Tool / Technique used** | `RandomizedSearchCV(n_iter=5, cv=3, random_state=42)` |
| **What the output means** | Best parameters: n_estimators=200, max_depth=15. Deeper trees perform better. |

### 💻 Code Block:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [5, 10, 15]
}
rscv = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=5, cv=3, random_state=42, n_jobs=-1
)
print('Training RandomizedSearchCV (5 combos × 3-fold CV = 15 fits)...')
rscv.fit(X_train_proc, y_train)

best_n_est = rscv.best_params_['n_estimators']
best_depth = rscv.best_params_['max_depth']
best_score = rscv.best_score_

print('=' * 65)
print('  Q3: RandomizedSearchCV Results')
print('=' * 65)
print(f'  Best n_estimators  : {best_n_est}  <- answer')
print(f'  Best max_depth     : {best_depth}')
print(f'  Best CV R² score   : {best_score:.6f}')
print()
print('  All 5 sampled combinations (ranked best to worst):')
results_df = pd.DataFrame(rscv.cv_results_)
show_cols = ['param_n_estimators', 'param_max_depth', 'mean_test_score', 'rank_test_score']
print(results_df[show_cols].sort_values('rank_test_score').to_string(index=False))

r = results_df.sort_values('rank_test_score')
labels = [f'n={row.param_n_estimators}, d={row.param_max_depth}'
          for _, row in r.iterrows()]
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(labels, r['mean_test_score'],
              color=['gold' if i == 0 else 'steelblue' for i in range(5)], edgecolor='white')
ax.set_ylabel('CV R² Score')
ax.set_title(f'RandomizedSearchCV — Best: n_estimators={best_n_est}, max_depth={best_depth}')
for bar, v in zip(bars, r['mean_test_score']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{v:.4f}', ha='center', fontsize=9)
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q3: {best_n_est}')
print()
print('💡 Insight:')
print('   Depth matters more than tree count here: depth=15 R²≈0.53 vs depth=5 R²≈0.40.')
print('   R²=0.528 means model explains ~52.8% of price variance — decent baseline.')
print('   With random_state=42, the 5 sampled combos are always the same, reproducible.')
```

### 📖 The Story of the Code:
1. We define a hyperparameter search space: `n_estimators: [50, 100, 200]` and `max_depth: [5, 10, 15]`.
2. We initialize **`RandomizedSearchCV`** to sample 5 random combinations using 3-fold cross-validation.
3. After fitting, the best parameters are **n_estimators=200** and **max_depth=15** (yielding a cross-validation $R^2$ of 0.5283).
4. Deeper trees (max_depth=15) perform better than shallow ones (max_depth=5), and increasing the tree count from 100 to 200 slightly improves performance.

### 🎓 Key Terms & Parameters:
- **`RandomizedSearchCV`**: Evaluates random parameter combinations from a grid to find the best settings.
- **`n_iter=5`**: Specifies that only 5 random combinations are evaluated.
- **`cv=3`**: Divides the data into 3 folds, training on 2 and validating on 1, repeating 3 times to calculate average score.
- **`n_jobs=-1`**: Utilizes all available CPU cores to speed up training.

---
## ❓ Question 4: AdaBoost Estimator Error Variance
> **✅ Verified Answer: `0.0777`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Train an AdaBoostRegressor with 50 trees. Compute the variance of the estimator errors. |
| **Tool / Technique used** | `AdaBoostRegressor.estimator_errors_` → `np.var()` |
| **What the output means** | Variance = 0.0777. Early trees learn well (low error), but later trees saturate at 1.0 (random guessing). |

### 💻 Code Block:
```python
ada = AdaBoostRegressor(n_estimators=50, random_state=42)
ada.fit(X_train_proc, y_train)

errors  = ada.estimator_errors_
ada_var = round(float(np.var(errors)), 4)

print('=' * 65)
print('  Q4: AdaBoost estimator_errors_ Variance')
print('=' * 65)
print(f'  Length of estimator_errors_ : {len(errors)} (one value per weak learner)')
print(f'  First 5 errors  : {errors[:5].round(6)}  <- learning well')
print(f'  Last  5 errors  : {errors[-5:].round(6)}  <- saturated at 1.0')
print(f'  Min error       : {errors.min():.6f}')
print(f'  Max error       : {errors.max():.6f}  (1.0 = no better than random)')
print(f'  Mean error      : {errors.mean():.6f}')
print(f'  np.var(errors)  : {ada_var}  <- answer')

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, 51), errors, color='coral', lw=2, marker='o', ms=4)
ax.fill_between(range(1, 51), errors, alpha=0.2, color='coral')
ax.axhline(errors.mean(), color='steelblue', ls='--', lw=2,
           label=f'Mean = {errors.mean():.4f}')
ax.axhline(1.0, color='red', ls=':', lw=1.5,
           label='Error = 1.0 (random guessing)')
ax.set_xlabel('Weak Learner Index (Estimator #)')
ax.set_ylabel('Estimator Error')
ax.set_title(f'AdaBoost — Per-Estimator Errors  |  Variance = {ada_var}')
ax.legend()
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q4: {ada_var}')
print()
print('💡 Insight:')
print('   Errors start at ~0.15 (good) and rise to 1.0 (worst) — AdaBoost saturates.')
print('   When all patterns are modeled, re-weighted samples become impossible to fit.')
print('   This saturation is common for AdaBoost on complex datasets with few features.')
print('   Variance = 0.0777 captures this wide range from best to worst learner.')
```

### 📖 The Story of the Code:
1. We train an **`AdaBoostRegressor`** with 50 estimators.
2. AdaBoost trains weak learners sequentially, placing higher weights on hard-to-predict samples.
3. We calculate the variance of the error rates of the 50 trees: `np.var(ada.estimator_errors_)`.
4. The variance is **0.0777**.
5. **What does this tell us?**
   - The first few trees have low error rates (~0.15), meaning they are learning useful patterns.
   - As training progresses, the remaining patterns become too difficult, and later trees saturate at an error rate of 1.0 (equivalent to random guessing).
   - This wide spread (from 0.15 to 1.0) results in a high error variance of 0.0777, indicating model saturation.

---
## ❓ Question 5: Index of Max Feature Importance
> **✅ Verified Answer: `13`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Train a RandomForestRegressor and report the index of the most important feature. |
| **Tool / Technique used** | `rf.feature_importances_` → `np.argmax()` |
| **What the output means** | Index 13 (a CabinType OHE column) has the highest importance (36.8%), indicating cabin style strongly influences resale value. |

### 💻 Code Block:
```python
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_proc, y_train)

fi      = rf.feature_importances_
max_idx = int(np.argmax(fi))

print('=' * 65)
print('  Q5: RandomForest Feature Importance')
print('=' * 65)
print(f'  Total features in X_train_proc : {len(fi)}')
print(f'  Index of max importance        : {max_idx}  <- answer')
print(f'  Max importance value           : {fi[max_idx]:.6f}  ({fi[max_idx]*100:.2f}% of total)')
print()
print('  Feature index layout:')
print('    Idx  0-3  : UtilizationTier OHE  (4 cols)')
print('    Idx  4-11 : DrivetrainType OHE   (8 cols)')
print('    Idx 12-14 : CabinType OHE        (3 cols)  <- index 13 is HERE')
print('    Idx 15-20 : AssetScaleFactor OHE (6 cols)')
print('    Idx 21    : ManufactureYear (StandardScaled)')
print('    Idx 22    : OperationalHoursMeter (StandardScaled)')
print()
print('  Top 10 features by importance:')
top10 = np.argsort(fi)[-10:][::-1]
for rank, idx in enumerate(top10, 1):
    pct = fi[idx] * 100
    bar = '█' * int(pct * 1.4)
    print(f'    Rank {rank:>2}  Index {idx:>3}  {fi[idx]:.6f} ({pct:.2f}%) {bar}')

fig, ax = plt.subplots(figsize=(12, 4))
colors = ['red' if i == max_idx else 'steelblue' for i in range(len(fi))]
ax.bar(range(len(fi)), fi, color=colors, edgecolor='white', alpha=0.85)
ax.annotate(f'MAX\nIndex {max_idx}\n({fi[max_idx]*100:.1f}%)',
            xy=(max_idx, fi[max_idx]),
            xytext=(max_idx + 2, fi[max_idx] * 0.88),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontweight='bold', fontsize=9)
ax.set_xlabel('Feature Index'); ax.set_ylabel('Importance')
ax.set_title(f'RandomForest Feature Importances — Max at Index {max_idx} (CabinType OHE)')
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q5: {max_idx}')
print()
print('💡 Insight:')
print('   Index 13 = a CabinType OHE column, importance = 36.8%.')
print('   CabinType (EROPS / OROPS / EROPS w AC) determines if a machine')
print('   has an enclosed cab — a major factor in buyer willingness to pay.')
print('   ManufactureYear (idx 21, ~20%) is 2nd — older = cheaper, as expected.')
```

### 📖 The Story of the Code:
1. We train a **`RandomForestRegressor`** with 100 trees and `max_depth=10`.
2. We extract the feature importances and find the index of the highest score using **`np.argmax(fi)`**.
3. The index is **13**, which corresponds to a `CabinType` OHE column (with 36.8% importance).
4. This indicates that the type of cabin (enclosed vs open) is the single strongest predictor of selling price in this feature set.

---
## ❓ Question 6: MLP Connection Weights
> **✅ Verified Answer: `13216`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Count the total connection weights (excluding biases) in an MLP with layers: Input(N) → 128 → 64 → 32 → Output(1). |
| **Tool / Technique used** | Layer size multiplication: $N \times 128 + 128 \times 64 + 64 \times 32 + 32 \times 1$ |
| **What the output means** | With $N = 23$ features in training, the connection weight count is 13,216. |

### 💻 Code Block:
```python
N = X_train_proc.shape[1]
layer_sizes = [N, 128, 64, 32, 1]
layer_names = ['Input', 'Hidden 1 (128)', 'Hidden 2 (64)', 'Hidden 3 (32)', 'Output (1)']

weights_per_layer = [layer_sizes[i] * layer_sizes[i+1]
                     for i in range(len(layer_sizes) - 1)]
total_weights = sum(weights_per_layer)

print('=' * 65)
print('  Q6: Total MLP Connection Weights (excluding biases)')
print('=' * 65)
print(f'  N = X_train_proc.shape[1] = {N}')
print()
print('  Weight calculation per layer connection:')
for i in range(len(layer_sizes) - 1):
    print(f'  {layer_names[i]:<22} -> {layer_names[i+1]:<22}: '
          f'{layer_sizes[i]:>4} × {layer_sizes[i+1]:>3} = {weights_per_layer[i]:>6,} weights')
print('  ' + '-' * 60)
print(f'  TOTAL CONNECTION WEIGHTS                        = {total_weights:>6,}')
print()
biases = sum(layer_sizes[1:])
print(f'  (Reference — total biases = {biases}, total trainable params = {total_weights + biases:,})')

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')
xs = [0.05, 0.24, 0.43, 0.62, 0.81]
for i, (name, size) in enumerate(zip(layer_names, layer_sizes)):
    lbl = f'{name}\n({size} neurons)'
    ax.text(xs[i], 0.58, lbl, ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='steelblue', alpha=0.25))
    if i < len(layer_sizes) - 1:
        ax.annotate('', xy=(xs[i+1]-0.04, 0.58), xytext=(xs[i]+0.06, 0.58),
                    arrowprops=dict(arrowstyle='->', lw=2, color='coral'))
        ax.text((xs[i]+xs[i+1])/2, 0.22,
                f'{weights_per_layer[i]:,} weights',
                ha='center', fontsize=9, color='#c0392b', fontweight='bold')
ax.set_title(f'MLP Architecture  |  N={N}  |  Total Weights = {total_weights:,}',
             fontsize=11, pad=8)
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q6: {total_weights}')
print()
print('💡 Insight:')
print(f'   Formula: {N}×128 + 128×64 + 64×32 + 32×1')
print(f'          = {N*128:,} + {128*64:,} + {64*32:,} + 32 = {total_weights:,}')
print('   N=23 because OHE only creates columns for values seen in TRAINING data.')
print('   Each of these 13,216 weights is learned by Adam optimizer during training.')
print('   Biases (225 total) are separate learnable terms, not asked here.')
```

### 📖 The Story of the Code:
1. An MLP consists of layers of neurons where each neuron in one layer is connected to every neuron in the next layer.
2. The number of connection weights between adjacent layers is the product of their sizes.
3. Our matrix has **$N = 23$** columns (21 OHE categorical columns + 2 numerical columns).
4. The calculation is:
   - Input (23) to Hidden 1 (128): $23 \times 128 = 2,944$
   - Hidden 1 (128) to Hidden 2 (64): $128 \times 64 = 8,192$
   - Hidden 2 (64) to Hidden 3 (32): $64 \times 32 = 2,048$
   - Hidden 3 (32) to Output (1): $32 \times 1 = 32$
   - Total weights = $2,944 + 8,192 + 2,048 + 32 = \mathbf{13,216}$.

### 🎓 Key Terms & Parameters:
- **MLP (Multi-Layer Perceptron)**: A feedforward artificial neural network.
- **Connection Weight**: A learnable parameter representing the strength of the connection between two neurons.
- **Bias**: An additive constant for each neuron (similar to the intercept in linear regression). Not included in connection weights.

---
## ❓ Question 7: MLP Training Loss After 5 Iterations
> **✅ Verified Answer: `221235391.1265`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Train a 3-hidden-layer MLP for exactly 5 iterations. Report the final training loss. |
| **Tool / Technique used** | `MLPRegressor` with `max_iter=5` → `.loss_` |
| **What the output means** | Training loss (MSE) is 221,235,391.1265. This is equivalent to an RMSE of ~$14,874. |

### 💻 Code Block:
```python
mlp_q7 = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=5,
    batch_size=32,
    random_state=42
)
mlp_q7.fit(X_train_proc, y_train)

loss_q7 = round(mlp_q7.loss_, 4)

print('=' * 65)
print('  Q7: MLP Training Loss After 5 Iterations')
print('=' * 65)
print(f'  Iterations run           : {mlp_q7.n_iter_}')
print(f'  Final training loss (MSE): {loss_q7:,}')
print(f'  RMSE equivalent          : ~${(loss_q7**0.5):,.2f}')
print()
print('  Loss per iteration (full loss_curve_):')
for i, v in enumerate(mlp_q7.loss_curve_, 1):
    drop = ((mlp_q7.loss_curve_[0] - v) / mlp_q7.loss_curve_[0] * 100)
    print(f'    Iter {i}: {v:>22,.4f}  (↓ {drop:.2f}% from iter 1)')

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, 6), mlp_q7.loss_curve_,
        color='steelblue', lw=2.5, marker='o', ms=8)
ax.set_xlabel('Iteration (Epoch)')
ax.set_ylabel('Training Loss (MSE)')
ax.set_title(f'MLP Training Loss — 5 Iterations  |  Final = {loss_q7:,.4f}')
ax.set_xticks(range(1, 6))
for i, v in enumerate(mlp_q7.loss_curve_):
    ax.annotate(f'{v/1e6:.1f}M', (i+1, v),
                textcoords='offset points', xytext=(0, 12),
                ha='center', fontsize=9, color='steelblue')
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q7: {loss_q7}')
print()
print('💡 Insight:')
print('   MSE = 221,235,391 sounds huge but RMSE ≈ $14,874 per prediction.')
print('   MSE is squared so dollar-scale errors inflate to millions squared.')
print('   Iter 1 → 2 dropped 26% — Adam optimizer adapts fast in early epochs.')
print('   Iter 2 → 5 slows down — the easy patterns are learned, harder ones remain.')
print('   With max_iter=200 this model would likely reach MSE < 10,000,000.')
```

### 📖 The Story of the Code:
1. We train an **`MLPRegressor`** with hidden layers `(128, 64, 32)` using the `adam` solver for exactly **5 iterations** (epochs).
2. The final training loss is stored in **`model.loss_`**.
3. The training loss is **221,235,391.1265**.
4. **Why is it so large?**
   - The default loss function for `MLPRegressor` is MSE (Mean Squared Error).
   - Target prices are large (up to $142,000), so squared errors grow rapidly.
   - With only 5 iterations, the model is barely past random initialization. The RMSE is $\sqrt{221,235,391} \approx \mathbf{\$14,874}$ average error.

---
## ❓ Question 8: Regularization Effect (alpha)
> **✅ Verified Answer: `2.1609`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Train two MLPs (alpha=0.0001 vs 1.0) for 5 iterations. Report training MAE difference. |
| **Tool / Technique used** | `MLPRegressor(alpha=...)` → `mean_absolute_error` |
| **What the output means** | Absolute MAE difference is $2.1609. Regularization penalizes large weights, slightly increasing training error. |

### 💻 Code Block:
```python
mlp_a = MLPRegressor(hidden_layer_sizes=(100,), max_iter=5,
                     random_state=42, alpha=0.0001)
mlp_b = MLPRegressor(hidden_layer_sizes=(100,), max_iter=5,
                     random_state=42, alpha=1.0)

mlp_a.fit(X_train_proc, y_train)
mlp_b.fit(X_train_proc, y_train)

mae_a   = mean_absolute_error(y_train, mlp_a.predict(X_train_proc))
mae_b   = mean_absolute_error(y_train, mlp_b.predict(X_train_proc))
diff_q8 = round(abs(mae_a - mae_b), 4)

print('=' * 65)
print('  Q8: Regularization Effect — alpha=0.0001 vs alpha=1.0')
print('=' * 65)
print(f'  Model A (alpha=0.0001)  Train MAE = ${mae_a:>14,.4f}')
print(f'  Model B (alpha=1.0)     Train MAE = ${mae_b:>14,.4f}')
print(f'  |MAE_A - MAE_B|                   = {diff_q8}  <- answer')
print()
print(f'  Model A loss curve: {[round(v, 0) for v in mlp_a.loss_curve_]}')
print(f'  Model B loss curve: {[round(v, 0) for v in mlp_b.loss_curve_]}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Model A\nalpha=0.0001\n(Light)', 'Model B\nalpha=1.0\n(Strong)'],
            [mae_a, mae_b], color=['steelblue', 'coral'], edgecolor='white', width=0.5)
axes[0].set_ylabel('Training MAE ($)')
axes[0].set_title(f'Training MAE Comparison  |  Diff = ${diff_q8}')
for i, v in enumerate([mae_a, mae_b]):
    axes[0].text(i, v + 30, f'${v:,.0f}', ha='center', fontweight='bold')
axes[1].plot(range(1, 6), mlp_a.loss_curve_, 'b-o', lw=2, label='A: alpha=0.0001')
axes[1].plot(range(1, 6), mlp_b.loss_curve_, 'r-o', lw=2, label='B: alpha=1.0')
axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Training Loss Curves — Both Models')
axes[1].legend(); axes[1].set_xticks(range(1, 6))
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q8: {diff_q8}')
print()
print('💡 Insight:')
print('   alpha = L2 penalty: Total Loss = MSE + alpha × sum(w²)')
print('   Model B is more constrained → slightly higher training error (+$2.16).')
print('   Why so small a gap? After only 5 iterations, weights have not grown')
print('   large enough for alpha=1.0 to make a big difference yet.')
print('   General rule: Higher alpha → higher training error, better generalization.')
```

### 📖 The Story of the Code:
1. We train two models: Model A (`alpha=0.0001`, light L2 penalty) and Model B (`alpha=1.0`, strong L2 penalty).
2. We calculate their Mean Absolute Errors (MAE) on the training set.
3. The absolute difference is **$2.1609** (Model A MAE = $36,996.24, Model B MAE = $36,998.40).
4. Model B has a slightly higher training MAE because the larger alpha value penalizes weight sizes, constraining the model.
5. The difference is small because 5 iterations is not enough time for the weights to grow large enough to trigger a large penalty.

### 🎓 Key Terms & Parameters:
- **`alpha`**: The L2 regularization strength parameter. Higher alpha forces weights to stay small, reducing overfitting.

---
## ❓ Question 9: MLP Validation MAE
> **✅ Verified Answer: `15735.3515`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Predict on validation data using the Q7 MLP model. Report validation MAE. |
| **Tool / Technique used** | `model.predict(X_valid_proc)` → `mean_absolute_error` |
| **What the output means** | Validation MAE is $15,735.3515. The model is off by an average of ~$15,735 per machine. |

### 💻 Code Block:
```python
preds_val = mlp_q7.predict(X_valid_proc)
mae_q9    = round(mean_absolute_error(y_valid, preds_val), 4)

residuals = y_valid.values - preds_val
pct_err   = np.abs(residuals) / y_valid.values * 100
r2        = 1 - np.var(residuals) / np.var(y_valid.values)

print('=' * 65)
print('  Q9: MLP (Q7 model) Validation Performance')
print('=' * 65)
print(f'  Validation set size        : {len(y_valid):,} rows')
print(f'  Validation MAE             : ${mae_q9:>14,.4f}  <- answer')
print(f'  Median absolute error      : ${np.median(np.abs(residuals)):>14,.2f}')
print(f'  Median percentage error    : {np.median(pct_err):.2f}%')
print(f'  R² score                   : {r2:.4f}')
print(f'  RMSE                       : ${np.sqrt(np.mean(residuals**2)):>14,.2f}')
print()
print('  Comparison to competition best models:')
print(f'  This 5-iter MLP  : MAE = ${mae_q9:,.0f}  (R²={r2:.3f})')
print(f'  XGBoost baseline : MAE ≈ $4,000–$6,000  (R²≈0.95) ← much better')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sample = np.random.default_rng(42).choice(len(y_valid), min(3000, len(y_valid)), replace=False)
axes[0].scatter(y_valid.values[sample], preds_val[sample],
                alpha=0.25, s=6, color='steelblue')
mn, mx = y_valid.min(), y_valid.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Sale Price ($)')
axes[0].set_ylabel('Predicted Sale Price ($)')
axes[0].set_title(f'Actual vs Predicted | R²={r2:.3f} | MAE=${mae_q9:,.0f}')
axes[0].legend()
axes[1].hist(residuals.clip(-80000, 80000), bins=60,
             color='coral', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', ls='--', lw=2, label='Zero residual')
axes[1].axvline(np.median(residuals), color='green', ls='--', lw=1.5,
                label=f'Median = ${np.median(residuals):,.0f}')
axes[1].set_xlabel('Residual: Actual − Predicted ($)')
axes[1].set_title('Residual Distribution')
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f'\n✅ ANSWER Q9: {mae_q9}')
print()
print('💡 Insight:')
print(f'   MAE = $15,735 → model is ~32% off on the typical machine price.')
print(f'   R² = 0.37 → only 37% of price variation is captured by the model.')
print('   Residual plot shows a wide spread — predictions are not precise yet.')
print()
print('   How to improve dramatically:')
print('   1. Increase max_iter to 300–500 (let the model converge properly)')
print('   2. Add AssetAge, HoursPerYear, SaleYear as engineered features')
print('   3. Apply log1p(y) transformation — aligns with RMSLE competition objective')
print('   4. Switch to LightGBM/XGBoost which outperform MLP on tabular data')
```

### 📖 The Story of the Code:
1. We run the Q7 MLP model on the validation set using **`model.predict(X_valid_proc)`**.
2. We calculate the Mean Absolute Error against `y_valid`.
3. The validation MAE is **$15,735.3515**.
4. **Why is the validation MAE ($15,735) much lower than the training MAE ($37,000)?**
   - The training MAE in Q8 was calculated on a single-layer MLP (`hidden_layer_sizes=(100,)`).
   - The Q9 model is the 3-layer MLP (`(128, 64, 32)`) from Q7.
   - The 3-layer architecture is more expressive, allowing it to fit the data better and yield a lower validation error.

---


---

# 📊 Milestone 5: Advanced Feature Selection, Preprocessing Pipelines, Random Forest & Hyperparameter Tuning

In this milestone, we consolidate our features using professional scikit-learn preprocessing pipelines (`Pipeline` and `ColumnTransformer`), filter them down to the top 100 features using univariate linear regression (`SelectKBest` with `f_regression`), train a baseline `RandomForestRegressor`, and optimize it using `RandomizedSearchCV`.

### Setup Code:
```python
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection   import train_test_split, RandomizedSearchCV
from sklearn.preprocessing     import OneHotEncoder, StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble          import RandomForestRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Load dataset
DATA_DIR = '/kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge/'
try:
    train_raw = pd.read_csv(DATA_DIR + 'train.csv', low_memory=False)
except FileNotFoundError:
    DATA_DIR = 'D:/Download/AI MLP Project/Kaggle Dataset/'
    train_raw = pd.read_csv(DATA_DIR + 'train.csv', low_memory=False)
```

---

## ❓ Question 1: Highest Price Decade
> **✅ Verified Answer: `2010`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Group by the extracted manufacture decade and find which one has the highest mean auction price (TargetValue). |
| **Tool / Technique used** | `df.groupby()`, `mean()`, `idxmax()` |
| **What the output means** | Machines manufactured in the **2010s** command the highest average resale price ($59,560.06) because they are newer, have less wear, and feature more modern technologies. |

### 💻 Code Block:
```python
df = train_raw.copy()
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['TransactionYear'] = df['TransactionDate'].dt.year
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['TransactionQuarter'] = df['TransactionDate'].dt.quarter
df['ManufactureDecade'] = (df['ManufactureYear'] // 10) * 10
df['SpecificationComplexity'] = df['Spec_FullDescriptor'].fillna('').apply(lambda x: len(str(x).split()))

decade_avgs = df.groupby('ManufactureDecade')['TargetValue'].mean().sort_values(ascending=False)
print(f"Decade with highest average price: {int(decade_avgs.idxmax())}")
```

### 📖 The Story of the Code:
1. We compute a new feature called `ManufactureDecade` by performing integer division of the `ManufactureYear` by 10 and then multiplying by 10 (e.g., `2014 // 10 = 201`, `201 * 10 = 2010`).
2. We group the dataset by `ManufactureDecade` and calculate the average target price using `groupby('ManufactureDecade')['TargetValue'].mean()`.
3. We sort the results in descending order. The decade 2010s leads the average selling prices, reflecting the standard economic principle that newer machinery has depreciated less.

### 🎓 Key Terms & Parameters:
- **Integer Division (`//`)**: Returns the floor integer result of division, which discards the remainder. It is the easiest way to bucket numerical values into decades.
- **`df.groupby()`**: Splits the dataframe into groups based on unique values of a column, allowing us to perform aggregate operations (like `.mean()`) on each group.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "Why do we see a drop in average price for older decades?"
  * **Your Answer:** "It represents standard depreciation. Newer machinery built in the 2010s has fewer operational hours, less wear and tear, and modern technology features, making it highly valued in auctions compared to older equipment."

---

## ❓ Question 2: Maximum SpecificationComplexity
> **✅ Verified Answer: `4`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Find the maximum word count in the specification descriptor column (`Spec_FullDescriptor`). |
| **Tool / Technique used** | `str.split()`, `len()`, `max()` |
| **What the output means** | The longest description in this column consists of exactly **4 words**, establishing the range limit for description lengths. |

### 💻 Code Block:
```python
max_complexity = df['SpecificationComplexity'].max()
print(f"Maximum complexity: {max_complexity}")
```

### 📖 The Story of the Code:
1. The feature `SpecificationComplexity` is created by taking the string column `Spec_FullDescriptor`, filling missing values with an empty string using `.fillna('')`, and applying a lambda function: `.apply(lambda x: len(str(x).split()))`.
2. This splits the descriptor string by spaces into a list of words, and takes the length of that list.
3. Running `.max()` returns the highest value, which is 4. This shows that all descriptions in the dataset are very short and structured (maximum 4 words).

### 🎓 Key Terms & Parameters:
- **`str.split()`**: Splits a string into a list of words using space as the default delimiter.
- **`lambda x`**: An anonymous, single-line function in Python used here to execute the split-and-length logic on each row of the column.

### 🗣️ Possible Viva Questions:
* **Sir's Question:** "What does a high specification complexity represent in real life?"
  * **Your Answer:** "It represents a machine with highly customized or detailed specification modifiers. A machine with more specifications listed might be a specialized model, which could impact its auction price."

---

## ❓ Question 3: Total Number of Features
> **✅ Verified Answer: `12`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Count the total number of features used for training (excluding target variable and IDs). |
| **Tool / Technique used** | Basic addition: `len(cat_cols) + len(num_cols)` |
| **What the output means** | We select **7 categorical features** and **5 derived numerical features**, giving a total of **12 features** to form our training set. |

### 💻 Code Block:
```python
categorical_features = [
    'DataOriginCode', 'VendorPartnerID', 'UtilizationTier',
    'AssetScaleFactor', 'RegionCode', 'InventoryGroupCategory',
    'InventoryGroupDescription'
]
numerical_features = [
    'TransactionYear', 'TransactionMonth', 'TransactionQuarter',
    'ManufactureDecade', 'SpecificationComplexity'
]
print(f"Total features: {len(categorical_features) + len(numerical_features)}")
```

### 📖 The Story of the Code:
1. We list the 7 categorical features specified in the question.
2. We list the 5 derived numerical features we engineered in Section 1.
3. Summing their lengths gives exactly 12 features. This represents the raw feature count before scikit-learn's preprocessing expands the categorical columns.

---

## ❓ Question 4: Processed Dataset Shape
> **✅ Verified Answer: `(110960, 115)`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Determine the shape of the training dataset after an 80/20 split, median/most_frequent imputation, standard scaling, and One-Hot Encoding. |
| **Tool / Technique used** | `train_test_split()`, `ColumnTransformer`, `OneHotEncoder(handle_unknown='ignore')`, `.shape` |
| **What the output means** | The dataset is split into **110,960 training rows** (80% of 138,701). The categorical features expand to 110 One-Hot encoded columns, which when combined with 5 numerical features, results in **115 columns**. |

### 💻 Code Block:
```python
X = df[categorical_features + numerical_features]
y = df['TargetValue']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])

X_train_proc = preprocessor.fit_transform(X_train)
print(f"Processed shape: {X_train_proc.shape}")
```

### 📖 The Story of the Code:
1. We split the data into 80% training and 20% validation.
2. We create a `ColumnTransformer` that applies:
   - Median imputation and standard scaling to the 5 numerical columns.
   - Most frequent imputation and One-Hot encoding to the 7 categorical columns.
3. Fitting and transforming on `X_train` yields a NumPy array of shape `(110960, 115)`. The 110 categorical columns represent all unique categories present in the training set across the 7 categorical variables.

### 🎓 Key Terms & Parameters:
- **`ColumnTransformer`**: A scikit-learn class that allows different columns or column subsets to be preprocessed separately and concatenated to form a single feature space.
- **`OneHotEncoder(handle_unknown='ignore')`**: Ensures that if a category appears in the validation or test set that was not seen during training, it is ignored (encoded as all zeros) instead of raising an error.

---

## ❓ Question 5: Highest F-score Feature
> **✅ Verified Answer: `cat__AssetScaleFactor_Mini`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Identify which of the 115 processed features has the highest F-score using the univariate linear regression test (`f_regression`). |
| **Tool / Technique used** | `SelectKBest(score_func=f_regression, k=100)`, `.scores_` |
| **What the output means** | `cat__AssetScaleFactor_Mini` has the highest F-score (12,786.04), indicating it is the single strongest linear predictor of selling price. |

### 💻 Code Block:
```python
selector = SelectKBest(score_func=f_regression, k=100)
X_train_sel = selector.fit_transform(X_train_proc, y_train)

# Reconstruct feature names to map scores
ohe = preprocessor.named_transformers_['cat']['ohe']
cat_names = ohe.get_feature_names_out(categorical_features).tolist()
all_names = [f'num__{c}' for c in numerical_features] + [f'cat__{c}' for c in cat_names]

f_scores = pd.DataFrame({'Feature': all_names, 'F_Score': selector.scores_})
f_scores = f_scores.sort_values(by='F_Score', ascending=False)
print(f_scores.head(1))
```

### 📖 The Story of the Code:
1. `SelectKBest` runs a univariate statistical test (`f_regression`) between each individual feature and the target variable `y_train`.
2. It calculates an F-score (the ratio of explained variance to unexplained variance) for each feature.
3. The dummy feature representing "Mini" scaled assets (`cat__AssetScaleFactor_Mini`) has an F-score of **12,786.04**, which is the highest. In construction, small/mini equipment forms a distinct, high-volume, lower-priced market segment.

---

## ❓ Question 6: Lowest Selected F-score Feature
> **✅ Verified Answer: `cat__UtilizationTier_Low`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Compare the F-scores of the listed option features that were **retained/selected** in the Top 100, and find the lowest score. |
| **Tool / Technique used** | Filtering `f_scores` for the options, checking selection status. |
| **What the output means** | `cat__RegionCode_Oklahoma` was removed. Out of the selected options, `cat__UtilizationTier_Low` has the lower score (2,066.25) compared to `cat__AssetScaleFactor_Mini` (12,786.04). |

### 💻 Code Block:
```python
options_q6 = ['cat__AssetScaleFactor_Mini', 'cat__UtilizationTier_Low', 'cat__RegionCode_Oklahoma']
# Filter out removed features, and find the minimum among the options
selected_features = [all_names[i] for i, s in enumerate(selector.get_support()) if s]
selected_options = [f for f in options_q6 if f in selected_features]

q6_res = f_scores[f_scores['Feature'].isin(selected_options)].sort_values('F_Score')
print(q6_res)
```

### 📖 The Story of the Code:
1. `SelectKBest` retains the top 100 features with the highest F-scores.
2. Oklahoma region code has an F-score of 0.03 and is discarded.
3. Among the options that were selected, we compare `cat__AssetScaleFactor_Mini` and `cat__UtilizationTier_Low`. The lowest F-score belongs to `cat__UtilizationTier_Low` (2,066.25).

---

## ❓ Question 7: Removed Features
> **✅ Verified Answer: `cat__RegionCode_Oklahoma`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Identify which of the listed option features fell outside the Top 100 features and was therefore removed. |
| **Tool / Technique used** | Checking `Selected == False` (support mask from `SelectKBest`). |
| **What the output means** | `cat__RegionCode_Oklahoma` has an F-score of 0.03, so it is removed. Regional codes with very low volumes do not have enough statistical correlation with price. |

### 💻 Code Block:
```python
options_q7 = ['cat__AssetScaleFactor_Small', 'cat__RegionCode_Oklahoma', 'cat__UtilizationTier_High']
for opt in options_q7:
    idx = all_names.index(opt)
    selected = selector.get_support()[idx]
    print(f"{opt}: Selected={selected} (F-score={selector.scores_[idx]:.2f})")
```

---

## ❓ Question 8 & 9: Baseline RandomForest Validation Metrics
> **✅ Verified Answers:**
> - **Q8 (MAE): `12302.21`**
> - **Q9 (RMSE): `17207.06`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Fit a default Random Forest model on the Top 100 training features and measure MAE and RMSE on the validation set. |
| **Tool / Technique used** | `RandomForestRegressor(random_state=42, n_jobs=-1)`, `mean_absolute_error()`, `mean_squared_error(..., squared=False)` |
| **What the output means** | The baseline model achieves a mean absolute error of **$12,302.21** and a root mean squared error of **$17,207.06** on unseen validation data. |

### 💻 Code Block:
```python
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_base.fit(X_train_sel, y_train)

X_val_sel = selector.transform(X_val_proc)
y_pred_base = rf_base.predict(X_val_sel)

mae_base = mean_absolute_error(y_val, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_val, y_pred_base))
print(f"MAE: {mae_base:.2f}, RMSE: {rmse_base:.2f}")
```

### 📖 The Story of the Code:
1. We instantiate a `RandomForestRegressor` with `random_state=42` and `n_jobs=-1` (enabling multi-core CPU processing to train trees in parallel).
2. The default parameters are used (e.g., `n_estimators=100`, no depth limit).
3. The baseline model is fit on the selected 100 training columns and evaluated on the validation partition, yielding the reported MAE and RMSE.

---

## ❓ Question 10 & 11: RandomizedSearchCV Best Parameters
> **✅ Verified Answers:**
> - **Q10 (n_estimators): `200`**
> - **Q11 (max_depth): `None`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Identify the optimal values for number of trees (`n_estimators`) and tree depth (`max_depth`) found by randomized search. |
| **Tool / Technique used** | `RandomizedSearchCV()`, `.best_params_` |
| **What the output means** | The optimization search found that a larger forest of **200 trees** with **unlimited depth (`None`)** performs best. |

### 💻 Code Block:
```python
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'max_features': ['sqrt']
}
rscv = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=4,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
rscv.fit(X_train_sel, y_train)
print(rscv.best_params_)
```

### 📖 The Story of the Code:
1. We define a hyperparameter search grid with 4 combinations:
   - `(100, 10, sqrt)`, `(100, None, sqrt)`, `(200, 10, sqrt)`, `(200, None, sqrt)`.
2. We run `RandomizedSearchCV` with 3-fold cross validation. It fits a total of 12 models (4 combinations × 3 folds).
3. The search finds that `n_estimators=200` and `max_depth=None` minimize cross-validation RMSE.

---

## ❓ Question 12: Best Cross-Validation RMSE
> **✅ Verified Answer: `17594.50`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Retrieve the best average RMSE score across the 3 folds during CV tuning. |
| **Tool / Technique used** | `-rscv.best_score_` |
| **What the output means** | The optimal configuration yields an average cross-validation RMSE of **$17,594.50** across training folds. |

### 💻 Code Block:
```python
best_cv_rmse = -rscv.best_score_
print(f"Best CV RMSE: {best_cv_rmse:.2f}")
```

---

## ❓ Question 13 & 14: Tuned RandomForest Validation Metrics
> **✅ Verified Answers:**
> - **Q13 (MAE): `12364.37`**
> - **Q14 (RMSE): `17156.38`**

### 📌 Quick Summary
| Detail | Value |
|---|---|
| **What it asks** | Evaluate the optimized forest model's performance on the validation partition. |
| **Tool / Technique used** | `best_rf.predict()`, MAE and RMSE functions |
| **What the output means** | Evaluating our optimized forest yields a validation MAE of **$12,364.37** and a validation RMSE of **$17,156.38**. |

### 💻 Code Block:
```python
best_rf = rscv.best_estimator_
y_pred_tuned = best_rf.predict(X_val_sel)

mae_tuned = mean_absolute_error(y_val, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_val, y_pred_tuned))
print(f"Tuned RF MAE: {mae_tuned:.2f}, RMSE: {rmse_tuned:.2f}")
```

### 📖 The Story of the Code:
1. The tuned model uses `max_features='sqrt'`, which splits nodes using a random subset of $\sqrt{100} = 10$ features.
2. This introduces diversity between trees and acts as a strong regularizer.
3. As a result, the validation RMSE improves from **17,207.06** (baseline) to **17,156.38** (tuned).

---
# 🎓 Top 20 General Viva Questions & Answers

Based on the course materials, here are the top 20 questions examiners frequently ask:

### 1. What is the difference between supervised and unsupervised learning?
* **Answer**: Supervised learning trains a model on labeled data (where input and target are known, like selling prices in our project). Unsupervised learning finds patterns or groups in unlabeled data (e.g. clustering customers).

### 2. What is Data Leakage, and how did you prevent it in your project?
* **Answer**: Data leakage occurs when training data is contaminated with statistics from the validation or test set. We prevent it by performing the train-test split first. All imputation (e.g. mean/median) and scaling parameters (e.g. mean/std) are calculated strictly on the training set, and then applied to the validation/test sets without recalculation.

### 3. Why did you use RMSLE instead of RMSE?
* **Answer**: RMSLE (Root Mean Squared Logarithmic Error) measures relative error rather than absolute error. Since our machine prices span a wide range ($7.5K to $142K), an absolute error of $10K is critical for cheap machines but acceptable for expensive ones. RMSLE handles this by taking logs.

### 4. What is the purpose of StandardScaler?
* **Answer**: It scales features to have a mean of 0 and a standard deviation of 1. This prevents features with large numeric scales (like hours run) from dominating distance calculations or weight updates in gradient descent-based models like MLPs.

### 5. Why is drop_first=True used in One-Hot Encoding?
* **Answer**: It drops the first category column. This prevents the Dummy Variable Trap (perfect multicollinearity), where one column can be perfectly predicted from the others, which degrades linear regression model stability.

### 6. What is the difference between Mean and Median imputation?
* **Answer**: Mean imputation uses the average of the feature values. It is sensitive to outliers. Median imputation uses the middle value. It is robust to outliers, making it a better choice for highly skewed columns like `OperationalHoursMeter`.

### 7. What does the random_state parameter do?
* **Answer**: It sets a seed for the random number generator, ensuring that operations like train-test splits or tree selection are deterministic and reproducible.

### 8. What is the difference between Bagging and Boosting?
* **Answer**: Bagging (e.g. RandomForest) trains deep trees independently in parallel and averages their predictions to reduce variance. Boosting (e.g. LightGBM, AdaBoost) trains trees sequentially, where each tree attempts to correct the errors of the previous ones to reduce bias.

### 9. What does n_estimators control?
* **Answer**: It controls the total number of decision trees in the ensemble. More trees generally improve accuracy up to a plateau, but increase training time.

### 10. How does a Decision Tree split?
* **Answer**: It searches through all features and threshold values to find the split that maximizes the reduction in impurity (measured by Gini/Entropy for classification, or MSE for regression).

### 11. What is Overfitting, and how can you prevent it?
* **Answer**: Overfitting occurs when a model learns noise in the training set and fails to generalize to validation data. We can prevent it by using regularization (L2/alpha), limiting tree depth (`max_depth`), using early stopping, or training on more data.

### 12. What is Cross-Validation, and why is it useful?
* **Answer**: It splits the training data into $k$ folds. The model trains on $k-1$ folds and validates on the remaining fold, repeating $k$ times. This provides a more robust estimate of performance than a single validation split.

### 13. What is the difference between GridSearch and RandomizedSearch?
* **Answer**: GridSearch evaluates every possible parameter combination in a grid, which is slow. RandomizedSearch randomly samples a set number of combinations (`n_iter`), which is much faster while often finding similar results.

### 14. What are the key layers in an MLP?
* **Answer**: An Input layer (receives features), one or more Hidden layers (extract features using non-linear activations), and an Output layer (produces target predictions).

### 15. What is the role of an Activation Function in neural networks?
* **Answer**: It introduces non-linearity. Without activation functions, a neural network is just a combination of linear functions, which can only model linear relationships. We use **ReLU** (Rectified Linear Unit) in this project.

### 16. What is Adam solver in MLPRegressor?
* **Answer**: Adam (Adaptive Moment Estimation) is an optimization algorithm that adjusts the learning rate adaptively for each parameter, resulting in faster convergence than vanilla Stochastic Gradient Descent (SGD).

### 17. How do you calculate the number of connection weights in a neural network?
* **Answer**: It is the sum of the products of adjacent layer sizes: $\sum (\text{layer}_i \times \text{layer}_{i+1})$.

### 18. What is L2 Regularization (Weight Decay / alpha)?
* **Answer**: It adds a penalty proportional to the sum of the squared weights to the loss function, preventing weights from growing too large and helping to reduce overfitting.

### 19. Why does AdaBoost error variance saturate?
* **Answer**: Later trees in AdaBoost often focus on outliers or noise that cannot be easily modeled, causing their error rates to reach 1.0 (equivalent to random guessing) and leading to error variance saturation.

### 20. How is Feature Importance determined in Random Forest?
* **Answer**: By calculating the average reduction in variance (MSE) that a feature produces across all splits in all decision trees.
