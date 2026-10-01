# Heavy Equipment Selling Price Prediction — Kaggle Competition Specification

## IIT Madras BS Degree — Machine Learning Practice (MLP) Project
**Competition Type:** Supervised Regression (Private In-Class Kaggle Competition)  
**Evaluation Metric:** Root Mean Squared Log Error (RMSLE)  
**Passing Cutoff:** RMSLE < 0.20 (Leaderboard eligibility threshold for Level-1 Viva)

---

## 1. Project Overview & Business Problem

This competition provides operational, transactional, and technical insights into how a global platform tracks, structures, and processes heavy industrial machinery and equipment lifecycles. Each record in the dataset captures a single, finalized transaction event, complete with system metadata, geographic indicators, component specifications, and mechanical configurations.

The primary objective is to build a robust, leak-free machine learning regression pipeline to predict the final transaction selling price (`TargetValue`) based on high-cardinality categorical attributes, temporal parameters, equipment operational hours, and mechanical specifications.

---

## 2. Dataset Description

The official dataset consists of four files located in the `01_Competition_Brief_&_Dataset` directory:

| Filename | Description | Dimensions / Size |
|---|---|---|
| `train.csv` | Labeled historical transaction data with target variable | 138,701 rows × 50 columns (~42 MB) |
| `test.csv` | Unlabeled evaluation set for Kaggle leaderboard scoring | 15,000 rows × 49 columns (~4.5 MB) |
| `sample_submission.csv` | Submission template with `TransactionID` and `TargetValue` | 15,000 rows |
| `metadata.csv` | Supplemental system metadata and column descriptions | System descriptors |

---

## 3. Evaluation Metric: Root Mean Squared Log Error (RMSLE)

Submissions are evaluated on **RMSLE** between predicted value and observed ground truth:

$$\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} ( \log(1 + \hat{y}_i) - \log(1 + y_i) )^2 }$$

### Why RMSLE instead of RMSE / MAE?
1. **Scale Invariance:** Heavy equipment selling prices vary from below $10,000 to over $140,000 (over a 14x range). Standard RMSE disproportionately penalizes absolute dollar errors on expensive bulldozers compared to small excavators.
2. **Relative Percentage Penalty:** RMSLE evaluates relative percentage error equally across the spectrum. A 10% error on a $10,000 machine is treated with identical mathematical weight as a 10% error on a $100,000 machine.
3. **Asymmetric Penalty:** Predicting an under-estimate is penalized slightly more heavily than an over-estimate in log space, encouraging models to produce calibrated estimates without collapse.

---

## 4. Strict Academic & Competition Regulations

The IIT Madras MLP evaluation team enforces strict integrity and technology constraints:

### Permitted Libraries
Only standard, interpretable tabular ML libraries are authorized:
- **Core Numerical & Tabular:** `NumPy`, `Pandas`, `SciPy`
- **Machine Learning & Modeling:** `Scikit-learn`, `LightGBM`, `CatBoost`, `XGBoost`, `Imblearn`, `statsmodels`
- **Visualization:** `Matplotlib`, `Seaborn`, `Plotly`, `Wordcloud`
- **Utilities:** Python standard built-in libraries, `re` (regex)

### Banned Libraries & Techniques (Immediate Disqualification)
- **Deep Learning Frameworks:** `PyTorch`, `TensorFlow`, `Keras`
- **NLP / Embeddings:** `NLTK`, `spaCy`, `word2vec`, `textblob`, `transformers`
- **AutoML Frameworks:** Auto-sklearn, TPOT, FLAML, H2O, etc.
- **Serialization Restrictions:** Submitting unverified black-box pickles without raw reproducible code
- **Pretrained Models:** Use of pre-trained external embeddings or models is strictly disallowed

### Anti-Leakage & Code Integrity Rules
- Submitting notebooks different from the registered URL leads to project invalidation.
- All preprocessing, scaling, missing-value imputation, and target encodings must be executed strictly within cross-validation folds using reusable classes or pipelines. Global transformations across train and test partitions result in automatic deduction or failure.
- Plagiarism across students or public sources results in a direct course failure.

---

## 5. Milestone & Viva Voce Examination Structure

To successfully clear the course project, candidates undergo two examination tiers:

### Level-1 Viva (Integrity & Methodology Audit)
- **Eligibility:** Kaggle Public Leaderboard RMSLE < 0.20.
- **Scope:**
  - Complete, reproducible workflow from Exploratory Data Analysis (EDA) to submission.
  - Systematic handling of missing values, high-cardinality encodings, and scaling.
  - Formal comparison of **at least three distinct model families** (e.g., Tree-based ensembles, Regularized Linear Models, Gradient Boosters).
  - Validation strategy proof (demonstrating zero data leakage via K-Fold CV).

### Level-2 Viva (Theoretical & Mathematical Defense)
- **Weightage:** Scored out of 50 marks (Pass: >= 25/50).
- **Core Testing Pillars:**
  1. **Algorithmic Mechanics (5 Components):** For every model utilized, explain:
     - Data representation & assumptions
     - Mathematical hypothesis function / tree split criterion
     - Loss function formulation
     - Optimization algorithm (Gradient Descent, Newton-Raphson, Second-order Taylor approximation in GBDT)
     - Evaluation metrics & validation properties
  2. **Live Programming / Scratch Implementation:** Modifying hyperparameter functions, writing custom loss functions, or explaining internal split logic on the spot.
  3. **Error Analysis & Overfitting Diagnostics:** Explaining bias-variance tradeoff, learning curves, and out-of-fold generalization gaps.
