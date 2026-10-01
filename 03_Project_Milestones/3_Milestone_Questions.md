# Milestone 3 (T2-2026)

The following questions are part of Milestone 3 of the Machine Learning Practice Project course. Perform the tasks as instructed in the questions. 

> **Note:** The milestone is considered submitted only when you receive the response copy to your email ID. If there are any issues, please reach out to the instructor before the deadline. 
> 
> **Deadline:** July 7th, 2026

---

## Section 1: Feature Engineering (Temporal & Mathematical)

### Question 1 (MCQ): Imputing Missing Numeric Data 
For a highly skewed numeric column like `OperationalHoursMeter`, the mean is heavily distorted by extreme outliers. What is the safest mathematical value to use for imputation in this specific column?
* A) 0
* B) Mean
* C) Median
* D) Forward-Fill

### Question 2 (MCQ): Datetime Extraction
Machine learning models cannot process raw date strings like `1990-01-31`. We must engineer numeric features from the `TransactionDate` column. 

After converting `TransactionDate` to a datetime object and extracting the `SaleYear`, what is the most frequent `SaleYear` (mode) in the training dataset?
* A) 2010
* B) 2009
* C) 1998
* D) 2012

### Question 3 (MCQ): Imputing Missing Categorical Data
Tree-based models can learn from the "absence" of data if we encode it properly. 

If we decide to fill all missing values in the categorical column `CabinType` with the string `"Missing"`, how many total rows will contain the value `"Missing"` in the training dataset?
* A) 65,000
* B) 80,000
* C) 0
* D) 100,500

### Question 4 (MCQ): Dropping Sparse Features
During EDA, we discovered several columns that are almost entirely empty. 

If we set a threshold to drop any column missing more than 80% of its data, exactly how many columns will be removed from the dataset?
* A) 10 columns
* B) 15 columns
* C) 20 columns
* D) 25 columns

### Question 5 (MCQ): Feature Interaction (Intensity)
Feature interactions reveal hidden patterns. Let's create an "Intensity" feature by dividing `OperationalHoursMeter` by `AssetAge` to see how heavily a machine was used per year. 

After creating `HoursPerYear`, what is the median number of hours a machine was used per year? *(Note: Ignore infinite/NaN values for this calculation).*
* A) ~250 hours/year
* B) ~475 hours/year
* C) ~320 hours/year
* D) ~1200 hours/year

### Question 6 (MCQ): Calculating Asset Age
A highly predictive feature for machinery pricing is the age of the asset at the time of sale. To calculate this accurately, we must first remove the anomaly where `ManufactureYear` is equal to 1000 or 1001 (a system default). 

After replacing `1000` and `1001` with `NaN` and calculating `AssetAge = SaleYear - ManufactureYear`, what is the median `AssetAge` in the training data?
* A) 7.0
* B) 10.0
* C) 8.0
* D) 15.0

---

## Section 2: Data Transformation & Encoding

### Question 7 (MCQ): Ordinal Encoding Application
We must convert text categories into numbers. `OrdinalEncoder` assigns a unique integer to each class. 

If we apply an `OrdinalEncoder` to the `UtilizationTier` column (which has 'Low', 'Medium', 'High', and 'Unspecified' before filling NaNs), how many unique integer values will the encoded column contain?
* A) 2
* B) 3
* C) 4
* D) 5

### Question 8 (MCQ): Preventing Data Leakage
To strictly avoid Data Leakage, at which stage of the machine learning pipeline must you apply your Train-Test Split?
* A) After applying `StandardScaler` to the entire dataset.
* B) Before applying any scalers, target encoders, or median imputations.
* C) After the model is fully trained.
* D) During Exploratory Data Analysis.

### Question 9 (MCQ): Train-Test Split Verification
If we define our target variable `y` as the `TargetValue` column, and our features `X` as the remaining dataframe, and perform a standard 80/20 train-test split (`test_size=0.2`), how many rows will be allocated to our `X_train` training set?
* A) 110,960 rows
* B) 27,741 rows
* C) 138,701 rows
* D) 15,000 rows

### Question 10 (MCQ): Log Transforming the Target
The competition evaluates models using Root Mean Squared Logarithmic Error (RMSLE). To optimize for this using standard regression algorithms (which use RMSE), we apply a `log1p` transformation to the `TargetValue`. 

After applying `np.log1p` to the `TargetValue` column, what is the new standard deviation of the transformed target?
* A) ~0.67
* B) ~26361.12
* C) ~0.81
* D) ~10.51

### Question 11 (MCQ): Target Encoding 
Which of the following scenarios is the primary reason a Data Scientist would choose to use Target Encoding instead of One-Hot Encoding?
* A) The feature is a continuous numeric variable like `Age`.
* B) The categorical feature has very high cardinality (e.g., hundreds of unique values like `Spec_BaseClass`).
* C) The target variable is binary (0 or 1).
* D) The dataset has no missing values.

---

## Section 3: Model Building & Evaluation

### Question 12 (MCQ): Hyperparameter Intuition
When initializing a `RandomForestRegressor` or `LGBMRegressor`, what does the `n_estimators` hyperparameter control?
* A) The number of rows used to train the model.
* B) The maximum depth of the individual trees.
* C) The learning rate of the gradient descent.
* D) The total number of individual decision trees built in the ensemble.

### Question 13 (MCQ): Algorithm Selection
Based on our EDA, our dataset contains significant missing values, highly skewed numerical features, and dozens of categorical text columns. 

Which of the following machine learning models handles this specific type of "messy" tabular data best without requiring massive manual scaling and imputation?
* A) Linear Regression
* B) Deep Neural Networks (Multilayer Perceptrons)
* C) Gradient Boosted Trees (LightGBM/XGBoost)
* D) K-Nearest Neighbors (KNN)

### Question 14 (MCQ): Feature Importance
After training a tree-based model, we can plot the `feature_importances_`. What does a high feature importance score mathematically imply?
* A) The feature has a high standard deviation.
* B) The model used that specific feature frequently to make highly accurate splits in the decision trees, meaning it strongly influences the target prediction.
* C) The feature has no missing values.
* D) The feature perfectly correlates with another independent variable.

### Question 15 (MCQ): Understanding Early Stopping
What is the primary benefit of utilizing an "Early Stopping" callback during model training (e.g., `stopping_rounds=100`)?
* A) It makes the model train faster by randomly skipping rows.
* B) It automatically halts training if the validation score stops improving, preventing the model from overfitting to the training data.
* C) It stops the script if the computer runs out of RAM.
* D) It prevents underfitting by forcing the model to reach a specific metric.