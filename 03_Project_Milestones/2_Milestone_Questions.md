# Milestone 2

The following questions are part of Milestone 2 of the Machine Learning Practice Project course. Perform the tasks as instructed in the questions. 

> **Note:** The milestone is considered submitted only when you receive the response copy to your email ID. If there are any issues, please reach out to the instructor before the deadline. 
> 
> **Deadline:** June 29th, 2026

---

## Section 1: Feature Engineering

Create the following new features on the Train Dataset:
* **TransactionYear:** Extracted from `TransactionDate`
* **TransactionQuarter:** Extracted from `TransactionDate`
* **AssetAge:** `TransactionYear` - `ManufactureYear`
* **DescriptorLength:** Stores the number of characters in `Spec_FullDescriptor`
* **HasOperationalHours:** A binary feature. Assign `1` if `OperationalHoursMeter` is available, assign `0` otherwise.
* **HasVariantModifier:** A binary feature. Assign `1` if `Spec_VariantModifier` is available, assign `0` otherwise.

### Question 1 (MCQ)
Which `AssetAge` occurs most frequently in the training dataset?
* A) 5
* B) 6
* C) 4
* D) 7

### Question 2 (MCQ)
Which `TransactionQuarter` has the highest average `TargetValue`?
* A) 1
* B) 2
* C) 3
* D) 4

### Question 3 (MCQ)
What is the maximum value of `DescriptorLength` in the training dataset?
* A) 19
* B) 13
* C) 12
* D) 11

### Question 4 (Numeric)
How many records have `HasOperationalHours = 1` in the training data?

### Question 5 (MCQ)
What percentage of records contain variant information in the training data? *(Select an answer range)*
* A) 37 - 38
* B) 40 - 41
* C) 28 - 29
* D) None of Them

### Question 6 (Numeric)
What is the mean of `ManufactureYear` in the training data? *(Round your answer to 2 decimal places)*

---

## Section 2: Encoding

Perform One-Hot Encoding on the following columns using `drop_first=True`:
* `RegionCode`
* `CabinType`
* `UtilizationTier`

### Question 7 (Numeric)
How many encoded columns are created in the training data after applying One-Hot Encoding to these three features?

---

## Section 3: Correlation Analysis

After creating the `AssetAge` feature, compute the Pearson correlation matrix using all numerical variables.

### Question 8 (MCQ)
Which of the following numerical features has the highest absolute correlation with `TargetValue` in the training data?
* A) ManufactureYear
* B) OperationalHoursMeter
* C) AssetID
* D) ProductConfigID

---

## Section 4: Feature Scaling

Apply `StandardScaler` to the following numerical features after imputing the missing values with the corresponding mean on the train data:
* `ManufactureYear`
* `OperationalHoursMeter`
* `AssetAge`

Split the processed dataset into training and validation sets using:
* `test_size = 0.20`
* `random_state = 42`

### Question 9 (MCQ)
What is the shape of the training dataset?
* A) (110960, 55)
* B) (27645, 53)
* C) (11340, 53)
* D) (27741, 55)

### Question 10 (MCQ)
What is the shape of the validation dataset?
* A) (110960, 55)
* B) (27645, 53)
* C) (11340, 53)
* D) (27741, 55)

---

## Section 5: Model Building

On the processed training data (imputed and scaled in the previous step), train a Linear Regression model using the following predictors:
* `ManufactureYear`
* `AssetAge`

Evaluate the model on the validation dataset.

### Question 11 (MCQ)
What is the $R^2$ score of the Linear Regression model on the validation dataset? 
* A) 0.02 - 0.03
* B) 0.4 - 0.5
* C) 0.2 - 0.3
* D) 0.04 - 0.05