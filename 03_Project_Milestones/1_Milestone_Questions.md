# Milestone 1

The following questions are part of Milestone 1 of the Machine Learning Practice Project course. Perform the tasks as instructed in the questions. 

> **Note:** The milestone is considered submitted only when you receive the response copy to your email ID. If there are any issues, please reach out to the instructor before the deadline. 
> 
> **Deadline:** June 23, 2026

---

## Section 1: Basic Dataset Understanding

### Question 1 (MCQ): Dataset Dimensions 
Before building any models, it is crucial to understand the scale of the data. How many records (rows) are present in the `train.csv` and `test.csv` datasets, respectively?
* A) Train: 100,000 | Test: 25,000
* B) Train: 138,701 | Test: 15,000
* C) Train: 150,000 | Test: 10,000
* D) Train: 138,701 | Test: 25,000

### Question 2 (MCQ): Target Variable Identification 
By doing a strict set difference between the columns in `train.csv` and `test.csv`, which column is correctly identified as the sole target variable we need to predict?
* A) OperationalHoursMeter
* B) TargetValue
* C) TransactionID
* D) AssetID

### Question 3 (Numeric): Feature Data Types
In the train dataset, how many columns are strictly recognized as numerical features (`int64` or `float64`) by default?
* A) 12
* B) 10
* C) 6
* D) 8

---

## Section 2: Missing Value Analysis

### Question 4 (MSQ): Extremely Sparse Columns 
Which of the following configuration columns exhibit a missing value rate higher than 90% (missing over 124,800 rows out of 138,701) in the training set and should likely be dropped or heavily grouped? *(Select all that apply)*
* [ ] A) col18
* [ ] B) col19
* [ ] C) CabinType
* [ ] D) col5

### Question 5 (Numeric): Missing Operational Metrics 
The `OperationalHoursMeter` represents the lifetime active runtime of the machine. What is the approximate percentage of missing values for this critical column in the training dataset?

---

## Section 3: Target Variable Distribution & Anomalies

### Question 6 (MCQ): Target Value Distribution 
Heavy machinery prices are often right-skewed. What is the median of `TargetValue` (transaction price) in the dataset?
* A) $41,521
* B) $30,000
* C) $35,000
* D) $50,000

### Question 7 (Numeric): Manufacture Year Anomaly
When analyzing the `ManufactureYear`, you will find a strange anomaly. What is the most frequently occurring (mode) `ManufactureYear` in the dataset, indicating a default placeholder value used by the data entry system?

### Question 8 (MCQ): Time Span of Transactions 
What is the earliest (minimum) `TransactionDate` recorded in the training dataset?
* A) 1990-01-31
* B) 1989-05-12
* C) 2000-01-01
* D) 1995-12-15

---

## Section 4: Categorical & High Cardinality Profiling

### Question 9 (Numeric): High Cardinality Identification
Machine learning models struggle with high-cardinality categorical features. How many unique string classes exist within the `Spec_BaseClass` column?

### Question 10 (MCQ): Geographical Volume 
Which region accounts for the highest volume of machinery transactions (`RegionCode`) in this dataset?
* A) Texas
* B) California
* C) Florida
* D) North Carolina

### Question 11 (Numeric): Regional Pricing Insights 
For the most frequent region that is identified as Florida, what is the approximate average `TargetValue` for machines sold there?

### Question 12 (MCQ): Asset Utilization 
The `UtilizationTier` metadata column ranks how heavily an asset was used. Based on the value counts in the training set, what is the most common utilization tier?
* A) High
* B) Medium
* C) Low
* D) Unspecified

### Question 13 (MCQ): Operational Correlation
Intuitively, one might assume that machines with higher `OperationalHoursMeter` values would sell for less. What is the Pearson correlation coefficient between `TargetValue` and `OperationalHoursMeter`?
* A) -0.45 (Strong Negative)
* B) -0.15 (Weak Negative)
* C) ~0.002 (Virtually Zero)
* D) +0.20 (Weak Positive)

### Question 14 (MCQ): Primary Equipment Types 
Based on the `FunctionalClassification` column, what is the description of the single most common exact machinery traded on this platform?
* A) Wheel Loader - 150.0 to 175.0 Horsepower
* B) Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons
* C) Skid Steer Loader - 75.0 to 90.0 Horsepower
* D) Track Excavators - Medium