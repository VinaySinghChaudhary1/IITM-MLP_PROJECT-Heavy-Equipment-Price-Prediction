# Milestone 5

The following questions are part of Milestone 5 of the Machine Learning Practice Project course. Perform the tasks as instructed in the questions. 

**Note:** The milestone is considered submitted only when you receive the response copy to your email ID. If there are any issues, please reach out to the instructor before the deadline. 

For NAT questions, enter the answer correct to two decimal places.

**Deadline:** July 22, 2026

---

## Section 1: Feature Engineering

Create the following derived features on the Training Data and answer the following questions:
* **TransactionYear** (Extract from `TransactionDate`)
* **TransactionMonth** (Extract from `TransactionDate`)
* **TransactionQuarter** (Extract from `TransactionDate`)
* **ManufactureDecade** (Extract from `ManufactureYear`)
* **SpecificationComplexity** (Number of words present in `Spec_FullDescriptor`)

### Q1. Which ManufactureDecade has the highest average TargetValue?
- [ ] 2010
- [ ] 2000
- [ ] 1990
- [ ] 1980

### Q2. What is the maximum value of SpecificationComplexity? (NAT)
**Your Answer:** ____________________________________

---

## Section 2: Feature Selection

From the Training Data, create a new Training Data using only the following categorical features:
* `DataOriginCode`
* `VendorPartnerID`
* `UtilizationTier`
* `AssetScaleFactor`
* `RegionCode`
* `InventoryGroupCategory`
* `InventoryGroupDescription`

**and**

Use all numerical features except:
* `TransactionID`
* `AssetID`

Answer the following question:

### Q3. What are the number of features in the Training Data?
**Your Answer:** ____________________________________

---

## Section 3: Train-Validation Split

Split the new training data into train and validation splits.
* **Validation Size** = 20%
* **Random State** = 42
* **Important:** Perform the split before any preprocessing to avoid data leakage.

---

## Section 4: Data Preprocessing

On both the Training and the Validation Split, perform the following:

### For the Numerical Features
1. `SimpleImputer` with strategy `"Median"`
2. `StandardScaler`

### For the Categorical Features
1. `SimpleImputer` with strategy `"Most Frequent"`
2. `OneHotEncoder(handle_unknown='ignore')`

*Perform preprocessing using a Pipeline and ColumnTransformer.*

### Q4. What is the shape of the resultant training dataset after preprocessing?
- [ ] (110960, 12)
- [ ] (110960, 115)
- [ ] (110955, 12)
- [ ] (110955, 115)

---

## Section 5: Feature Selection

On the Scaled Training and Validation set, apply `SelectKBest` using `f_regression` and retain the **Top 100 features**.

### Q5. On the processed Training set, which feature has the highest F-score according to SelectKBest?
- [ ] `num__ManufactureDecade`
- [ ] `cat__AssetScaleFactor_Mini`
- [ ] `cat__UtilizationTier_Low`

### Q6. On the processed Training set, which feature has the lowest F-score among the selected features?
- [ ] `cat__AssetScaleFactor_Mini`
- [ ] `cat__UtilizationTier_Low`
- [ ] `cat__RegionCode_Oklahoma`

### Q7. On the processed Training set, which of the following features were removed by SelectKBest?
- [ ] `cat__AssetScaleFactor_Small`
- [ ] `cat__RegionCode_Oklahoma`
- [ ] `cat__UtilizationTier_High`

---

## Section 6: Baseline Model

Train a `RandomForestRegressor` using default parameters and `random_state=42`. 
*Truncate your answers to 2 decimal places where applicable.*

### Q8. What is the validation MAE of the baseline Random Forest model?
**Your Answer:** ____________________________________

### Q9. What is the RMSE on the Validation set using the baseline Random Forest model? 
**Your Answer:** ____________________________________

---

## Section 7: Hyperparameter Tuning

Perform `RandomizedSearchCV` using the following search space:
* `n_estimators`: `[100, 200]`
* `max_depth`: `[10, None]`
* `max_features`: `["sqrt"]`

**Use:**
* 3-fold Cross Validation Scoring
* **Metric** = Negative RMSE
* `n_iter` = 4
* `random_state` = 42

*Truncate your answers to 2 decimal places where applicable.*

### Q10. Which is the value of n_estimators obtained using RandomizedSearchCV?
**Your Answer:** ____________________________________

### Q11. Which is the value of best max_depth obtained using RandomizedSearchCV?
**Your Answer:** ____________________________________

### Q12. What is the best cross-validation RMSE obtained during hyperparameter tuning? 
**Your Answer:** ____________________________________

### Q13. What is the validation MAE of the hyperparameter tuned Random Forest model?
**Your Answer:** ____________________________________

### Q14. What is the validation RMSE of the hyperparameter tuned Random Forest model?
**Your Answer:** ____________________________________
