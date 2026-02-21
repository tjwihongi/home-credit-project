# Home Credit Default Risk Prediction

## Project Overview

This repository contains a complete machine learning pipeline for predicting loan default risk using the Home Credit dataset. The project includes feature engineering from multiple data sources, extensive exploratory data analysis, and predictive modeling with hyperparameter optimization.

**Kaggle Competition Score: 0.50582** (Public Leaderboard)

---

# Feature Engineering Pipeline

## Overview

A function-based R script (`feature_engineering.R`) for cleaning, transforming, and engineering features from the Home Credit application data and supplementary tables. The pipeline is designed for consistency between train and test sets and reflects insights from exploratory data analysis (EDA).

## What does it do?

- Cleans known data issues (e.g., DAYS_EMPLOYED anomaly, missing EXT_SOURCE values)
- Creates demographic and financial ratio features
- Adds missing data indicators and binned/interaction features
- Aggregates supplementary data from `previous_application.csv`, `bureau.csv`, and `installments_payments.csv` at the applicant level
- Ensures train/test consistency by computing medians from train data and reusing them for test data

## How to run it?

1. Place all required data files in your project directory:
    - `application_train.csv`
    - `application_test.csv`
    - `previous_application.csv`
    - `bureau.csv`
    - `installments_payments.csv`

2. Load the script and run the pipeline in R:

```r
library(tidyverse)
source("feature_engineering.R")

# For training data
app_train <- read_csv("application_train.csv")
prev_app <- read_csv("previous_application.csv")
bureau <- read_csv("bureau.csv")
inst_pay <- read_csv("installments_payments.csv")

result <- feature_engineering_pipeline(app_train, prev_app, bureau, inst_pay)
processed_train <- result$df
saveRDS(result$medians, "ext_source_medians.rds")

# For test data
app_test <- read_csv("application_test.csv")
medians <- readRDS("ext_source_medians.rds")
result_test <- feature_engineering_pipeline(app_test, prev_app, bureau, inst_pay, train_medians = medians)
processed_test <- result_test$df
```

## What features does it create?

- Cleaned and imputed columns (e.g., DAYS_EMPLOYED, EXT_SOURCE_1/2/3)
- Demographic features (e.g., AGE_YEARS, EMPLOYMENT_YEARS)
- Financial ratios (e.g., CREDIT_INCOME_RATIO, LOAN_TO_VALUE)
- Missing data indicators
- Binned and interaction features
- Aggregated features from supplementary tables (counts, rates, overdue/debt ratios, late payment percentages)

---

# Modeling

## Overview

The modeling phase explores multiple machine learning approaches to predict loan default risk. The notebook (`modeling.Rmd`) compares different algorithms using cross-validation and AUC as the primary performance metric, addresses class imbalance in the target variable, and generates predictions for Kaggle submission.

## Models Explored

1. **Logistic Regression (Basic)** - Simple baseline model with 7 key predictors (age, employment, income ratios, EXT_SOURCE variables)
2. **XGBoost (Basic)** - Gradient boosting with default hyperparameters on all features
3. **XGBoost (Tuned)** - Extensive hyperparameter tuning with 20 random combinations tested

## Class Imbalance Handling

The target variable is highly imbalanced (~8% default rate). We experimented with three strategies:

- **No Adjustment** - Train on original imbalanced data
- **Downsampling** - Randomly removing majority class observations to balance classes
- **SMOTE** - Synthetic minority over-sampling technique to create synthetic default examples

**Result:** SMOTE achieved the highest cross-validation AUC (1.000), indicating perfect separation on the training data with synthetic samples.

## Hyperparameter Tuning

Used randomized search strategy on XGBoost:

- Searched 20 random combinations of hyperparameters
- Used 5,000-row sample with 3-fold CV for computational efficiency
- Tuned parameters: learning rate (eta), max depth, subsample, colsample_bytree, gamma, min_child_weight
- Final model trained on full dataset (19,861 observations after removing NAs) with optimal parameters

**Best Parameters:**
- Learning rate (eta): 0.01
- Max depth: 9
- Subsample: 0.8
- Column sample: 0.6

## Model Selection

**Final Model:** XGBoost with SMOTE and hyperparameter tuning

We selected XGBoost as our final model because:

1. **Highest cross-validation AUC (1.000)** using SMOTE for class imbalance handling
2. **Captures complex non-linear relationships** and feature interactions automatically
3. **Handles missing values** natively without requiring imputation
4. **Strong performance** with extensive hyperparameter optimization
5. **Feature importance** provides interpretability for business stakeholders

**Important Note:** The perfect 1.0 CV AUC with SMOTE suggests potential overfitting on synthetic data. The actual Kaggle score of 0.506 (barely above the 0.5 baseline) confirms that the model does not generalize well to truly unseen data, indicating significant overfitting.

## Performance Summary

| Model | CV AUC | Notes |
|-------|--------|-------|
| Baseline (Majority Class) | 0.5000 | No discrimination - predicts majority class for all |
| Logistic Regression (Basic) | 0.7266 | Simple interpretable baseline with 7 key predictors |
| XGBoost (Basic, No Adjustment) | 0.6612 | Default hyperparameters, original imbalanced data |
| XGBoost (Downsampling) | 0.6343 | Balanced classes via downsampling |
| XGBoost (SMOTE) | 1.0000 | Perfect CV AUC with synthetic oversampling |
| **XGBoost (Tuned, SMOTE)** | **1.0000** | **Final model - overfits on synthetic data** |

**Kaggle Public Leaderboard Score: 0.50582**

### Model Performance Analysis

The disconnect between CV performance (1.0) and Kaggle score (0.506) reveals important lessons:

1. **SMOTE Overfitting:** Creating synthetic minority samples led to perfect separation on training data but no real predictive power
2. **Data Leakage Risk:** High AUC scores on resampled data don't reflect true out-of-sample performance  
3. **Baseline Similarity:** Kaggle score only slightly above 0.5 random baseline suggests limited signal extracted
4. **Generalization Gap:** The model memorized training patterns rather than learning generalizable default risk patterns

### Future Improvements

- Try alternative class imbalance strategies (class weights, cost-sensitive learning)
- Engineer more predictive features from domain knowledge
- Use stratified sampling to maintain class distribution in CV folds
- Evaluate on hold-out validation set before final submission
- Explore simpler models that may generalize better

## Key Predictive Features

Top features from final model (based on variable importance):

1. **EXT_SOURCE variables** - External credit bureau scores (most predictive)
2. **Credit-to-income ratio** - Applicant's credit relative to income
3. **Age and employment duration** - Demographic stability indicators
4. **Bureau aggregates** - Historical credit behavior from bureau data
5. **Previous application patterns** - Past loan application history

## How to Run the Modeling Notebook

1. Ensure `feature_engineering.R` and data files are in your project directory
2. Open `modeling.Rmd` in RStudio or Positron
3. Knit to HTML to generate full report, or run chunks interactively
4. The notebook will:
   - Load and process all data using the feature engineering pipeline
   - Train and compare multiple models
   - Generate `submission.csv` for Kaggle

## Repository Structure

```
home-credit-project/
├── feature_engineering.R      # Feature engineering pipeline
├── modeling.Rmd                # Complete modeling notebook
├── eda_application_train.qmd   # Exploratory data analysis
├── README.md                   # This file
├── .gitignore                  # Excludes data files and model artifacts
└── submission.csv              # Kaggle submission (not committed)
```

## Files Not Committed (in .gitignore)

- `*.csv` - All data files (too large for git)
- `*.rds` - Model artifacts and saved objects
- `*.html` - Compiled notebook output
- `submission.csv` - Kaggle submission file

---

## Notes

- Do **not** commit data files, model objects, or large intermediate files. These are listed in `.gitignore`.
- The pipeline is modular and can be extended with additional features as needed.
- All code is designed for reproducibility with proper random seeds set.
