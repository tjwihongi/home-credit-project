# Home Credit Feature Engineering Pipeline

## Overview

This repository contains a function-based R script (`feature_engineering.R`) for cleaning, transforming, and engineering features from the Home Credit application data and supplementary tables. The pipeline is designed for consistency between train and test sets and reflects insights from exploratory data analysis (EDA).

## What does it do?

- Cleans known data issues (e.g., DAYS_EMPLOYED anomaly, missing EXT_SOURCE values)
- Creates demographic and financial ratio features
- Adds missing data indicators and binned/interaction features
- Aggregates supplementary data from `previous_application.csv`, `bureau.csv`, and `installments_payments.csv` at the applicant level
- Ensures train/test consistency by computing medians from train data and reusing them for test data

## How do you run it?

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

## Notes

- Do **not** commit data files, model objects, or large intermediate files. Add these to your `.gitignore`.
- The pipeline is modular and can be extended with additional features as needed.
