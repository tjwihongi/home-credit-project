# feature_engineering.R
# Feature engineering pipeline for Home Credit project
# Author: [Your Name]
# Date: 2026-02-03

library(tidyverse)

#' Clean DAYS_EMPLOYED anomaly and create employment features
#' @param df Data frame with DAYS_EMPLOYED column
#' @return Data frame with cleaned DAYS_EMPLOYED, EMPLOYMENT_YEARS, and missing indicator
clean_days_employed <- function(df) {
  df |> mutate(
    DAYS_EMPLOYED = if_else(DAYS_EMPLOYED == 365243, NA_real_, DAYS_EMPLOYED),
    EMPLOYMENT_YEARS = if_else(is.na(DAYS_EMPLOYED), NA_real_, -DAYS_EMPLOYED / 365.25),
    DAYS_EMPLOYED_MISSING = is.na(DAYS_EMPLOYED)
  )
}

#' Handle missing EXT_SOURCE variables and create missing indicators
#' @param df Data frame with EXT_SOURCE columns
#' @param train_medians Named vector of medians from train set (optional)
#' @return List: data frame with imputed EXT_SOURCE and missingness indicators, and medians used
handle_ext_source_missing <- function(df, train_medians = NULL) {
  ext_cols <- c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
  if (is.null(train_medians)) {
    medians <- df |> summarise(across(all_of(ext_cols), ~median(., na.rm = TRUE))) |> as.list()
  } else {
    medians <- train_medians
  }
  for (col in ext_cols) {
    miss_col <- paste0(col, "_MISSING")
    df[[miss_col]] <- is.na(df[[col]])
    df[[col]][is.na(df[[col]])] <- medians[[col]]
  }
  list(df = df, medians = medians)
}

#' Create demographic and financial ratio features
#' @param df Application data frame
#' @return Data frame with new features
create_demographic_financial_features <- function(df) {
  df |> mutate(
    AGE_YEARS = -DAYS_BIRTH / 365.25,
    CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
    ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL,
    CREDIT_ANNUITY_RATIO = AMT_CREDIT / AMT_ANNUITY,
    LOAN_TO_VALUE = AMT_CREDIT / AMT_GOODS_PRICE
  )
}

#' Add missing data indicators for selected columns
#' @param df Data frame
#' @param columns Vector of column names
#' @return Data frame with new indicator columns
add_missing_indicators <- function(df, columns) {
  for (col in columns) {
    miss_col <- paste0(col, "_MISSING")
    df[[miss_col]] <- is.na(df[[col]])
  }
  df
}

#' Create binned and interaction features
#' @param df Data frame
#' @return Data frame with new features
create_binned_interaction_features <- function(df) {
  df |> mutate(
    AGE_BIN = cut(AGE_YEARS, breaks = c(20, 30, 40, 50, 60, 70), right = FALSE),
    CREDIT_INCOME_BIN = cut(CREDIT_INCOME_RATIO, breaks = quantile(CREDIT_INCOME_RATIO, probs = seq(0, 1, 0.2), na.rm = TRUE), include.lowest = TRUE),
    AGE_EMPLOYED_INTERACT = AGE_YEARS * EMPLOYMENT_YEARS
  )
}

#' Aggregate previous_application.csv at SK_ID_CURR level
#' @param prev_app Data frame of previous applications
#' @return Data frame with aggregates
aggregate_previous_application <- function(prev_app) {
  prev_app |> group_by(SK_ID_CURR) |> summarise(
    PREV_APP_COUNT = n(),
    PREV_APPROVED = sum(NAME_CONTRACT_STATUS == "Approved", na.rm = TRUE),
    PREV_REFUSED = sum(NAME_CONTRACT_STATUS == "Refused", na.rm = TRUE),
    PREV_APPROVAL_RATE = mean(NAME_CONTRACT_STATUS == "Approved", na.rm = TRUE)
  )
}

#' Aggregate bureau.csv at SK_ID_CURR level
#' @param bureau Data frame of bureau records
#' @return Data frame with aggregates
aggregate_bureau <- function(bureau) {
  bureau |> group_by(SK_ID_CURR) |> summarise(
    BUREAU_COUNT = n(),
    BUREAU_ACTIVE = sum(CREDIT_ACTIVE == "Active", na.rm = TRUE),
    BUREAU_CLOSED = sum(CREDIT_ACTIVE == "Closed", na.rm = TRUE),
    BUREAU_OVERDUE = sum(AMT_CREDIT_SUM_OVERDUE, na.rm = TRUE),
    BUREAU_DEBT_RATIO = sum(AMT_CREDIT_SUM_DEBT, na.rm = TRUE) / sum(AMT_CREDIT_SUM, na.rm = TRUE)
  )
}

#' Aggregate installments_payments.csv at SK_ID_CURR level
#' @param inst_pay Data frame of installment payments
#' @return Data frame with aggregates
aggregate_installments_payments <- function(inst_pay) {
  inst_pay |> mutate(LATE_PAYMENT = (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)) |> 
    group_by(SK_ID_CURR) |> summarise(
      INSTAL_PAYMENT_COUNT = n(),
      INSTAL_LATE_PAYMENTS = sum(LATE_PAYMENT, na.rm = TRUE),
      INSTAL_LATE_PAYMENT_PCT = mean(LATE_PAYMENT, na.rm = TRUE),
      INSTAL_PAYMENT_TREND = mean(AMT_PAYMENT / AMT_INSTALMENT, na.rm = TRUE)
    )
}

#' Main pipeline function
#' @param app_df Application data frame
#' @param prev_app_df Previous application data frame
#' @param bureau_df Bureau data frame
#' @param inst_pay_df Installments payments data frame
#' @param train_medians Optional: medians from train for EXT_SOURCE imputation
#' @return List: processed data frame and medians used
feature_engineering_pipeline <- function(app_df, prev_app_df, bureau_df, inst_pay_df, train_medians = NULL) {
  app_df <- clean_days_employed(app_df)
  ext_result <- handle_ext_source_missing(app_df, train_medians)
  app_df <- ext_result$df
  medians_used <- ext_result$medians
  app_df <- create_demographic_financial_features(app_df)
  app_df <- create_binned_interaction_features(app_df)
  prev_agg <- aggregate_previous_application(prev_app_df)
  bureau_agg <- aggregate_bureau(bureau_df)
  inst_agg <- aggregate_installments_payments(inst_pay_df)
  app_df <- app_df |> 
    left_join(prev_agg, by = "SK_ID_CURR") |> 
    left_join(bureau_agg, by = "SK_ID_CURR") |> 
    left_join(inst_agg, by = "SK_ID_CURR")
  list(df = app_df, medians = medians_used)
}

# Example usage (uncomment and edit paths as needed):
# app_train <- read_csv("application_train.csv")
# prev_app <- read_csv("previous_application.csv")
# bureau <- read_csv("bureau.csv")
# inst_pay <- read_csv("installments_payments.csv")
# result <- feature_engineering_pipeline(app_train, prev_app, bureau, inst_pay)
# processed_train <- result$df
# saveRDS(result$medians, "ext_source_medians.rds")
#
# app_test <- read_csv("application_test.csv")
# prev_app_test <- read_csv("previous_application.csv")
# bureau_test <- read_csv("bureau.csv")
# inst_pay_test <- read_csv("installments_payments.csv")
# medians <- readRDS("ext_source_medians.rds")
# result_test <- feature_engineering_pipeline(app_test, prev_app_test, bureau_test, inst_pay_test, train_medians = medians)
# processed_test <- result_test$df
