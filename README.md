# Diabetes Classification (BRFSS 2015) - Tidymodels Project

## Goal
This project asks a practical question with societal and business relevance:

- Can we reliably identify diabetics from a simple questionnaire-style dataset that reflects a real-world class-imbalance problem?
- Which self-reported health indicators carry the strongest signal for classifying diabetics?

The focus is a binary classifier for `diabetes_binary` (0 = non-diabetic, 1 = diabetic).

## Data
- Source: BRFSS 2015 health indicators dataset (CSV files included in the repo).
- Files:
  - `diabetes_binary_health_indicators_BRFSS2015.csv` (unbalanced; used for modeling)
  - `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (balanced; used only for EDA plots where raw frequencies are otherwise hard to compare)
- Key challenge: strong class imbalance in the unbalanced dataset (diabetics are the minority).

## What The Script Does (diabetes__.R)
`diabetes__.R` is runnable top-to-bottom and contains:

1. Data import and inspection (structure, summaries, missingness, class balance).
2. Data cleaning (casting binary indicators to factors; ordered factors for age/income/education/general health).
3. Train/test split (stratified) and class weights:
   - We compute per-class weights on the training set.
   - We store weights in a `case_wt` column using `hardhat::importance_weights()`.
4. EDA (optional Shiny dashboard):
   - Unbalanced vs balanced visual comparisons (binary distributions use the balanced dataset).
   - Correlation-with-outcome bar chart, correlation heatmap with extremes, numeric boxplots, and age/BMI distributions.
   - Note: the `shinyApp(ui, server)` call is commented out so `source()` does not block.
5. A unified preprocessing recipe for tree-based models (`model_recipe`):
   - Converts the 4 ordered predictors (`gen_hlth`, `age`, `education`, `income`) to numeric versions and removes the originals.
   - Dummy encodes remaining factor predictors (binary indicators).
   - Applies Tomek links via `themis::step_tomek()` (to remove ambiguous border cases).
   - Drops zero-variance predictors.
6. Baseline models for signal and interpretability:
   - Logistic regression (`glm`) on the unweighted unbalanced training set.
   - CART (`rpart`) using the recipe.
   - We extract feature signal using:
     - logistic regression z-scores (`broom::tidy()`)
     - CART `variable.importance`
   - A small list of low-signal engineered/dummy features is removed to form `cleaned_recipe`.
7. XGBoost model (main model):
   - Uses `xgboost` via tidymodels.
   - Uses relaxed class weighting in the engine (`scale_pos_weight = sqrt(sub_class_ratio)`).
   - Tunes a hyperparameter grid with cross-validation (`tune_grid()`).
   - Tuning metrics include both:
     - `pr_auc` (yardstick default event level; effectively targets class "0" if levels are c("0","1"))
     - `pr_auc_tune` (custom metric targeting the positive class, i.e., diabetics)
     - `roc_auc`
8. Threshold selection for classification:
   - We do not rely on the default 0.5 cutoff.
   - We sweep thresholds on out-of-fold predictions and select a cutoff (used in evaluation).
   - Current threshold used in the script: `xgb_threshold <- 0.08`.
9. Final evaluation and diagnostics:
   - `eval_metrics()` produces accuracy, precision, recall, F1, ROC-AUC, and PR-AUC plus confusion matrix and ROC curve.
   - We compare train vs test metrics to check overfitting.
   - We compare logistic vs CART vs XGBoost side-by-side.
10. Model interpretation:
   - XGBoost feature importance plot via `vip::vip()`.
   - A single XGBoost tree visualization via `xgboost::xgb.plot.tree()`.

## Results (Test Set)
From `Final run with tomek squared weights.txt` (threshold = 0.08), the test-set metrics were:

- Logistic (glm baseline):
  - accuracy 0.865, precision 0.562, recall 0.152, f_meas 0.239, roc_auc 0.825, pr_auc 0.414
- CART (rpart baseline):
  - accuracy 0.866, precision 0.612, recall 0.100, f_meas 0.172, roc_auc 0.647, pr_auc 0.353
- XGBoost (tuned + thresholded at 0.08):
  - accuracy 0.807, precision 0.381, recall 0.619, f_meas 0.471, roc_auc 0.829, pr_auc 0.430
  - confusion matrix:
    - TN 54832, FP 10669, FN 4044, TP 6560

Interpretation:
- Logistic and CART are stable and interpretable but have low recall for diabetics at the chosen threshold.
- XGBoost increases recall substantially (more diabetics found) at the cost of more false positives.

## Key Issues We Hit (And Fixes)
- Package/version issues:
  - `tidymodels` required `rlang >= 1.1.6`; restarting R and updating packages was necessary.
  - Some packages were missing (e.g., `doParallel`) and had to be installed.
- Metric targeting confusion:
  - `pr_auc()` in yardstick defaults to the first factor level as the "event" unless specified.
  - Because our positive class is "1", we added a custom metric (`pr_auc_tune`) that explicitly targets `event_level = "second"` for tuning.
- Threshold matters:
  - Optimizing PR-AUC or ROC-AUC does not automatically produce a good F1 at threshold 0.5.
  - A lower threshold (0.08) materially improved recall/F1 for diabetics.
- Runtime and memory:
  - Saving out-of-fold predictions (`save_pred = TRUE`) is helpful for threshold selection, but increases memory usage.
  - Large cross-joins (predictions x thresholds) can cause memory errors; threshold sweeps should be kept modest.
- Tomek links integration:
  - `step_tomek()` requires numeric predictors, so factor/ordered variables must be converted (dummy-encoded and/or numeric-encoded) before the Tomek step.

## Reproducibility
- The script sets a seed for major random steps (e.g., train/test split).
- Exact bit-for-bit reproducibility is not guaranteed across machines due to:
  - parallel execution,
  - stochastic model training inside xgboost,
  - potential differences in package versions.

For grading/review, the script is written so another person can run it end-to-end and obtain very similar results when using the same R version and package versions.

## How To Run
Recommended: run in RStudio from the project directory.

- Run the full script:
  - `source("diabetes__.R", echo = TRUE, max.deparse.length = Inf)`

- Save a full console log to a file (non-interactive run):
  - Use `run_with_log.R` (already included) and run it with Rscript.
  - If `Rscript` is not on PATH, use the full path, e.g.:
    - `& "C:\\Program Files\\R\\R-4.4.3\\bin\\Rscript.exe" run_with_log.R`

## Skipping Tuning (Instructor Quick Start)
The script includes a clearly marked commented block for XGBoost manual parameters.

To skip tuning:
- Comment out the tuning block (the script tells you exactly which lines).
- Uncomment the `xgb_manual_params <- tibble(...)` block and the `final_xgb_fit <- ...` lines.
- Keep `xgb_threshold <- 0.08` (or adjust if you want a different precision/recall trade-off).

## Repository Files
- `diabetes__.R`: main script (EDA + modeling + evaluation).
- `run_with_log.R`: helper to run `diabetes__.R` and write output to a timestamped log.
- `Final run with tomek squared weights.txt`: reference run output used for reporting results.
- `Final run without tomek with sqrt weights.txt`: older reference run used for history.
