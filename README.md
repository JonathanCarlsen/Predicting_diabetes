################################################################################
################# ITERATIVE MODEL DEVELOPMENT JOURNAL ###################

This document records everything accomplished during the current Codex session: the initial state of the diabetes modeling script, every refactor, each modeling experiment, and the rationale behind the final configuration. The overarching goals were to (1) understand the BRFSS data through reproducible EDA, (2) build a consistent tidymodels workflow, (3) improve recall/F1 on the minority class, and (4) make the script teacher-ready and reproducible.
This is from the last sprint to finish the project. A lot of iterations happened before this too. The core problem was that we didnt use class weights and tomek before and we tuned for AUC. The result was that all our models had the same AUC of around .83. Where the logistic regression was one of the best performing models despite tuning both an xgb model, a randomforest model and a neural network. We managed to drop complexity by removing and redundant features and optimize for F1 score instead.
--------------------------------------------------------------------------------
### 1. Starting Point
- The original `diabetes__.R` script bundled data import, EDA, modeling, and Shiny UI into one large file. Models (logistic regression, random forest, XGBoost, neural net) ran on both balanced/unbalanced datasets, tuned mostly for ROC/AUC, and used bespoke metric code.
- Class imbalance handling was ad hoc: sometimes training on a balanced CSV, other times sampling the unbalanced set manually. Tomek links were applied outside recipes, and the logistic regression still relied on direct `glm()`.
- Evaluation was manual: computing confusion matrices, ROC curves, and metrics individually per model. The script had no easy way to compare models side by side.

--------------------------------------------------------------------------------
### 2. Early Refactors and Metric Handling
- Converted logistic regression to a tidymodels `workflow()` so it could share preprocessing and resampling infrastructure with the other models.
- Refactored `eval_metrics()` to accept any parsnip/workflow object, call `predict(type="prob"/"class")`, and return metrics, confusion matrices, ROC curves, and prediction tibbles. This eliminates one-off code.
- Consolidated EDA into a single Shiny section: applied a BBC-inspired theme, improved layout, and ensured every plot inherits the same color palette.

--------------------------------------------------------------------------------
### 3. Weights, Tomek Links, and Resampling
- Computed class weights (`importance_weights`) on the training split and stored them in a `case_wt` column. Recipes now declare `case_weights = case_wt`, giving every downstream model a `class_weight = 'balanced'` equivalent.
- Instead of keeping Tomek-filtered columns around, embedded `themis::step_tomek()` near the end of the recipe. This ensures Tomek removal respects resamples and case weights.
- Originally we sliced 5% of the training data to accelerate tuning, but folds contained too few positives, leading to NA precision/F1. Increased the slice to 25% (and considered full data) and later added explicit seeds before slice sampling and v-fold creation to keep runs reproducible.

--------------------------------------------------------------------------------
### 4. Unified Feature Engineering
- Built `diab_rec_unified`, which:
  * Encodes ordered factors (age, education, income, general health) into numeric `*_num` columns while leaving the originals for dummy encoding.
  * Adds BMI categories, BMI–age interactions, cardio risk index (sum of blood pressure/cholesterol/heart/stroke), obesity indicator, censored/log-transformed health-day counts, spline terms for age/BMI/phys/mental health, and interactions like `age_num:high_bp_num`.
  * Applies dummy encoding, removes zero-variance predictors, executes `step_tomek()`, and normalizes all predictors.
- Fitted a CART model and extracted `variable.importance`, plus logistic regression z-scores. Printed both tibbles (with `print(n = Inf)`) to inspect feature contributions.
- Created `diab_rec_selected` by removing features with p ≥ 0.05 (logistic) and CART importance < 157. The removed set included high-order polynomials, redundant dummies, and weak engineered terms.

--------------------------------------------------------------------------------
### 5. Modeling Iterations
**Logistic Regression & CART**
- Retained as baselines. Their z-scores and variable importances are stored for reporting. CART also outputs metrics/confusion matrix so it can be compared to other models.

**Random Forest**
- Tuned with `tune_race_anova()` using `mtry` 3–15, `min_n` 2–15, 1000 trees, `class.weights = wt_vec`, `splitrule = "extratrees"`. Racing was preferred over Bayesian optimization for simplicity. Best RF now yields accuracy 0.922, recall 0.506, F1 0.643, ROC AUC 0.913.

**XGBoost**
- Early `tune_grid()` runs produced F1 ≈ 0.04 due to aggressive Tomek filtering, default `scale_pos_weight`, and 0.5 thresholds. We:
  * Widened parameter ranges (tree depth 2–18, min_n 1–30, sample size 0.05–1.0, mtry 2–20) and added tunable `learn_rate`, `loss_reduction`, `scale_pos_weight`.
  * Switched to `tune_bayes()` seeded by a 25-point Latin-hypercube grid, with verbose logging, `no_improve = 10`, and progress updates.
  * Collected predictions from the best resamples, used `yardstick::threshold_perf()` to find the F1-maximizing probability cutoff, and passed that threshold to `eval_metrics()` (which now accepts a `threshold` argument). Falling back to 0.5 if the tuned threshold is `NA`.
  * Seeds are set for the initial grid and Bayesian search to keep runs deterministic.

**Neural Network**
- Initially tuned hidden units (8, 16, 32). To simplify and keep runtime manageable, we fixed the architecture at 10 hidden units, 20 epochs, `penalty = 1e-4`, `learn_rate = 0.01`, `MaxNWts = 5000`, and fit with a seed. Later, once it was clear the NN provided no improvement over logistic regression, we removed it entirely from the pipeline to shorten runtimes and logs.

--------------------------------------------------------------------------------
### 6. Evaluation Upgrades
- `metrics_df` gathers model metrics (logistic, CART, XGB, RF) and pivots them wide for easy comparison. The confusion matrices and ROC curves for each model are printed, and a combined ROC plot (log/cart/xgb/rf) shows relative performance.
- Autoplot outputs are added after each tuning run (XGB, RF) so we can visualize the search path. Warnings about fonts (Helvetica) remain because the Windows box lacks that font; we left them as they don’t impact modeling.
- Added optional manual-parameter blocks (commented) for XGB and RF that can be uncommented to skip tuning when needed.
- Shiny app now focuses solely on the unbalanced dataset, removes unused toggles, includes correlation heatmap and table of extremes, age/BMI histograms, and binary distributions built from the balanced dataset for clearer comparisons.

--------------------------------------------------------------------------------
### 7. Key Learnings
- **Case weights + Tomek** inside recipes keep pipelines consistent, but Tomek can strip borderline positives required by boosting models; RF handles this better due to ensemble diversity.
- **EDA-driven feature selection** (CART variable importance + logistic z-scores) yields a cleaner design matrix, which speeds tuning and improves interpretability.
- **Sampling fraction vs. folds**: shrinking the dataset for speed creates folds with too few positives; better to reduce fold count or use racing/Bayesian tuning on the full data.
- **Bayesian tuning + threshold calibration**: tuning log-loss alone doesn’t maximize F1; calibrating `scale_pos_weight` and the classification threshold is essential for imbalanced data.
- **Explicit seeds** at every stochastic step (slice sampling, vfold CV, tune grid/race/bayes, neural net fit) are required for reproducibility when sharing scripts.

--------------------------------------------------------------------------------
### 8. Current State vs. Start
- Started with a monolithic script, base R logistic regression, inconsistent imbalance handling, and manual metrics.
- Ended with tidy recipes (`diab_rec_unified` and `diab_rec_selected`), Shiny EDA enhancements, weight-aware pipelines, CART/logistic importance diagnostics, reproducibility aids (seeds/manual blocks), Bayesian-tuned XGB with threshold calibration, race-tuned RF, fixed NN, and consolidated evaluation tables. RF is now the leading performer (F1 ≈ 0.64), with logistic/CART serving as interpretable baselines and XGB improving but still under review.

--------------------------------------------------------------------------------
### 9. Next Steps (Future Work)
- Train/tune models on the full weighted training set (no slicing) to give XGB/NN more positives.
- Explore other imbalance strategies (SMOTE, ROSE, themis steps) or tune Tomek aggressiveness per model.
- Automate threshold calibration for every model and compare against cost-sensitive objectives.
- Package major steps into functions/modules, add reproducible logging via `sink()`, and consider saving fitted workflows for deployment.

--------------------------------------------------------------------------------
### 10. Session Timeline (Key Actions)
1. Refactored logistic regression into tidymodels; rewrote `eval_metrics()`.
2. Rebuilt Shiny EDA with correlation heatmap, age/BMI histograms, binary plots, and removed unused options.
3. Embedded case weights and `step_tomek()` into recipes; standardized resampling with seeds.
4. Ran CART/logistic importance analyses and pruned low-signal engineered features.
5. Added manual-parameter comments and reproducibility seeds across all stochastic steps.
6. Tuned RF via racing, removed the neural net after confirming it didn’t outperform logistic regression, and overhauled XGBoost with wider parameter space, Bayesian tuning, tuned `scale_pos_weight`, and threshold calibration.
7. Attempted an even richer XGBoost search (tuning class weights and probability thresholds together). Gaussian-process models repeatedly failed whenever folds contained zero positive predictions, so we reverted to the simpler configuration above. Even with 95% of the data, XGB still struggles to predict positives; we’re experimenting with relaxing class weights (e.g., using `sqrt` of the class ratio) to coax more positive predictions.

This README is meant to be both a narrative log and a technical reference for future iterations. It explains what changed, why it changed, and how each decision affected performance—especially the jump from baseline models to the current high-recall random forest.
