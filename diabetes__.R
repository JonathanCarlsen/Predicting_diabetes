# Hi Qiqi, this script takes a long time to run. Even with the sample size of .25 of the total dataset.
# Therefore i have added commented out manual parameters and a guide so you can skip the tuning.
# All visualizations from the tuning of XGboost and RandomForest are in the appendix.

################################################################################
################# SETUP AND LIBRARIES ##########################################
# We load the packages and set shared plotting defaults.
library(tidyverse)
library(tidymodels)
library(janitor)
library(corrplot)
library(shiny)
library(parsnip)
library(xgboost)
library(finetune)
library(ranger)
library(yardstick)
library(hardhat)
library(doParallel)
library(themis)
library(vip)
library(broom)

bbc_colors <- c(
  "#BB1919", "#0066B3", "#029232ff", "#F6A800", "#D5D5D5")

color_scheme <- function() {
  list(
    scale_fill_manual(values = bbc_colors),
    scale_color_manual(values = bbc_colors)
  )
}

font <- "Helvetica"
bbc_theme <- ggplot2::theme(
  plot.title = ggplot2::element_text(family = font, size = 20, face = "bold", color = "#222222"),
  plot.subtitle = ggplot2::element_text(family = font, size = 16, face = "bold"),
  legend.position = "bottom",
  legend.text = ggplot2::element_text(family = font, size = 16, color = "#222222", hjust = 0),
  legend.background = ggplot2::element_blank(),
  legend.title = ggplot2::element_blank(),
  legend.key = ggplot2::element_blank(),
  axis.text = ggplot2::element_text(family = font, size = 10, color = "#222222"),
  axis.text.x = ggplot2::element_text(margin = ggplot2::margin(5, b = 10)),
  axis.ticks = ggplot2::element_blank(),
  panel.grid.minor = ggplot2::element_blank(),
  panel.grid.major.y = ggplot2::element_line(color = "#cbcbcb"),
  panel.grid.major.x = ggplot2::element_blank(),
  panel.background = ggplot2::element_blank(),
  strip.background = ggplot2::element_rect(fill = NA, colour = NA),
  strip.text = ggplot2::element_text(size = 18, hjust = 0, margin = ggplot2::margin(t = 4, b = 2))
)

theme_set(bbc_theme)

# We define a single evaluation helper so all models are scored the same way.
eval_metrics <- function(fit, new_data, outcome, threshold = 0.5) {
  outcome <- enquo(outcome)
  prob <- predict(fit, new_data = new_data, type = "prob")
  cls <- predict(fit, new_data = new_data, type = "class")
  preds <- bind_cols(new_data, prob, cls) %>%
    mutate(
      .pred_class = factor(
        if_else(.pred_1 >= threshold, "1", "0"),
        levels = c("0", "1")
      )
    )
  list(
    metrics = metric_set(accuracy, precision, recall, f_meas, roc_auc, pr_auc)(
      preds,
      truth = !!outcome,
      estimate = .pred_class,
      .pred_1,
      event_level = "second"
    ),
    confusion_matrix = conf_mat(
      preds,
      truth = !!outcome,
      estimate = .pred_class
    ),
    roc_curve = roc_curve(
      preds,
      truth = !!outcome,
      .pred_1,
      event_level = "second"
    )
  )
}
# We create a custom F1 metric that targets the positive class.
f_meas_pos <- metric_tweak("f_meas_pos", f_meas, event_level = "second")

################################################################################
################# DATA IMPORT AND INITIAL INSPECTION ###########################

# We load both datasets and inspect structure, missingness, and basic distributions.

set.seed(123)

df_unbal <- read_csv("diabetes_binary_health_indicators_BRFSS2015.csv") %>% clean_names()
df_bal <- read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv") %>% clean_names()

str(df_unbal)
summary(df_unbal)
str(df_bal)
summary(df_bal)

colSums(is.na(df_unbal))
colSums(is.na(df_bal))

prop.table(table(df_unbal$diabetes_binary))
prop.table(table(df_bal$diabetes_binary))

df_unbal %>% summarise(across(everything(), n_distinct)) %>% glimpse()
df_unbal %>%
  summarise(across(
    c(bmi, gen_hlth, ment_hlth, phys_hlth, age, education, income),
    ~ list(sort(unique(.x)))
  )) %>%
  glimpse()

################################################################################
################# DATA CLEANING AND TRANSFORMATIONS ############################

# We standardize factor levels, split data, and compute class weights.

df_unbal <- df_unbal %>%
  mutate(
    diabetes_binary = factor(diabetes_binary, levels = c(0, 1)),
    across(
      c(
        high_bp, high_chol, chol_check, smoker, stroke, heart_diseaseor_attack,
        phys_activity, fruits, veggies, hvy_alcohol_consump, any_healthcare,
        no_docbc_cost, diff_walk, sex
      ),
      factor
    ),
    gen_hlth = factor(gen_hlth, levels = 1:5, ordered = TRUE),
    age = factor(age, levels = 1:13, ordered = TRUE),
    education = factor(education, levels = 1:6, ordered = TRUE),
    income = factor(income, levels = 1:8, ordered = TRUE)
  )

split_unbal <- initial_split(df_unbal, prop = 0.7, strata = diabetes_binary)
train_unbal <- training(split_unbal)
test_unbal <- testing(split_unbal)

wt_tbl <- train_unbal %>% count(diabetes_binary) %>% mutate(wt = sum(n) / (n_distinct(diabetes_binary) * n))
wt_vec <- setNames(wt_tbl$wt, wt_tbl$diabetes_binary)

train_unbal_w <- train_unbal %>% mutate(case_wt = hardhat::importance_weights(wt_vec[as.character(diabetes_binary)]))
test_unbal <- test_unbal %>% mutate(case_wt = hardhat::importance_weights(wt_vec[as.character(diabetes_binary)]))

# We create a subsample because of time and compute constraints (adjust the prop for smaller or bigger sample)
train_unbal_sub <- train_unbal_w %>% slice_sample(prop = 0.5)
sub_class_ratio <- train_unbal_sub %>%
  summarise(ratio = sum(diabetes_binary == "0") / sum(diabetes_binary == "1")) %>%
  pull(ratio)

unbal_folds_sub <- vfold_cv(train_unbal_sub, v = 5, strata = diabetes_binary)

################################################################################
################# EXPLORATORY DATA ANALYSIS ####################################

# Shiny dashboard with 6 tabs

ui <- fluidPage(
  tags$head(tags$style(HTML(".tab-content {padding-top: 15px;}"))),
  titlePanel("Exploratory Data Analysis - Diabetes"),
  tabsetPanel(
    id = "eda_tabs",
    tabPanel("Correlation with outcome", plotOutput("cor_outcome_plot", height = "75vh")),
    tabPanel("Numeric boxplots", plotOutput("boxplot_numeric", height = "75vh")),
    tabPanel("Binary distributions", plotOutput("binary_barplot", height = "75vh")),
    tabPanel("Correlation heatmap", fluidRow(
        column(8, plotOutput("corr_heatmap", height = "75vh")),
        column(4, h4("Most extreme correlations"), tableOutput("corr_extremes"))
      )
    ),
    tabPanel("Age distribution", plotOutput("age_distribution_plot", height = "75vh")),
    tabPanel("BMI distribution", plotOutput("bmi_distribution_plot", height = "75vh"))
  )
)

server <- function(input, output, session) {
  numeric_vars_all <- df_unbal %>%
    select(where(is.numeric)) %>%
    keep(~ n_distinct(., na.rm = TRUE) > 2) %>%
    names() %>%
    setdiff("diabetes_binary")
  
  binary_vars_all <- df_unbal %>%
    select(where(~ n_distinct(., na.rm = TRUE) == 2)) %>%
    names() %>%
    setdiff(c(
      "diabetes_binary", "sex", "chol_check", "any_healthcare",
      "smoker", "fruits", "veggies", "no_docbc_cost", "hvy_alcohol_consump"
    ))
  
  data_numeric <- reactive({
    df_unbal %>% mutate(across(where(is.factor), ~ as.numeric(as.character(.x))))
  })
  
  output$cor_outcome_plot <- renderPlot({
    data_numeric() %>%
      summarise(across(-diabetes_binary, ~ cor(.x, diabetes_binary, use = "pairwise.complete.obs"))) %>%
      pivot_longer(everything(), names_to = "variable", values_to = "correlation") %>%
      ggplot(aes(x = reorder(variable, correlation), y = correlation, fill = correlation > 0)) +
      geom_col(show.legend = FALSE) +
      coord_flip() +
      labs(x = "Variable", y = "Correlation with diabetes_binary", title = "Correlation with diabetes_binary (unbalanced data)") +
      color_scheme()
  })
  
  output$boxplot_numeric <- renderPlot({
    selected_numeric <- c("gen_hlth", "phys_hlth")
    
    df_unbal %>%
      mutate(across(all_of(selected_numeric), ~ as.numeric(as.character(.x)))) %>%
      pivot_longer(
        cols = all_of(selected_numeric),
        names_to = "variable",
        values_to = "value"
      ) %>%
      mutate(value = case_when(
        variable %in% c("ment_hlth", "phys_hlth") ~ pmax(pmin(value, 20), 0),
        TRUE ~ value
      )) %>%
      ggplot(aes(
        x = factor(diabetes_binary, labels = c("Non diabetic", "Diabetic")),
        y = value,
        fill = factor(diabetes_binary, labels = c("Non diabetic", "Diabetic"))
      )) +
      geom_boxplot() +
      facet_wrap(vars(variable), scales = "free_y") +
      labs(
        x = "Diabetes status",
        y = "Value",
        fill = "Diabetes",
        title = "Numeric variables by diabetes status (unbalanced data)"
      ) +
      scale_fill_manual(values = bbc_colors[c(1, 2)]) +
      color_scheme()
  })
  
  output$binary_barplot <- renderPlot({
    df_bal %>%
      mutate(diabetes_binary = factor(diabetes_binary, levels = c(0, 1), labels = c("Non diabetic", "Diabetic"))) %>%
      pivot_longer(cols = all_of(binary_vars_all), names_to = "variable", values_to = "value") %>%
      mutate(value = factor(value)) %>%
      ggplot(aes(x = value, fill = diabetes_binary)) +
      geom_bar(position = "dodge") +
      facet_wrap(vars(variable), scales = "free", ncol = 3) +
      labs(x = "", y = "Frequency", fill = "Diabetes", title = "Binary predictors (balanced dataset)") +
      color_scheme()
  })
  
  output$age_distribution_plot <- renderPlot({
    df_diab <- df_unbal %>%
      filter(diabetes_binary == 1) %>%
      mutate(age_num = as.numeric(as.character(age))) %>%
      count(age_num) %>%
      mutate(prop = n / sum(n))
    
    df_overall <- df_unbal %>%
      mutate(age_num = as.numeric(as.character(age))) %>%
      count(age_num) %>%
      mutate(prop = n / sum(n))
    
    age_labels <- c("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", ">80")
    
    ggplot() +
      geom_col(data = df_diab, aes(x = age_num, y = prop), fill = bbc_colors[1], width = 0.8) +
      geom_line(data = df_overall, aes(x = age_num, y = prop), color = bbc_colors[2], linewidth = 1.5) +
      geom_point(data = df_overall, aes(x = age_num, y = prop), color = bbc_colors[2], size = 2) +
      scale_x_continuous(breaks = 1:13, labels = age_labels) +
      scale_y_continuous(labels = scales::percent_format()) +
      labs(x = "Age category", y = "Share of population", title = "Age distribution: diabetics (bars) vs overall (line)") +
      color_scheme()
  })
  
  output$bmi_distribution_plot <- renderPlot({
    df_diab_bmi <- df_unbal %>%
      filter(diabetes_binary == 1) %>%
      mutate(bmi_bin = pmin(pmax(bmi, 15), 55)) %>%
      count(bmi_bin) %>%
      mutate(prop = n / sum(n))
    
    df_overall_bmi <- df_unbal %>%
      mutate(bmi_bin = pmin(pmax(bmi, 15), 55)) %>%
      count(bmi_bin) %>%
      mutate(prop = n / sum(n))
    
    ggplot() +
      geom_col(data = df_diab_bmi, aes(x = bmi_bin, y = prop), fill = bbc_colors[1]) +
      geom_line(data = df_overall_bmi, aes(x = bmi_bin, y = prop), color = bbc_colors[2], linewidth = 1.2) +
      geom_point(data = df_overall_bmi, aes(x = bmi_bin, y = prop), color = bbc_colors[2], size = 1.5) +
      scale_x_continuous(limits = c(15, 55), breaks = seq(15, 55, by = 5)) +
      scale_y_continuous(labels = scales::percent_format()) +
      labs(x = "BMI", y = "Share of population", title = "BMI distribution: diabetics (bars) vs overall (line)") +
      color_scheme()
  })
  
  output$corr_heatmap <- renderPlot({
    corr_mat <- data_numeric() %>%
      select(where(is.numeric)) %>%
      cor(use = "pairwise.complete.obs")
    corrplot::corrplot(
      corr_mat,
      method = "color",
      col = colorRampPalette(c(bbc_colors[2], "white", bbc_colors[1]))(200),
      tl.cex = 0.8,
      number.cex = 0.7,
      order = "hclust",
      addCoef.col = NA
    )
  })
  
  output$corr_extremes <- renderTable({
    corr_extremes <- data_numeric() %>%
      select(where(is.numeric)) %>%
      cor(use = "pairwise.complete.obs") %>%
      as.data.frame() %>%
      rownames_to_column("var1") %>%
      pivot_longer(cols = -var1, names_to = "var2", values_to = "cor") %>%
      filter(var1 < var2) %>%
      arrange(cor)
    
    bind_rows(tail(corr_extremes, 5), head(corr_extremes, 5)) %>%
      arrange(desc(cor)) %>%
      select(`Variable 1` = var1, `Variable 2` = var2, Correlation = cor)
  })
}

# shinyApp(ui, server)

################################################################################
################# FEATURE ENGINEERING ##########################################

# We build the recipe for our classification trees
model_recipe <- recipe(
  diabetes_binary ~ .,
  data = train_unbal_w,
  case_weights = case_wt
) %>%
  step_mutate(
    gen_hlth_num = as.numeric(gen_hlth),
    age_num = as.numeric(age),
    education_num = as.numeric(education),
    income_num = as.numeric(income)
  ) %>%
  step_rm(gen_hlth, age, education, income) %>%
  step_dummy(all_factor_predictors(), one_hot = FALSE) %>%
  step_tomek(diabetes_binary, skip = TRUE) %>%
  step_zv(all_predictors())

# Prep once
rec_prep <- prep(model_recipe, training = train_unbal_w, retain = TRUE)

# Processed training/test sets
train_x <- juice(rec_prep)
test_x  <- bake(rec_prep, new_data = test_unbal)

# We list which columns were created or dropped by the recipe.
new_cols <- setdiff(names(train_x), names(train_unbal_w))
dropped  <- setdiff(names(train_unbal_w), names(train_x))

new_cols
dropped


################################################################################
################# FEATURE IMPORTANCE AND BASIC LOGISTIC REGRESSION and CART ####
# We fit baseline logistic and CART models to get initial signals.
log_spec <- logistic_reg(mode = "classification") %>% set_engine("glm")
log_wf <- workflow() %>% add_formula(diabetes_binary ~ .) %>% add_model(log_spec)
log_fit <- fit(log_wf, data = train_unbal)

log_train_results  <- eval_metrics(log_fit,  new_data = train_unbal,   outcome = diabetes_binary)
log_results <- eval_metrics(fit = log_fit, new_data = test_unbal, outcome = diabetes_binary)

cart_spec <- decision_tree(
  cost_complexity = 0.001,
  tree_depth = 10,
  min_n = 20
) %>%
  set_mode("classification") %>%
  set_engine("rpart", model = TRUE)

cart_wf <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(cart_spec)

cart_fit <- fit(cart_wf, data = train_unbal_w)

cart_train_results <- eval_metrics(cart_fit, new_data = train_unbal,   outcome = diabetes_binary)
cart_results <- eval_metrics(fit = cart_fit, new_data = test_unbal, outcome = diabetes_binary)

# We use logistic regression z-scores and CART feature importance for guidance.
log_feat_importance <- log_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  mutate(
    feature = term,
    z_score = statistic
  ) %>%
  arrange(desc(abs(z_score)))

cart_importance <- cart_fit %>%
  extract_fit_engine() %>%
  pluck("variable.importance") %>%
  enframe(name = "feature", value = "importance") %>%
  arrange(desc(importance))

log_feat_importance %>% print(n = Inf)
cart_importance %>% print(n = Inf)

cart_importance %>%
  mutate(feature = fct_reorder(feature, importance)) %>%
  ggplot(aes(x = feature, y = importance)) +
  geom_col(fill = bbc_colors[2]) +
  coord_flip() +
  labs(
    title = "CART feature importance",
    x = "Feature",
    y = "Importance"
  )

# These features are not signifant at a 5% level and are not important for the CART model. 
# Therefore we drop them.
low_importance_features <- c(
  "fruits_X1",
  "veggies_X1",
  "no_docbc_cost_X1",
  "education_num")

cleaned_recipe <- model_recipe %>%
  step_rm(any_of(low_importance_features))

################################################################################
################# MODELING XGBoost & RandomForest ############################
doParallel::registerDoParallel()
# We tune and fit the XGBoost and Random Forest models.
# We relax the positive class weight to encourage more positive predictions.
pos_weight <- sqrt(sub_class_ratio)

# Step 1: tune learning rate and trees with other fixed parameters.
xgb_spec <- boost_tree(
  trees = 5000,
  learn_rate = 0.01,
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_engine(
    "xgboost",
    scale_pos_weight = pos_weight,
    lambda = tune(),
    alpha  = tune(),
    early_stopping_rounds = 30
  ) %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_recipe(cleaned_recipe) %>%
  add_model(xgb_spec)

xgb_params <- xgb_wf %>%
  extract_parameter_set_dials() %>%
  update(
    tree_depth      = tree_depth(range = c(2L, 4L)),
    min_n           = min_n(range = c(20L, 60L)),
    sample_size     = sample_prop(range = c(0.5, 0.8)),
    mtry            = finalize(mtry(range = c(3L, 10L)), train_unbal),
    loss_reduction  = loss_reduction(range = c(0, 5)),
    lambda          = penalty(range = c(1, 20)),
    alpha           = penalty(range = c(0.5, 10))
  )

xgb_grid <- grid_space_filling(xgb_params, size = 40)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = unbal_folds_sub,
  grid = xgb_grid,
  metrics = metric_set(pr_auc),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

best_xgb <- select_best(xgb_res, metric = "pr_auc")
final_xgb_wf <- finalize_workflow(xgb_wf, best_xgb)
final_xgb_fit <- fit(final_xgb_wf, data = train_unbal_w)
show_best(xgb_res, metric = "pr_auc")

# We plot tuning performance across each hyperparameter.
autoplot(xgb_res, metric = "pr_auc") + ggtitle("XGBoost tuning performance")

# We sweep thresholds using CV predictions to optimize for F1-score.
preds <- collect_predictions(xgb_res) %>%
  select(id, .config, diabetes_binary, .pred_1)

thr_grid <- seq(0.02, 0.22, by = 0.01)

f_by_thr_xgb <- map_dfr(thr_grid, function(thr) {
  preds %>%
    group_by(id, .config) %>%
    summarise(
      f = f_meas_vec(
        truth = diabetes_binary,
        estimate = factor(if_else(.pred_1 >= thr, "1", "0"),
                          levels = c("0", "1")),
        event_level = "second"
      ),
      .groups = "drop"
    ) %>%
    group_by(.config) %>%
    summarise(mean_f = mean(f, na.rm = TRUE), .groups = "drop") %>%
    mutate(.threshold = thr)
})

best_f <- f_by_thr_xgb %>%
  group_by(.config) %>%
  slice_max(mean_f, n = 1) %>%
  arrange(desc(mean_f))

best_f$.threshold[1]

  # We can skip XGBoost tuning by uncommenting the block below.
  # We comment out xgb_res through final_xgb_fit, then use the manual block right before xgb_results.
# xgb_manual_params <- tibble(
#   mtry           = 9L,
#   min_n          = 48L,
#   tree_depth     = 3L,
#   loss_reduction = 4.38,
#   sample_size    = 0.623,
#   lambda         = 8.89e2,
#   alpha          = 17.0
# )
# final_xgb_wf <- finalize_workflow(xgb_wf, xgb_manual_params)
  # final_xgb_fit <- fit(final_xgb_wf, data = train_unbal_w)

xgb_threshold <- 0.08
xgb_train_results <- eval_metrics(
  final_xgb_fit,
  new_data = train_unbal_w,
  outcome = diabetes_binary,
  threshold = xgb_threshold
)
xgb_results <- eval_metrics(
  fit = final_xgb_fit,
  new_data = test_unbal,
  outcome = diabetes_binary,
  threshold = xgb_threshold
)


rm(preds, best_f)
gc()

vip::vip(
  extract_fit_parsnip(final_xgb_fit),
  num_features = 40,
  geom = "col",
  aesthetics = list(fill = bbc_colors[1])
) + ggtitle("XGBoost feature importance")


# Random Forest
rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_mode("classification") %>%
  set_engine(
    "ranger",
    class.weights   = wt_vec,
    importance      = "impurity",
    splitrule       = "extratrees",
    sample.fraction = tune()
  )

rf_wf <- workflow() %>% add_recipe(cleaned_recipe) %>% add_model(rf_spec)

rf_params <- rf_wf %>%
  extract_parameter_set_dials() %>%
  update(
    mtry        = mtry(range = c(3L, 15L)),
    min_n       = min_n(range = c(5L, 15L)),
    sample.fraction = sample_prop(range = c(0.4, 0.9))
  )


rf_grid <- rf_params %>% grid_space_filling(size = 20)
rf_ctrl <- control_race(save_pred = TRUE, parallel_over = "resamples", verbose_elim = TRUE, verbose = TRUE)

# QUICK START comment the tuning section below.
# Comment out rf_res through final_rf_fit, then use the manual block right before rf_results.

rf_res <- tune_race_anova(
  rf_wf,
  resamples = unbal_folds_sub,
  grid = rf_grid,
  control = rf_ctrl,
  metrics = metric_set(pr_auc)
)

plot_race(rf_res)

collect_metrics(rf_res, summarize = FALSE)
best_rf <- select_best(rf_res, metric = "pr_auc")
show_best(rf_res, metric = "pr_auc")
# We plot tuning performance across each hyperparameter.
autoplot(rf_res, metric = "pr_auc") + ggtitle("Random Forest tuning performance")

# Threshold sweeping
preds <- collect_predictions(rf_res) %>%
  select(id, .config, diabetes_binary, .pred_1)

f_by_thr_rf <- map_dfr(thr_grid, function(thr) {
  preds %>%
    group_by(id, .config) %>%
    summarise(
      f = suppressWarnings(
        f_meas_vec(
          truth = diabetes_binary,
          estimate = factor(
            if_else(.pred_1 >= thr, "1", "0"),
            levels = c("0", "1")
          ),
          event_level = "second"
        )
      ),
      .groups = "drop"
    ) %>%
    group_by(.config) %>%
    summarise(mean_f = mean(f, na.rm = TRUE), .groups = "drop") %>%
    mutate(.threshold = thr)
})

rf_best_f <- f_by_thr_rf %>%
  group_by(.config) %>%
  slice_max(mean_f, n = 1, with_ties = FALSE) %>%
  arrange(desc(mean_f))

rf_best_f$.threshold[1]

# (no tuning): uncomment the block below.
# rf_manual_params <- tibble(mtry = 3L, min_n = 11L)
# final_rf_wf <- finalize_workflow(rf_wf, rf_manual_params)
# final_rf_fit <- fit(final_rf_wf, data = train_unbal_w)
# rf_results <- eval_metrics(fit = final_rf_fit, new_data = test_unbal, outcome = diabetes_binary)

rf_threshold <- 0.19
rf_train_results   <- eval_metrics(final_rf_fit,  new_data = train_unbal_w, outcome = diabetes_binary, threshold = rf_threshold)
rf_results <- eval_metrics(fit = final_rf_fit, new_data = test_unbal, outcome = diabetes_binary, threshold = rf_threshold)

vip::vip(
 extract_fit_parsnip(final_rf_fit),
 num_features = 40,
 geom = "col",
 aesthetics = list(fill = bbc_colors[2])
) + ggtitle("RanfomForest feature importance")

################################################################################
################# EVALUATION ###################################################
# Checking for overfitting
# Logistic
bind_rows(
  test  = log_results$metrics,
  train = log_train_results$metrics,
  .id = "split"
) %>% pivot_wider(names_from = split, values_from = .estimate)
# CART
bind_rows(
  test  = cart_results$metrics,
  train = cart_train_results$metrics,
  .id = "split"
) %>% pivot_wider(names_from = split, values_from = .estimate)
# XGboost
bind_rows(
  test  = xgb_results$metrics,
  train = xgb_train_results$metrics,
  .id = "split"
) %>% pivot_wider(names_from = split, values_from = .estimate)
# Random Forest
bind_rows(
  test  = rf_results$metrics,
  train = rf_train_results$metrics,
  .id = "split"
) %>% pivot_wider(names_from = split, values_from = .estimate)

# We compare evaluation metrics, confusion matrices, and ROC curves across models.
metrics_df <- bind_rows(
  Logistic = log_results$metrics,
  CART = cart_results$metrics,
  XGBoost = xgb_results$metrics,
  RandomForest = rf_results$metrics,
  .id = "model"
) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = model, values_from = .estimate)

metrics_df

log_results$confusion_matrix
cart_results$confusion_matrix
xgb_results$confusion_matrix
rf_results$confusion_matrix

bind_rows(
  Logistic = log_results$roc_curve,
  CART = cart_results$roc_curve,
  XGBoost = xgb_results$roc_curve,
  RandomForest = rf_results$roc_curve,
  .id = "model"
) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(alpha = 1.0) +
  coord_equal() +
  labs(
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model",
    title = "ROC Curves: Logistic vs CART vs XGBoost vs Random Forest"
  ) +
  color_scheme()
