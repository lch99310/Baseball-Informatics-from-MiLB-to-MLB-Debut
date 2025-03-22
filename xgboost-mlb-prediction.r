# MLB Debut Prediction: XGBoost Models
# ------------------------------

# Load required libraries (at the beginning of script)
library(tidyverse)
library(tidymodels)
library(xgboost)
library(vip)
library(doParallel)

# Data Preparation
# ------------------------------

# Prepare dataset for XGBoost modeling
prepare_xgboost_data <- function(df_mlb) {
  df_mlb %>% 
    select(-name, -highLevel, -school) %>% 
    fastDummies::dummy_cols(
      select_columns = c('team', 'schooltype', 'bats', 'throws', 'position', 'sch_reg', 'birth_place'),
      remove_selected_columns = TRUE
    )
}

# Apply data preparation
df_mlb_xg <- prepare_xgboost_data(df_mlb)

# Data Splitting
# ------------------------------

# Create reproducible train-test split with stratification
set.seed(123)
df_split_xg <- initial_split(df_mlb_xg, strata = mlb_debut)
df_train_xg <- training(df_split_xg)
df_test_xg <- testing(df_split_xg)

# Create cross-validation folds for model tuning
set.seed(123)
dfa_folds_xgb <- vfold_cv(df_train_xg, v = 5, strata = mlb_debut)

# Model Specification
# ------------------------------

# Create XGBoost model specification with tunable parameters
xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), 
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(), 
  mtry = tune(),
  learn_rate = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  translate()

# Define hyperparameter search space
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df_train_xg),
  learn_rate(),
  size = 5
)

# Helper Functions
# ------------------------------

# Function to tune XGBoost models
tune_xgb_model <- function(workflow, folds, grid) {
  # Set up parallel processing
  doParallel::registerDoParallel(cores = parallel::detectCores())
  
  # Tune model
  set.seed(123)
  results <- tune_grid(
    workflow,
    resamples = folds,
    grid = grid,
    control = control_grid(save_pred = TRUE)
  )
  
  # Stop parallel processing
  doParallel::stopImplicitCluster()
  
  return(results)
}

# Function to finalize and evaluate model
evaluate_xgb_model <- function(workflow, best_params, train_data, test_data, split) {
  # Finalize model with best parameters
  final_workflow <- finalize_workflow(workflow, best_params)
  
  # Fit model on training data
  fit_model <- fit(final_workflow, data = train_data)
  
  # Generate predictions on test data
  results <- predict(fit_model, test_data, type = 'prob') %>% 
    pluck(2) %>% 
    bind_cols(test_data, Predicted_Probability = .) %>% 
    mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
  
  # Calculate confusion matrix
  conf_mat_result <- conf_mat(results, 
                             truth = mlb_debut, 
                             estimate = predictedClass)
  
  # Get metrics
  metrics <- summary(conf_mat_result, event_level = 'second')
  auc <- roc_auc(results, 
                truth = mlb_debut, 
                Predicted_Probability, 
                event_level = 'second')
  
  # Variable importance
  importance <- last_fit(final_workflow, split = split) %>% 
    pluck(".workflow", 1) %>%   
    extract_fit_parsnip() %>% 
    vip::vip(num_features = 10)
  
  # Return results as a list
  return(list(
    model = fit_model,
    results = results,
    conf_mat = conf_mat_result,
    metrics = metrics,
    auc = auc,
    importance = importance
  ))
}

# Function to plot ROC curve
plot_roc <- function(results) {
  roc_curve(results, 
           truth = mlb_debut,
           Predicted_Probability,
           event_level = 'second') %>% 
    ggplot(aes(x = 1 - specificity, y = sensitivity)) +
    geom_path() +
    geom_abline(lty = 3) +
    coord_equal() +
    theme_bw() +
    labs(title = "ROC Curve", x = "False Positive Rate", y = "True Positive Rate")
}

# Function to plot probability vs OPS
plot_prob_vs_ops <- function(results) {
  results %>% 
    ggplot() +
    geom_point(aes(x = ops, y = Predicted_Probability, color = mlb_debut)) +
    labs(x = "OPS", y = "Predicted Probability", color = "MLB") +
    theme_light() +
    scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no")) +
    labs(title = "Predicted Probability vs OPS")
}

# Model 1: XGBoost with All Variables
# ------------------------------

# Create recipe and workflow
recipe_xgb_all <- recipe(mlb_debut ~ ., data = df_train_xg) %>% 
  step_zv(all_predictors())

xgb_workflow_all <- workflow() %>% 
  add_recipe(recipe_xgb_all) %>% 
  add_model(xgb_spec)

# Tune model
xgb_res_all <- tune_xgb_model(xgb_workflow_all, dfa_folds_xgb, xgb_grid)

# Get best parameters
best_auc_all <- select_best(xgb_res_all, "roc_auc")

# Evaluate model
model_all_results <- evaluate_xgb_model(
  xgb_workflow_all, 
  best_auc_all, 
  df_train_xg, 
  df_test_xg, 
  df_split_xg
)

# Model 2: XGBoost with LASSO Selected Variables (All Data)
# ------------------------------

# Create recipe and workflow
recipe_xgb_lasso4all <- recipe(mlb_debut ~ ., data = df_train_xg) %>% 
  step_rm(weight, year, round, g, r, b2, bb, pa, obp, slg, hr_ab, iso) %>% 
  step_zv(all_predictors())

xgb_workflow_lasso4all <- workflow() %>% 
  add_recipe(recipe_xgb_lasso4all) %>% 
  add_model(xgb_spec)

# Tune model
xgb_res_lasso4all <- tune_xgb_model(xgb_workflow_lasso4all, dfa_folds_xgb, xgb_grid)

# Get best parameters
best_auc_lasso4all <- select_best(xgb_res_lasso4all, "roc_auc")

# Evaluate model
model_lasso4all_results <- evaluate_xgb_model(
  xgb_workflow_lasso4all, 
  best_auc_lasso4all, 
  df_train_xg, 
  df_test_xg, 
  df_split_xg
)

# Model 3: XGBoost with Expert-Selected Variables
# ------------------------------

# Create recipe and workflow with expert-selected variables
recipe_xgb_selected_lch <- recipe(
  mlb_debut ~ bats_B + bats_L + bats_R + age + bmi + round + overall_pick + 
  team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + 
  team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + 
  team_MIA + team_MIL + team_MIN + team_MON + team_NYA + team_NYN + team_OAK + 
  team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + 
  team_TEX + team_TOR + team_WAS + avg + obp + slg + ops + iso + bb_so + hr_ab + 
  sbr + ab, 
  data = df_train_xg
) %>% 
  step_zv(all_predictors())

xgb_workflow_selected_lch <- workflow() %>% 
  add_recipe(recipe_xgb_selected_lch) %>% 
  add_model(xgb_spec)

# Tune model
xgb_res_selected_lch <- tune_xgb_model(xgb_workflow_selected_lch, dfa_folds_xgb, xgb_grid)

# Get best parameters
best_auc_selected_lch <- select_best(xgb_res_selected_lch, "roc_auc")

# Evaluate model
model_selected_lch_results <- evaluate_xgb_model(
  xgb_workflow_selected_lch, 
  best_auc_selected_lch, 
  df_train_xg, 
  df_test_xg, 
  df_split_xg
)

# Model 4: XGBoost with LASSO Selected Variables (LCH)
# ------------------------------

# Create recipe and workflow
recipe_xgb_lasso4lch <- recipe(
  mlb_debut ~ bats_B + bats_L + bats_R + age + bmi + overall_pick + 
  team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + 
  team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + 
  team_MIA + team_MIL + team_MIN + team_MON + team_NYA + team_NYN + team_OAK + 
  team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + 
  team_TEX + team_TOR + team_WAS + avg + slg + ops + bb_so + sbr + ab, 
  data = df_train_xg
) %>% 
  step_zv(all_predictors())

xgb_workflow_lasso4lch <- workflow() %>% 
  add_recipe(recipe_xgb_lasso4lch) %>% 
  add_model(xgb_spec)

# Tune model
xgb_res_lasso4lch <- tune_xgb_model(xgb_workflow_lasso4lch, dfa_folds_xgb, xgb_grid)

# Get best parameters
best_auc_lasso4lch <- select_best(xgb_res_lasso4lch, "roc_auc")

# Evaluate model
model_lasso4lch_results <- evaluate_xgb_model(
  xgb_workflow_lasso4lch, 
  best_auc_lasso4lch, 
  df_train_xg, 
  df_test_xg, 
  df_split_xg
)

# Model Comparison
# ------------------------------

# Compare AUC values
model_comparison <- tibble(
  Model = c("All Variables", "LASSO All", "Expert-Selected", "LASSO LCH"),
  AUC = c(
    model_all_results$auc$.estimate,
    model_lasso4all_results$auc$.estimate,
    model_selected_lch_results$auc$.estimate,
    model_lasso4lch_results$auc$.estimate
  )
)

# Plot model comparison
model_comparison %>%
  ggplot(aes(x = reorder(Model, AUC), y = AUC)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(x = "", y = "AUC", title = "Model AUC Comparison") +
  theme_minimal() +
  geom_text(aes(label = round(AUC, 3)), hjust = -0.2)
