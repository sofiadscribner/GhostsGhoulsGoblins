library(tidymodels)
library(xgboost)
library(embed)
library(tidyverse)
library(vroom)

# read data
train <- read_csv("train.csv") %>%
  mutate(across(where(is.character), as.factor),
         type = as.factor(type)) %>%
  select(-id)

test <- read_csv("test.csv") %>%
  mutate(across(where(is.character), as.factor))



# --- Recipe (recommended: no normalization for trees) ---
xgb_recipe <- recipe(type ~ ., data = train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# --- Model ---
xgb_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# --- Workflow ---
xgb_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_model)

# --- Cross-validation ---
folds <- vfold_cv(train, v = 5, strata = type)

# --- New grid (optimized) ---
xgb_grid <- grid_latin_hypercube(
  trees(range = c(500, 1500)),           # larger ensemble for stability
  tree_depth(range = c(3, 10)),          # avoid overfitting with shallow trees
  min_n(range = c(2, 30)),               # control leaf size
  learn_rate(range = c(0.01, 0.2)),      # smaller = slower but better generalization
  loss_reduction(range = c(0, 5)),       # gamma: regularization term
  sample_size(range = c(0, 1.0)),      # subsampling to reduce variance
  size = 40                              # 40 trials = solid balance
)

# --- Tuning run ---
xgb_tune <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc)
)

# --- Best parameters ---
best_params <- select_best(xgb_tune, metric = "roc_auc")

# --- Finalize workflow and fit ---
final_xgb_wf <- xgb_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

# --- Predict and save ---
xgb_preds <- predict(final_xgb_wf, new_data = test, type = "class")

xgb_sub <- bind_cols(test %>% select(id), xgb_preds) %>%
  rename(type = .pred_class)

vroom_write(xgb_sub, file = "./XGBFinalPreds.csv", delim = ",")

# SECOND TRY

# recipe 

ggg_recipe <- recipe(type ~ ., data = train) %>% 
  step_zv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  step_normalize(all_numeric_predictors())


# model 

ggg_model <- rand_forest( mtry = tune(), 
                          min_n = tune(), 
                          trees = 500) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

# grid

final_grid <- grid_random(
  mtry(range = c(2, 10)),
  min_n(range = c(1, 15)),
  size = 30
)

# workflow 
ggg_wf <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(ggg_model)

# cross validation 
folds <- vfold_cv(train, v = 5, strata = type)

# tune 
ggg_tune <- ggg_wf %>% 
  tune_grid( resamples = folds, 
             grid = final_grid, 
             metrics = metric_set(accuracy, roc_auc))

# save best params 
best_params <- select_best(ggg_tune, metric = "roc_auc")

# finalize and fit 
final_ggg_wf <- ggg_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = train) 

# predict and submit 
ggg_preds <- predict(final_ggg_wf, new_data = test, type = "class") 

# format for kaggle submission 
ggg_sub <- bind_cols(test %>% select(id), ggg_preds %>% select(.pred_class)) %>% rename(type = .pred_class) 

# save predictions locally 
vroom_write(x=ggg_sub, file="./UpdatedForestPreds.csv", delim=",")

# TRYING ENSEMBLING

rf_probs <- predict(final_ggg_wf, new_data = test, type = "prob")
xgb_probs <- predict(final_xgb_wf, new_data = test, type = "prob")

blended_probs <- (rf_probs + xgb_probs) / 2

# Remove ".pred_" prefix so we just have class names
colnames(blended_probs) <- sub("^\\.pred_", "", colnames(blended_probs))

# For each row, pick the column with the highest probability
blended_preds <- blended_probs %>%
  mutate(.pred_class = colnames(.)[max.col(., ties.method = "first")]) %>%
  select(.pred_class) %>%
  mutate(.pred_class = as.factor(.pred_class))

test_ids <- read_csv("test.csv") %>% select(id)

ggg_sub <- bind_cols(test_ids, blended_preds) %>%
  rename(type = .pred_class)

vroom_write(x = ggg_sub, file = "./BlendedPreds.csv", delim = ",")

# TRYING TO IMPROVE RANDOM FOREST AGAIN

# recipe 

ggg_recipe <- recipe(type ~ ., data = train) %>% 
  step_zv(all_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  step_mutate(
    soul_length = has_soul * hair_length,
    rot_bone = rotting_flesh * bone_length
  )


# model 

ggg_model <- rand_forest( mtry = tune(), 
                          min_n = tune(), 
                          trees = 500) %>% 
  set_engine("ranger", importance = "permutation") %>% 
  set_mode("classification")

# grid

final_grid <- grid_random(
  mtry(range = c(2, 10)),
  min_n(range = c(1, 15)),
  size = 30
)

# workflow 
ggg_wf <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(ggg_model)

# cross validation 
folds <- vfold_cv(train, v = 5, strata = type)

# tune 
ggg_tune <- ggg_wf %>% 
  tune_grid( resamples = folds, 
             grid = final_grid, 
             metrics = metric_set(accuracy, roc_auc))

# save best params 
best_params <- select_best(ggg_tune, metric = "roc_auc")

# finalize and fit 
final_ggg_wf <- ggg_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = train) 

# predict and submit 
ggg_preds <- predict(final_ggg_wf, new_data = test, type = "class") 

# format for kaggle submission 
ggg_sub <- bind_cols(test %>% select(id), ggg_preds %>% select(.pred_class)) %>% rename(type = .pred_class) 

# save predictions locally 
vroom_write(x=ggg_sub, file="./UpdatedForestPreds.csv", delim=",")

# KNN MAYBE

# Recipe
knn_recipe <- recipe(type ~ ., data = train) %>% 
  step_zv(all_predictors()) %>%            # remove zero-variance features
  step_dummy(all_nominal_predictors()) %>% # handle color/factor vars
  step_normalize(all_numeric_predictors()) 

# Model
knn_model <- nearest_neighbor(
  mode = "classification",
  neighbors = tune(),     # how many neighbors to consider
  weight_func = tune(),   # uniform, distance, etc.
  dist_power = tune()     # Minkowski distance parameter (1 = Manhattan, 2 = Euclidean)
) %>%
  set_engine("kknn")

# Grid
knn_grid <- grid_regular(
  neighbors(range = c(1, 50)),
  dist_power(range = c(1, 2)),
  weight_func(values = c("rectangular", "triangular", "gaussian")),
  levels = 5
)

# Workflow
knn_wf <- workflow() %>% 
  add_recipe(knn_recipe) %>% 
  add_model(knn_model)

# Cross-validation
folds <- vfold_cv(train, v = 5, strata = type)

# Tune
knn_tune <- tune_grid(
  knn_wf,
  resamples = folds,
  grid = knn_grid,
  metrics = metric_set(roc_auc)
)

# Best parameters
best_knn <- select_best(knn_tune, metric = "roc_auc")

# Finalize and fit
final_knn_wf <- knn_wf %>%
  finalize_workflow(best_knn) %>%
  fit(data = train)

# Predict and format for submission
knn_preds <- predict(final_knn_wf, new_data = test, type = "class")

knn_sub <- bind_cols(test_ids, knn_preds) %>%
  rename(type = .pred_class)

vroom_write(knn_sub, "./KNN_Preds.csv", delim = ",")

# NAIVE BAYES

# Load required packages
library(tidymodels)
library(naivebayes)   # provides the underlying engine

# Define the model
nb_model <- 
  naive_Bayes(
    smoothness = tune(),
    Laplace = tune()
  ) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_recipe <- recipe(type ~ ., data = train) %>%
  step_normalize(all_numeric_predictors())

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

# Cross-validation
folds <- vfold_cv(train, v = 5, strata = type)

# Tune the hyperparameters
nb_tune <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_regular(
      smoothness(range = c(0, 1)),
      Laplace(range = c(0, 1)),
      levels = 5
    ),
    metrics = metric_set(accuracy)
  )

# Select the best model
best_nb <- select_best(nb_tune, metric = "accuracy")

# Finalize and fit
final_nb_wf <- nb_wf %>%
  finalize_workflow(best_nb) %>%
  fit(data = train)

# Predict on test
nb_preds <- predict(final_nb_wf, new_data = test, type = "class")

# Format submission
nb_sub <- bind_cols(test_ids, nb_preds) %>%
  rename(type = .pred_class)

vroom_write(nb_sub, "NaiveBayesPreds.csv", delim = ",")

# DIFFERENT NB RECIPE

# Recipe with engineered features
nb_recipe <- recipe(type ~ ., data = train) %>%
  step_zv(all_predictors()) %>%                     # remove zero-variance columns
  step_other(color, threshold = 0.05) %>%          # group rare colors into "other"
  step_dummy(all_nominal_predictors()) %>%         # encode categorical variables
  step_mutate(
    soul_hair = has_soul * hair_length,
    rot_bone_ratio = rotting_flesh / (bone_length + 1e-5),
    hair_bone = hair_length / (bone_length + 1e-5)
  ) %>%
  step_normalize(all_numeric_predictors())         # scale numeric predictors

# Model
nb_model <- naive_Bayes(
  smoothness = tune(),
  Laplace = tune()
) %>%
  set_engine("naivebayes", classprior = "uniform") %>%
  set_mode("classification")

# Workflow
nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

# Cross-validation
folds <- vfold_cv(train, v = 5, strata = type)

# Tuning grid
nb_grid <- grid_regular(
  smoothness(range = c(0.001, 1)),
  Laplace(range = c(0.001, 1)),
  levels = 10
)

# Tune
nb_tune <- tune_grid(
  nb_wf,
  resamples = folds,
  grid = nb_grid,
  metrics = metric_set(accuracy)   # or roc_auc if supported multiclass
)

# Select best parameters
best_nb <- select_best(nb_tune, metric = "accuracy")

# Finalize workflow with best parameters
final_nb_wf <- nb_wf %>%
  finalize_workflow(best_nb) %>%
  fit(data = train)

# Predict on test set
nb_preds <- predict(final_nb_wf, new_data = test, type = "class")

# Format for Kaggle submission
nb_sub <- bind_cols(test_ids, nb_preds) %>%
  rename(type = .pred_class)

# Save CSV
vroom_write(nb_sub, "./NaiveBayes_Improved.csv", delim = ",")

