####################################################################################
# Build Machine Learning pipeline that:
# 1. Imputes missing values (optional).
# 2. Tunes and benchmarks a range of learners.
# 3. Handles imbalanced data in different ways.
# 4. Identifies optimal learner for the task at hand.

# Abbreviations
# 1. td: Tuned. Learner already tuned with optimal hyperparameters, as found empirically by Probst et al. (2009).
# 2. tn: Tuner. Optimal hyperparameters for the learner to be determined within the Tuner.
# 3. raw: Raw dataset in that class imbalances were not treated in any way.
# 4. up: Data upsampling to balance class imbalances.
# 5. down: Data downsampling to balance class imbalances.
# 6. smote: SMOTE method to balance class imbalances.

# References
# Probst et al. (2009). http://jmlr.csail.mit.edu/papers/volume20/18-444/18-444.pdf

# NOTE 1: Probst et al. (2009) calculate three sets of optimal defaults, using three different performance measures:
# Accuracy, ROC AUC and Brier score. Accuracy is the least desirable measure in our case, because we have imbalanced classes.
# ROC AUC isn't that sensitive to imbalanced classes either, but it's based on predicted probabilities than classes.
# The Brier score is supposed to be more sensitive to imbalanced classes, and it's also based on predicted probabilities than classes.
# My proposal: to begin with, use defaults for the Brier score, Table 9 in Probst et al. (2009).
# NOTE 2: This script is for models 'td'. No reference to 'tn' is made below.
####################################################################################

####################################################################################
# We will be using various learners (incl. XGBoost and Random Forest) throughout with different hyperparameter settings.
# Learners are defined below in alphabetical order
####################################################################################
cat('Setting up learners.\n')

# Define Random Forest learner with the optimal hyperparameters of Probst et al.
# Learner will be added to the pipeline later on, in conjunction with and without class balancing.
rf_td <- lrn("classif.ranger", predict_type = "prob")
rf_td$param_set$values <- list(
  num.trees = 198,
  replace = FALSE,
  sample.fraction = 0.667,
  mtry = round(0.666 * (task$ncol - 1)), # NOTE: if feature selection is performed, user must change this manually to reflect the smaller number of features in the new task.
  respect.unordered.factors = 'ignore',
  min.node.size = 1
)

# Define XGBoost learner with the optimal hyperparameters of Probst et al.
# Learner will be added to the pipeline later on, in conjuction with and without class balancing.

# Optimal defaults with Brier score (Table 9, Probst et al. 2009)
xgb_td_brier <- lrn("classif.xgboost", predict_type = 'prob')
xgb_td_brier$param_set$values <- list(
  booster = "gbtree", 
  nrounds = 2563, 
  max_depth = 11, 
  min_child_weight = 1.75, 
  subsample = 0.873, 
  eta = 0.052,
  colsample_bytree = 0.713,
  colsample_bylevel = 0.638,
  lambda = 0.101,
  alpha = 0.894
)

# Optimal defaults with ROC AUC (Table 3, Probst et al. 2009)
xgb_td_roc <- lrn("classif.xgboost", predict_type = "prob")
xgb_td_roc$param_set$values <- list(
  booster = "gbtree", 
  nrounds = 4168, 
  max_depth = 13, 
  min_child_weight = 2.06, 
  subsample = 0.839, 
  eta = 0.018,
  colsample_bytree = 0.752,
  colsample_bylevel = 0.585,
  lambda = 0.982,
  alpha = 1.113
)

# Define graph learners to be inserted in pipeline: up/down/SMOTE/no balancing
# First for Brier score optimals
xgb_td_brier_raw <- GraphLearner$new(
  po_raw %>>%
    po('learner', xgb_td_brier, id = 'xgb_td_brier'),
  predict_type = 'prob'
)
xgb_td_brier_up <- GraphLearner$new(
  po_over %>>%
    po('learner', xgb_td_brier, id = 'xgb_td_brier'),
  predict_type = 'prob'
)

xgb_td_brier_down <- GraphLearner$new(
  po_under %>>%
    po('learner', xgb_td_brier, id = 'xgb_td_brier'),
  predict_type = 'prob'
)

xgb_td_brier_smote <- GraphLearner$new(
  po_smote %>>%
    po('learner', xgb_td_brier, id = 'xgb_td_brier'),
  predict_type = 'prob'
)

# Now for ROC AUC optimals
xgb_td_roc_raw <- GraphLearner$new(
  po_raw %>>%
    po('learner', xgb_td_roc, id = 'xgb_td_roc'),
  predict_type = 'prob'
)
xgb_td_roc_up <- GraphLearner$new(
  po_over %>>%
    po('learner', xgb_td_roc, id = 'xgb_td_roc'),
  predict_type = 'prob'
)

xgb_td_roc_down <- GraphLearner$new(
  po_under %>>%
    po('learner', xgb_td_roc, id = 'xgb_td_roc'),
  predict_type = 'prob'
)

xgb_td_roc_smote <- GraphLearner$new(
  po_smote %>>%
    po('learner', xgb_td_roc, id = 'xgb_td_roc'),
  predict_type = 'prob'
)

learners_td <- list(
  rf_td
)
names(learners_td) <- sapply(learners_td, function(x) x$id)

####################################################################################
# Remove correlated predictors from tasks
source('mlr3_prompt_remove_correlated_features.R')

if (remove_correlated_predictors) {
  po_remove_corr_preds <- po("select", id = 'remove_correlated_features',
     selector = selector_invert(selector_name(unwanted_features)))
} else {
  po_remove_corr_preds <- po("nop")
}



po_date <- po("datefeatures", 
  param_vals = list(year = TRUE, month = TRUE, day_of_month = FALSE,
    week_of_year = FALSE, day_of_year = FALSE, day_of_week = FALSE,
    hour = FALSE, minute = FALSE, second = FALSE, is_day = FALSE))


po_text <- po("textvectorizer", 
  param_vals = list(stopwords_language = "en"),
  selector = selector_name('improve'))

####################################################################################
# Handle missing values
# Different imputation methods will be used depending on number of missing values in a feature
# 1. Detect features with NAs
# 2. Detect features with 'many' NAs (e.g. >= 20% of records)
# 3. Detect features with 'few' NAs (e.g. < 20% of records)
# 4. Impute features with many NAs using the histogram method
# 5. Impute features with few NAs using the median method
# 6. In addition, create new features indicating if now-imputed NA value was present or missing from original dataset
#    This step helps the learner 'understand' value has been imputed
# 7. Put all of the above into a Graph pipeline. Graph will then be added to the GraphLearner
#    To get to grips with Graphs, see comments in section 'PIPELINE FOR TUNED LEARNERS' below
####################################################################################
cat('Preparing pipe operators for missing data imputation.\n')

# Imputes values based on histogram
hist_imp <- po("imputehist")

# Add an indicator column for each feature with missing values
miss_ind <- po("missind") %>>% 
  po("encode") %>>%
  po("select", 
     selector = selector_invert(selector_type(c("factor"))), 
     id = 'dummy_encoding')

impute_data <- po_remove_corr_preds %>>%
  po("copy", 2) %>>%
  gunion(list(hist_imp, miss_ind)) %>>%
  po("featureunion")

impute_data$plot() # This is the Graph we'll add to the pipeline
impute_data$plot(html = TRUE)

####################################################################################
# PIPELINE FOR TUNED LEARNERS ('td')
# Create pipeline as a graph. This way, pipeline can be plotted. Pipeline can then be converted into a learner with GraphLearner$new(pipeline).
# Pipeline is a collection of Graph Learners (type ?GraphLearner in the command line for info).
# Each GraphLearner is a td model (see abbreviations above) with or without class balancing.
# Up/down or no sampling happens within each GraphLearner, otherwise an error during tuning indicates that there are >= 2 data sources.
# Up/down or no sampling within each GraphLearner can be specified by chaining the relevant pipe operators (function po(); type ?PipeOp in command line) with the PipeOp of each learner.
# We are going to build a graph pipeline for the tuned learners ('td').
####################################################################################
cat('Perparing pipeline, parameter sets and resampling strategy.\n')

# Our pipeline
graph_td <- 
  impute_data %>>%
  po("branch", names(learners_td)) %>>% 
  gunion(unname(learners_td)) %>>%
  po("unbranch")

graph_td$plot() # Plot pipeline
graph_td$plot(html = TRUE) # Plot pipeline

pipe_td <- GraphLearner$new(graph_td) # Convert pipeline to learner
pipe_td$predict_type <- 'prob' # Don't forget to specify we want to predict probabilities and not classes.

# Define resampling strategy to be applied within pipeline
# Instantiate it to ensure data split is the same in all pipelines and benchmarking exercises to follow
source('mlr3_prompt_resampling_strategy.R')

source('mlr3_prompt_cv_folds.R')

if (resampling_strategy == "repeated_cv") {
  cv_reps <- cv_folds
  cv <- rsmp("repeated_cv", repeats = cv_reps, folds = cv_folds) # Repeats and folds should be set to 5 (5 x 5 runs) or 10 (10 x 10 runs)
} else if (resampling_strategy == "cv") {
  cv_reps <- 1 # Need that for later
  cv <- rsmp("cv", folds = cv_folds) # Repeats and folds should be set to 5 (5 x 5 runs) or 10 (10 x 10 runs)
}
cv$instantiate(task_train)

# Parameter set of the pipeline
ps_table_td <- as.data.table(pipe_td$param_set)
View(ps_table_td[, 1:4])

ps_class_balancing_td <- ps_table_td$id %>%
  lapply(
    function(x) {
      if (all(grepl('up\\.', x), grepl('\\.ratio', x))) {
        ParamDbl$new(x, 
          lower = log(upsample_ratio / 2), upper = log(upsample_ratio))
      } else if (all(grepl('down\\.', x), grepl('\\.ratio', x))) {
        ParamDbl$new(x, 
          lower = log(1 / downsample_ratio), upper = log(2 / downsample_ratio))
      } else if (grepl('smote\\.', x)) {
        if (grepl('\\.dup_size', x)) {
          ParamInt$new(x, lower = 1, upper = round(upsample_ratio))
        } else if (grepl('\\.K', x)) {
          ParamInt$new(x, lower = 1, upper = round(upsample_ratio))
        }
      }
    }
  )
ps_class_balancing_td <- Filter(Negate(is.null), ps_class_balancing_td)
ps_class_balancing_td <- ParamSet$new(ps_class_balancing_td)

param_set_td <- ParamSetCollection$new(list(
  ParamSet$new(list(pipe_td$param_set$params$branch.selection$clone())), # ParamFct can be copied.
  ps_class_balancing_td
))

# In ML it is common to apply a transformation to the (hyper)parameters to be tuned by the pipeline, to better search the (hyper)parameter space
# Here, we'll apply a trafo to SMOTE's k-nearest neighbours variable K, so that K is converted into 2 ^ K
# This requires a little bit of hacking. First, the resampling strategy needs to be defined first, so I've moved the cv stuff above
# Second, we want to make sure that the trafoed K doesn't exceed the sample size of the training set inside the CV process
# See https://stackoverflow.com/questions/61772147/tuning-smotes-k-with-a-trafo-fails-warningk-should-be-less-than-sample-size/61818298#61818298
smote_k_thresh <- 1:(cv_folds * cv_reps) %>%
  lapply(
    function(x) {
      index <- cv$train_set(x)
      aux <- as.data.frame(task$data())[index, task$target_names]
      aux <- min(table(aux))
    }
  ) %>%
  bind_cols %>%
  min %>%
  unique

param_set_td$trafo <- function(x, param_set) {
  
  index <- which(grepl('\\.K', names(x)))
  if (sum(index) != 0){
    aux <- round(2 ^ x[[index]])
    if (aux < smote_k_thresh) {
      x[[index]] <- aux
    } else {
      x[[index]] <- sample(smote_k_thresh - 1, 1)
    }
  }
  
  index <- which(grepl('up\\.', names(x)))
  if (sum(index) != 0) {
    x[[index]] <- exp(x[[index]])
  }
  
  index <- which(grepl('down\\.', names(x)))
  if (sum(index) != 0) {
    x[[index]] <- 1 / exp(x[[index]])
  }
  
  x
}

# Add dependencies. For instance, we can only set the mtry value if the pipe is configured to use the Random Forest (ranger).
# In a similar manner, we want do add a dependency between, e.g. hyperparameter "raw.xgb_td.xgb_tn.booster" and branch "raw.xgb_td"
# See https://mlr3gallery.mlr-org.com/posts/2020-02-01-tuning-multiplexer/
param_set_td$ids()[-1] %>%
  lapply(
    function(x) {
      aux <- names(learners_td) %>%
        sapply(
          function(y) {
            grepl(y, x)
          }
        )
      aux <- names(aux[aux])
      param_set_td$add_dep(x, "branch.selection", CondEqual$new(aux))
    }
  )

# Cost matrix to be used in model tuning with cost-sensitive measure. Optional.
# Interpretation: 
# Predicting decline as decline has a non-trivial benefit
# Predicting no_decline as no_decline also has a non-trivial benefit, although smaller
# Predicting decline as no_decline comes has an enormous cost
# Predicting no_decline as decline comes has a relatively big cost
cost_matrix <- matrix(c(80, -23, -131, 1), nrow = 2, byrow = TRUE) # Negative numbers from table(task_train$data()[[response_name]])
dimnames(cost_matrix) <- list(c('decline', 'no_decline'), c('decline', 'no_decline'))
msr_costs <- msr("classif.costs", costs = cost_matrix, normalize = FALSE)

# Load measure PR AUC in case we'd like to optimized based on it
source('measure_prauc.R')
as.data.table(mlr_measures) # Script automatically adds it to the mlr3measures

source('mlr3_prompt_performance_measure.R')
source('mlr3_prompt_cost_matrix.R')
source('mlr3_prompt_n_evaluations.R')

# Set up tuning instance
instance_td <- TuningInstanceSingleCrit$new(
  task = task_train,
  learner = pipe_td,
  resampling = cv,
  measure = measure_classif,
  search_space = param_set_td,
  terminator = trm("evals", n_evals = n_evaluations), 
  store_models = TRUE)
tuner <- TunerRandomSearch$new() # One of many ways of searching the (hyper)parameter space. Type ?Tuner so see all of them.

# Tune pipe learner to find best-performing branch
cat('Tuning pipeline.\n')
tuner$optimize(instance_td)

# Take a look at the results
instance_td # The parameters shown here are still transformed (recall trafo stuff). We are interested in the nested values in x_domain
instance_td$archive # Similar to the above
print(instance_td$result$branch.selection) # Best model
tuning_runs <- instance_td$archive$data(unnest = 'x_domain') # As above but in data frame format
View(tuning_runs, 'Tuning runs') 

# Visually inspect the results
performance_measure <- sub('classif.', '', names(tuning_runs)
  [grep('classif', names(tuning_runs))])

tryCatch(
  {
    archive_up <- tuning_runs %>%
      filter(grepl('up\\.', branch.selection)) %>%
      select_if(.predicate = ~ all(!is.na(.))) %>% 
      rename_at(vars(contains('classif.')), ~ 'measure') %>%
      as.data.table %>%
      select(matches('measure|x_')) %>%
      select(-contains('branch.selection')) %>%
      rename_at(vars(contains('up.')), ~ 'ratio')
  }, error = function(e){}
)

tryCatch(
  {
    archive_down <- tuning_runs %>%
      filter(grepl('down\\.', branch.selection)) %>%
      select_if(.predicate = ~ all(!is.na(.))) %>% 
      rename_at(vars(contains('classif.')), ~ 'measure') %>%
      as.data.table %>%
      select(matches('measure|x_')) %>%
      select(-contains('branch.selection')) %>%
      rename_at(vars(contains('down.')), ~ 'ratio')
  }, error = function(e){}
)

tryCatch(
  {
    archive_smote <- tuning_runs %>%
      filter(grepl('smote\\.', branch.selection)) %>%
      select_if(.predicate = ~ all(!is.na(.))) %>% 
      rename_at(vars(contains('classif.')), ~ 'measure') %>%
      as.data.table %>%
      select(matches('measure|x_')) %>%
      select(-contains('branch.selection')) %>%
      rename_at(vars(contains('.K')), ~ 'K') %>%
      rename_at(vars(contains('.dup_size')), ~ 'dup_size')
  }, error = function(e){}
)

tryCatch(
  {
    viz_up <- archive_up %>%
      ggplot(aes(ratio, measure)) + 
      geom_point(size = 3) + 
      geom_line() + 
      xlab('Over-balancing ratio') + 
      ylab(performance_measure) +
      theme_bw()
  }, error = function(e){}
)

tryCatch(
  {
    print(viz_up)
  }, error = function(e){}
)

tryCatch(
  {
    viz_down <- archive_down %>%
      ggplot(aes(ratio, measure)) + 
      geom_point(size = 3) + 
      geom_line() + 
      xlab('Down-balancing ratio') + 
      ylab(performance_measure) +
      theme_bw()
  }, error = function(e){}
)

tryCatch(
  {
    print(viz_down)
  }, error = function(e){}
)

tryCatch(
  {
    viz_smote <- archive_smote %>%
      ggplot(aes(dup_size, measure, col = factor(K))) + 
      geom_point(size = 3) + 
      geom_line() + 
      xlab('dup size') + 
      ylab(performance_measure) +
      theme_bw()
  }, error = function(e){}
)

tryCatch(
  {
    print(viz_smote)
  }, error = function(e){}
)


# We can now train and test the best-performing model
# 1. Set optimal (hyper)parameter values
# 2. Train on training dataset
# 3. Test on training data. Performance should be overoptimistic and we should expect a near-perfect confusion matrix
# 4. Test on test data. A way of assessing the predictive ability of the model on unseen data. Performance should range from reasonably pessimistic to awful. Latter scenario would indicate model doesn't perform well on fresh data.
# 5. Train model on whole dataset (train + test)
# 6. Calculate feature importance and remove irrelevant features, e.g. predictors contributing less than 5% to an XGBoost model
# 7. Re-train model on whole dataset (train + test)
cat('Setting optimal tuning in pipeline.\n')
pipe_td$param_set$values <- instance_td$result_learner_param_vals # 1. Set optimal (hyper)parameter values
cat('Training pipeline on training dataset.\n')
pipe_td$train(task_train) # 2. Train learner
#pipe_td$predict(task_train)$confusion # 3. Overly optimistic on training data
cat('Assessing performance on test dataset.\n')
pred_td <- pipe_td$predict(task_test)
conf_mat_test <- pred_td$confusion %>% # 4. How is model performing on unseen data?
  as.data.frame.matrix(row.names = paste0(rownames(.), '_pred')) %>%
  as.data.frame %>%
  rename_all(.funs = ~ paste0(., '_truth'))
View(conf_mat_test, 'Confusion matrix on test data')
as.data.table(pred_td) %>% # 4. Also take a quick glance at the predicted probabilities
  arrange(truth, response) %>%
  View('Predictions on test dataset')
print(autoplot(pred_td))
viz_roc <- autoplot(pred_td, type = "roc")
print(viz_roc)
viz_prc <- autoplot(pred_td, type = "prc")
print(viz_prc)
cat('Training pipeline on whole dataset.\n')
pipe_td$train(task) # 5. Train on whole dataset

# Step 6
# For feature importance, the only part of the pipeline we're interested in is the actual learner of the best-performing branch
# Anything other operations (e.g. imputation, class-balancing) aren't relevant here
br_sel <- instance_td$result$branch.selection # Name of best model
learner_best <- pipe_td$ # Pull best model out of pipeline
  graph$
  pipeops[[br_sel]]$
  learner$
  graph$
  pipeops[[sub('.*\\.', '', br_sel)]]$
  learner

# If learner is a Random Forest, we need to define the type of feature importance to be calculated
if (learner_best$id == 'classif.ranger') {
  index <- grepl('importance', learner_best$id)
  learner_best$param_set$values$importance <- 'impurity'
}

# Now train learner on whole dataset
# Don't forget that we've detached the learner from the data imputation operator, as well as from any other operator, e.g. class-balancing
# So we need to manually apply the detached pipe operator(s) on the task prior to training
#if (grepl('raw.', br_sel)) {
#  po_final <- po_raw
#} else if (grepl('up.', br_sel)) {
#  po_final <- po_over
#} else if (grepl('down.', br_sel)) {
#  po_final <- po_under
#} else if (grepl('smote.', br_sel)) {
#  po_final <- po_smote
#}
#learner_best$train((impute_data %>>% po_final)$train(task)[[1]])

# Now train learner on whole dataset
# Don't forget that we've detached the learner from the data imputation operator
# So we need to manually apply the detached pipe operator(s) on the task prior to training
learner_best$train(impute_data$train(task)[[1]])

feature_importance <- learner_best$importance() %>%
  as_tibble(rownames = 'feature') %>%
  rename(score = 'value')

# Plot feature importance scores
viz_feature_importance <- feature_importance %>%
  ggplot(aes(score, factor(feature, levels = feature[order(score)]))) +
  geom_col(width = 0.5) +
  xlab('') +
  ylab('') +
  theme_bw()
print(viz_feature_importance)

# Run this code only if it is desirable to re-train the best learner only with the relevant features
#fun <- function() {
#
#  retrain_with_relevant_features_only <- readline(
#    "Would you like to remove irrelevant features and re-train the learner?
#    Answer by typing TRUE or FALSE: "
#  )
#  retrain_with_relevant_features_only <- 
#    as.logical(retrain_with_relevant_features_only)
#  return(retrain_with_relevant_features_only)
#}
#retrain_with_relevant_features_only <- if(interactive()) fun()
#
#retrain_with_relevant_features_only <- FALSE # Set this to true to run commands below
#if (retrain_with_relevant_features_only) {
#  
#  fun <- function() {
#    
#    cat(
#"Threshold for which features to keep is user-defined.\n
#For XGBoost, it's a % value, e.g. discard all features with < 5% importance.\n
#For Random Forest, it could be the Gini index (for classification).\n
#Choose threshold accordingly.\n
#Hint: a diagnostic plot can help decide what the threshold value should be."
#    )
#    
#    importance_threshold <- readline(
#    "Features with importance < threshold will be discarded.
#    Provide feature importance threshold: "
#    )
#    importance_threshold <- as.numeric(importance_threshold)
#    return(importance_threshold)
#  }
#  importance_threshold <- if(interactive()) fun()
#  
#  #importance_threshold <- 0.5 # User-defined. For XGBoost, it's a % value, e.g. discard all features with < 5% importance. For Random Forest, it's the Gini index for classification. Choose threshold accordingly
#  task_relevant_features <- task$clone(deep = TRUE) # Not clear to me what arg. 'deep' does. Setting to TRUE to be on the safe side.
#  task_relevant_features$
#    select(
#      feature_importance %>%
#        filter(score >= importance_threshold) %>%
#        select(feature) %>%
#        deframe
#    )
#  
#  # If a Random Forest is used, mtry needs to be re-adjusted as its value depends on the number of features
#  if (learner_best$id == 'classif.ranger') {
#    index <- grep('mtry', names(pipe_td$param_set$values))
#    pipe_td$param_set$values[index] <- 
#      round(0.666 * (task_relevant_features$ncol - 1)) # Recall multiplier 0.666 is optimal value from Probst et al. for the Brier score (Table 9)
#  }
#  
#  # Train learner again
#  pipe_td$train(task_relevant_features)
#  # Learner now ready for predictions
#}

####################################################################################

# A few general tips I've gathered here and there
# Method $instantiate() fixes the train-test splits for the whole script.
# Avoid invalid learner configs with tune_ps$add_dep(id, on, cond).
# Don't instantiate inner resampling for AutoTuner.
# By calling the method instantiate(), we split the indices of the data into indices for training and test sets.
# Resampling strategies are not allowed to be instantiated when passing the argument, and instead will be instantiated per task internally.
# With a Task, a Learner and a Resampling object we can call resample(), which fits the learner to the task at hand according to the given resampling strategy.