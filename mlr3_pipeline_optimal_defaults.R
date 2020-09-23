####################################################################################
# Build Machine Learning pipeline that:
# 1. Performs feature engineering on date, categorical and text columns
# 2. Tunes and benchmarks a range of learners
# 3. Identifies optimal learner for the task at hand.

# References
# Probst et al. (2009). http://jmlr.csail.mit.edu/papers/volume20/18-444/18-444.pdf
####################################################################################

future::plan("multiprocess")

####################################################################################
# We will be using various learners (incl. XGBoost and Random Forest) throughout with different hyperparameter settings
# Learners are defined below in alphabetical order
####################################################################################
message('Setting up learners.\n')

# Generalized Linear Model with Elastic Net
glmnet_brier <- lrn("classif.glmnet", predict_type = "prob")
glmnet_brier$param_set$values <- list(
  alpha = 0.997,
  lambda = 0.004
)

nb_learner <- lrn("classif.naive_bayes", predict_type = "prob")

# Random Forest
rf_brier <- lrn("classif.ranger", predict_type = "prob")
rf_brier$param_set$values <- list(
  num.trees = 198,
  replace = FALSE,
  sample.fraction = 0.667,
  mtry = round(0.666 * (task$ncol - 1)), # NOTE: if feature selection is performed, user must change this manually to reflect the smaller number of features in the new task
  respect.unordered.factors = 'ignore',
  min.node.size = 1
)

# Support Vector Machines
svm_brier <- lrn("classif.svm", predict_type = "prob")
svm_brier$param_set$values <- list(
  kernel = 'radial',
  cost = 950.787,
  gamma = 0.005,
  #degree = 3, # This is irrelevant when kernel isn't polynomial
  type = 'C-classification'
)

# Define XGBoost learner with the optimal hyperparameters of Probst et al. for the Brier score
xgb_brier <- lrn("classif.xgboost", predict_type = 'prob')
xgb_brier$param_set$values <- list(
  booster = "gbtree", 
  #nrounds = 2563, # NOTE!!! This is the optimal value found by Probst et al. However, this value in combination with the eta value slow down things quite a lot. I've replaced with 3, for the pipeline to run in a reasonable amount of time
  nrounds = 3,
  max_depth = 11, 
  min_child_weight = 1.75, 
  subsample = 0.873, 
  eta = 0.052,
  colsample_bytree = 0.713,
  colsample_bylevel = 0.638,
  lambda = 0.101,
  alpha = 0.894
)

learners <- list(
  glmnet_brier#, # Having problems with it inside the pipeline. GLM NET uses Elastic Net to perform feature selection during training. It looks like the pipeline gets confused when testing on the k-th CV fold, as the training and test datasets have different features
  #nb_learner, # Naive Bayes can handle datasets with high dimensionality
  #rf_brier#, # Being a tree, I'm guessing it can also handle the high-dimensionality nature of text data?
  #svm_brier#,
  #xgb_brier # Can be slow with the defaults of Probst et al. It may be the learning rate in combination with the large number of trees?
)
names(learners) <- sapply(learners, function(x) x$id)

####################################################################################
# Feature engineering pipe operators
# Convert date column into two numeric columns for month and year
# Vectorize the text columns
# One-hot encode categorical columns
# Before one-hot encoding, ensure categorical columns are factors, because PipeOpEncode works with factors only
####################################################################################

po_date <- po("datefeatures", 
  param_vals = list(year = TRUE, month = TRUE, day_of_month = FALSE,
    week_of_year = FALSE, day_of_year = FALSE, day_of_week = FALSE,
    hour = FALSE, minute = FALSE, second = FALSE, is_day = FALSE))

po_text <- po(
  "textvectorizer", 
  param_vals = list(
    stopwords_language = "en",
    scheme_df = 'inverse',
    remove_punct = TRUE,
    remove_symbols = TRUE,
    remove_numbers = TRUE
  ),
  affect_columns = selector_name('improve')
)

po_char_to_factor <- po(
  'colapply', 
  affect_columns = 
    selector_union(
      selector_type(c("character")),
      selector_grep('date.', fixed = TRUE)
    ), 
  applicator = as.factor
)

po_one_hot <- po(
  "encode", 
  affect_columns = selector_type(c("factor"))
)

feature_engineering <-  
  po_date %>>%
  po_text %>>%
  po_char_to_factor %>>%
  po_one_hot

####################################################################################
# PIPELINE
# Create pipeline as a graph. This way, pipeline can be plotted. Pipeline can then be converted into a learner with GraphLearner$new(pipeline)
# Pipeline is a collection of pipe operators and learners
####################################################################################
message('Perparing pipeline, parameter sets and resampling strategy.\n')

# Our pipeline
graph <- 
  feature_engineering %>>%
  po("branch", names(learners)) %>>% # Branches will be as many as the learners
  gunion(unname(learners)) %>>%
  po("unbranch") # Gather results for individual learners into a results table

graph$plot() # Plot pipeline
graph$plot(html = TRUE) # Plot pipeline

pipe <- GraphLearner$new(graph) # Convert pipeline to learner
pipe$predict_type <- 'prob' # Don't forget to specify we want to predict probabilities and not classes

# Define resampling strategy to be applied within pipeline
# Instantiate it to ensure data split is the same in all pipelines and benchmarking exercises to follow
source('mlr3_prompt_resampling_strategy.R')

source('mlr3_prompt_cv_folds.R')

if (resampling_strategy == "repeated_cv") {
  cv_reps <- cv_folds_or_holdout_sample_size
  resampling_strategy <- rsmp("repeated_cv", 
    repeats = cv_reps, folds = cv_folds_or_holdout_sample_size) # Repeats and folds should be set to 5 (5 x 5 runs) or 10 (10 x 10 runs)
  resampling_strategy$instantiate(task_train)
} else if (resampling_strategy == "cv") {
  cv_reps <- 1 # Need that for later
  resampling_strategy <- rsmp("cv", folds = cv_folds_or_holdout_sample_size) # Repeats and folds should be set to 5 (5 x 5 runs) or 10 (10 x 10 runs)
  resampling_strategy$instantiate(task_train)
} else if (resampling_strategy == "holdout") {
  resampling_strategy <- 
    rsmp("holdout", ratio = cv_folds_or_holdout_sample_size)
  resampling_strategy$instantiate(task_train)
}


# Parameter set of the pipeline
ps_table <- as.data.table(pipe$param_set)
View(ps_table[, 1:4])

ps_text <- ParamSet$new(list(
  ParamInt$new('textvectorizer.n', lower = 2, upper = 3))
)

param_set <- ParamSetCollection$new(list(
  ParamSet$new(list(pipe$param_set$params$branch.selection$clone())), # ParamFct can be copied.
  ps_text
))

# Load measure PR AUC in case we'd like to optimized based on it
source('measure_prauc.R')
as.data.table(mlr_measures) # Script automatically adds it to mlr3measures

source('mlr3_prompt_performance_measure.R')
source('mlr3_prompt_n_evaluations.R')

# Set up tuning instance
instance <- TuningInstanceSingleCrit$new(
  task = task_train,
  learner = pipe,
  resampling = resampling_strategy,
  measure = measure_classif,
  search_space = param_set,
  terminator = trm("evals", n_evals = n_evaluations), 
  store_models = TRUE)
tuner <- TunerRandomSearch$new() # One of many ways of searching the (hyper)parameter space. Type ?Tuner so see all of them

# Tune pipe learner to find best-performing branch
message('Tuning pipeline.\n')
tuner$optimize(instance)

# Take a look at the results
instance # The parameters shown here will be transformed (if a trafo was applied). We are interested in the nested values in x_domain
instance$archive # Similar to the above
print(instance$result$branch.selection) # Best model
tuning_runs <- instance$archive$data(unnest = 'x_domain') # As above but in data frame format
View(tuning_runs, 'Tuning runs') 

# Visually inspect the results
performance_measure <- sub('classif.', '', names(tuning_runs)
  [grep('classif', names(tuning_runs))])

tryCatch(
  {
    archive_ngrams <- tuning_runs %>%
      select_if(.predicate = ~ all(!is.na(.))) %>% 
      rename_at(vars(contains('classif.')), ~ 'measure') %>%
      as_tibble %>%
      select(matches('measure|x_')) %>%
      rename_at(vars(contains('textvectorizer.')), ~ 'n') %>%
      rename_at(vars(contains('branch.')), ~ 'model') %>%
      mutate_at(
        'model',
        ~ sub('classif.', '', .)
      )
  }, error = function(e){}
)

tryCatch(
  {
    viz_ngrams <- archive_ngrams %>%
      ggplot(aes(n, measure, shape = model, colour = model)) + 
      geom_point(size = 3) + 
      geom_line() + 
      xlab('n value in n-grams') + 
      ylab(performance_measure) +
      theme_bw()
  }, error = function(e){}
)

tryCatch(
  {
    print(viz_ngrams)
  }, error = function(e){}
)

# We can now train and test the best-performing model
# 1. Set optimal (hyper)parameter values
# 2. Train on training dataset
# 3. Test on training data. Performance should be overoptimistic and we should expect a near-perfect confusion matrix
# 4. Test on test data. A way of assessing the predictive ability of the model on unseen data. Performance should range from reasonably pessimistic to awful. Latter scenario would indicate model doesn't perform well on fresh data.
# 5. Train model on whole dataset (train + test)
# OPTIONAL:
# 6. Calculate feature importance and remove irrelevant features, e.g. predictors contributing less than 5% to an XGBoost model
# 7. Re-train model on whole dataset (train + test)
message('Setting optimal tuning in pipeline.\n')
pipe$param_set$values <- instance$result_learner_param_vals # 1. Set optimal (hyper)parameter values

message('Training pipeline on training dataset.\n')
pipe$train(task_train) # 2. Train learner

#pipe$predict(task_train)$confusion # 3. Overly optimistic on training data

message('Assessing performance on test dataset.\n')
pred <- pipe$predict(task_test)

conf_mat_test <- pred$confusion %>% # 4. How is model performing on unseen data?
  as.data.frame.matrix(row.names = paste0(rownames(.), '_pred')) %>%
  as.data.frame %>%
  rename_all(.funs = ~ paste0(., '_truth'))
View(conf_mat_test, 'Confusion matrix on test data')

actual_class <- pred$truth
predicted_class <- pred$response
tab_class <- table(actual_class, predicted_class)
caret::confusionMatrix(tab_class, mode = "everything")

as.data.table(pred) %>% # 4. Also take a quick glance at the predicted probabilities
  arrange(truth, response) %>%
  View('Predictions on test dataset')

pred %>%
  as.data.table %>%
  as_tibble %>%
  roc_curve(truth, starts_with('prob.')) %>%
  autoplot

pred %>%
  as.data.table %>%
  as_tibble %>%
  pr_curve(truth, starts_with('prob.')) %>%
  autoplot

message('Training pipeline on whole dataset.\n')
pipe$train(task) # 5. Train on whole dataset

# Step 6
# For feature importance, the only part of the pipeline we're interested in is the actual learner of the best-performing branch
# Anything other operations (e.g. imputation, class-balancing) aren't relevant here
br_sel <- instance$result$branch.selection # Name of best model
learner_best <- pipe$ # Pull best model out of pipeline
  graph$
  pipeops[[br_sel]]$
  learner

# If learner is a Random Forest, we need to define the type of feature importance to be calculated
if (learner_best$id == 'classif.ranger') {
  index <- grepl('importance', learner_best$id)
  learner_best$param_set$values$importance <- 'impurity'
}

# Now train learner on whole dataset
# Don't forget that we've detached the learner from the data imputation operator
# So we need to manually apply the detached pipe operator(s) on the task prior to training
learner_best$train(feature_engineering$train(task)[[1]])

feature_importance <- learner_best$importance() %>%
  as_tibble(rownames = 'feature') %>%
  rename(score = 'value')

# Plot feature importance scores
viz_feature_importance <- feature_importance %>%
  filter(!near(score * 100, 0)) %>%
  ggplot(aes(score, factor(feature, levels = feature[order(score)]))) +
  geom_col(width = 0.5) +
  xlab('') +
  ylab('') +
  theme_bw()
print(viz_feature_importance)
