####################################################################################
# Prepare dataset for mlr3
# 1. Clean data fram names
# 2. Sort columns in alphabetical order. Good practice as subsequent manipulation of the dataset is easier when columns are in alph. order
# 3. Prepare training and test tasks for mlr3
# 4. Remove correlated predictors from tasks
####################################################################################

source('mlr3_prompt_dataset.R')
source('mlr3_prompt_task_id.R')


# Make data frame names R-friendly
dataset <- dataset %>% 
  clean_names %>%
  as.data.table

View(head(dataset), 'Dataset - first 6 rows')

source('mlr3_prompt_response_name.R')

# Sort columns in alph. order but place response in 1st column
dataset <- dataset %>%
  select(
    all_of(response_name), # all_of() deals with unambiguity when vectors (e.g. response_name) and column names (e.g. conv_outcome) are the same. Function ensures operation is performed on the actual column
    all_of(sort(names(.)[-which(names(.) == response_name)]))
  ) %>%
  mutate_at(response_name, as.factor)

#source('mlr3_prompt_positive_class.R')

# Prepare training and test tasks for mlr3
#task <- TaskClassif$new("task_id", dataset, 
#  target = response_name, positive = positive_class)
#task$col_roles$stratum <- response_name
task <- TaskClassif$new("task_id", dataset, 
  target = response_name)
task$col_roles$stratum <- response_name

source('mlr3_prompt_train_test_split.R')

# Indices for splitting data into training and test sets
train.idx <- dataset %>%
  select(all_of(response_name)) %>%
  rownames_to_column %>%
  group_by_at(vars(one_of(response_name))) %>% # As with all_of() within select(), syntax here ensures dplyr understands that vector refers to column name
  sample_frac(training_frac) %>% # Stratified sample to maintain proportions between classes.
  ungroup %>%
  select(rowname) %>%
  deframe %>%
  as.numeric
test.idx <- setdiff(seq_len(task$nrow), train.idx)

# Define training and test sets in task format
# We clone task first, otherwise original task would be filtered too
task_train <- task$clone()$filter(train.idx)
task_test <- task$clone()$filter(test.idx)

sort(task$missings()[task$missings() > 0], decreasing = TRUE)
sort(task_train$missings()[task_train$missings() > 0], decreasing = TRUE)
sort(task_test$missings()[task_test$missings() > 0], decreasing = TRUE)

# Detect and remove (optional) correlated predictors from tasks
# If we have missing data, they will be imputed before calculating correlations
# Imputation methods are many: histogram, mean, median, mode, sampling from data, use KNN, use bagged trees
# Some are available in mlr3, others in tidymodels only
# We'll use all of them and select the final set from all methods
# Also, for simplicity, we're running the whole process on the original, unsplit task

source('mlr3_prompt_correlation_threshold.R')

cat(
'Correlations between features will now be calculated.\n
The processs imputes missing data (if any) before calculating the correlations.\n
One of the imputation methods used is bagged trees.\n
Bagged trees takes a few seconds to run even on a data frame with < 250 rows.\n
Please be patient.'
)
# Approach 1: data imputation with mlr3
imputation_methods <- c('imputehist', 'imputemean', 'imputemedian', # Imputation methods available in mlr3
  'imputemode', 'imputesample')
# Run all available imputation methods on data, get correlations, filter those with Spearman's corr >= user-defined threshold
corr_vars_mlr3 <- imputation_methods %>%
  lapply(
    function(x) {
      # Correlation matrix
      aux <- po(x)$ # Pipe operator for the imputation method 'x' passed into lapply() from vector imputation_methods 
        train(list(task = task$clone()))$ # Run imputation method on task. Don't forget to clone task first, otherwise original task would be imputed too
        output$ # This and next line get the imputed data from the cloned task
        data() %>%
        select(task$feature_names) %>% # We don't want the response variable to be part of the corr. matrix
        correlate(method = 'spearman', quiet = TRUE) %>%
        shave() %>% 
        melt(na.rm = TRUE) %>% # All pairwise corrs are now in pairs per row, with 3rd col. having the corr. value
        mutate_if(is.factor, as.character) %>%
        as_tibble %>%
        filter(abs(value) >= corr_threshold) %>%
        mutate(imputation_method = x, imputation_method_package = 'mlr3')
    }
  ) %>%
  bind_rows %>%
  distinct %>%
  arrange(desc(abs(value)), variable) %>%
  select(variable, everything()) %>%
  rename(
    feature_with_high_correlations = variable, 
    feature_it_is_highly_correlated_with = rowname, 
    corr_coef = value
  )

# Approach 2: data imputation with tidymodels
# Two available data imputation methods are KNN and bagged trees
# KNN
predictor_names <- paste(task$feature_names, collapse = "+") 
model_formula <- paste(response_name, "~ ", predictor_names, sep = " ")
corr_vars_knn <- recipe(formula = model_formula, dataset) %>%
  step_knnimpute(all_predictors()) %>%
  prep %>%
  juice %>%
  select_if(is.numeric) %>%
  correlate(method = 'spearman', quiet = TRUE) %>%
  shave() %>% 
  melt(na.rm = TRUE) %>% # All pairwise corrs are now in pairs per row, with 3rd col. having the corr. value
  mutate_if(is.factor, as.character) %>%
  as_tibble %>%
  filter(abs(value) >= corr_threshold) %>%
  distinct %>%
  arrange(desc(abs(value)), variable) %>%
  select(variable, everything()) %>%
  rename(
    feature_with_high_correlations = variable, 
    feature_it_is_highly_correlated_with = rowname, 
    corr_coef = value
  )

# Bagged trees
corr_vars_bag <- recipe(formula = model_formula, dataset) %>%
  step_bagimpute(all_predictors()) %>%
  prep %>%
  juice %>%
  select_if(is.numeric) %>%
  correlate(method = 'spearman', quiet = TRUE) %>%
  shave() %>% 
  melt(na.rm = TRUE) %>% # All pairwise corrs are now in pairs per row, with 3rd col. having the corr. value
  mutate_if(is.factor, as.character) %>%
  as_tibble %>%
  filter(abs(value) >= corr_threshold) %>%
  distinct %>%
  arrange(desc(abs(value)), variable) %>%
  select(variable, everything()) %>%
  rename(
    feature_with_high_correlations = variable, 
    feature_it_is_highly_correlated_with = rowname, 
    corr_coef = value
  )

correlated_predictors <- corr_vars_mlr3 %>% 
  bind_rows(
    mutate(
      corr_vars_knn, 
      imputation_method = 'step_knn', 
      imputation_method_package = 'tidymodels'
    ), 
    mutate(
      corr_vars_bag, 
      imputation_method = 'step_bag', 
      imputation_method_package = 'tidymodels'
    )
  ) %>%
  arrange(desc(abs(corr_coef)), feature_with_high_correlations, 
    feature_it_is_highly_correlated_with, imputation_method)

View(correlated_predictors, 'Correlated predictors')

cat('Training and test tasks are now ready for the pipeline.')