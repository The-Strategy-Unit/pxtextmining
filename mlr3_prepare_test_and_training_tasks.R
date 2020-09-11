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
    all_of(response_name), # all_of() deals with ambiguity when vectors (e.g. response_name) and column names (e.g. conv_outcome) are the same. Function ensures operation is performed on the actual column
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

task$droplevels()
task_train$droplevels()
task_test$droplevels()

sort(task$missings()[task$missings() > 0], decreasing = TRUE)
sort(task_train$missings()[task_train$missings() > 0], decreasing = TRUE)
sort(task_test$missings()[task_test$missings() > 0], decreasing = TRUE)

cat('Training and test tasks are now ready for the pipeline.')