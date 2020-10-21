# Before running the pipeline, be aware that it may take a while.
# For example, a 5-fold repeated CV with 20 evaluations within a random search tuner,
#  set to benchmark four XGBoost models, takes about 1h for the KS1 data.
# However, also note that Random Forest is significantly faster:
#  it can crunch the data with the aforementioned settings in a few minutes.
# For larger tasks and/or number of folds and evaluations,
#  it may be necessary to optimize overnight.
# For experimentation puproses, set appropriate values for cross-validation and
#  number of evaluations. Appropriate messages will pop in the command line
#  when the time comes for these values to be set.

source('mlr3_load_packages.R')

source('mlr3_data_clean_and_prepare.R')

source('mlr3_prepare_test_and_training_tasks.R')

source('mlr3_pipeline_optimal_defaults.R')
