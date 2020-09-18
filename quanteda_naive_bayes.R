library(quanteda)
library(quanteda.textmodels)
source('mlr3_load_packages.R')

# Load data and prepare training and test sets
source('mlr3_data_clean_and_prepare.R')
source('mlr3_prepare_test_and_training_tasks.R')
# The loaded data frame is called pipeline_data

pipeline_data_dfm <- dfm(pipeline_data$improve)

nb <- textmodel_nb(
  x = pipeline_data_dfm[train.idx, ], 
  y = pipeline_data$super[test.idx], 
  prior = 'termfreq')

preds <- 
  data.frame(
    predicted = predict(nb, newdata = pipeline_data_dfm[test.idx, ])
  ) %>%
  mutate(true = pipeline_data$super[test.idx])

sum(preds$true == preds$predicted) / nrow(preds)
