library(tidyverse)
library(quanteda)
library(quanteda.textmodels)
library(caret) # Function confusionMatrix()

# Train and test text data with Quanteda's multinomial Naive Bayes
# Predictor is variable 'improve' and response is variable 'super'
# See https://tutorials.quanteda.io/machine-learning/nb/

# Load, clean and prepare data
load('cleanData.RData')

quanteda_data <- trustData %>%
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.Date, as.POSIXct) %>%
  clean_names %>%
  left_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  select(super, date, division2, directorate2, improve) %>%
  filter_all(~ !is.na(.)) %>%
  filter(
    !super %in% c('Equality/Diversity', 
      'Physical Health', 'Record Keeping', 'Safety', 'MHA', 'Smoking', 'Leave')
  ) %>% # Too few in the data (e.g. 2-3 in 10-30% data samples)
  as_tibble

names(quanteda_data)

# Create a corpus that also has a doc ID and a dependent variable
corpus_patient_feedback <- corpus(quanteda_data$improve)
corpus_patient_feedback$id_numeric <- 1:ndoc(corpus_patient_feedback)
corpus_patient_feedback$super <- quanteda_data$super
summary(corpus_patient_feedback, 5)

# Random sample without replacement for training
id_train <- sample(
  1:length(corpus_patient_feedback), 
  round(2 / 3 * length(corpus_patient_feedback)), 
  replace = FALSE
)

# Training set
quanteda_data_training <- corpus_subset(
  corpus_patient_feedback, 
  id_numeric %in% id_train
) %>%
  dfm(remove = stopwords("english"), stem = TRUE)

# Test set
quanteda_data_test <- corpus_subset(
  corpus_patient_feedback, 
  !id_numeric %in% id_train
) %>%
  dfm(remove = stopwords("english"), stem = TRUE)

# Naive Bayes
quanteda_nb <- textmodel_nb(quanteda_data_training, 
  quanteda_data_training$super, prior = 'termfreq')
summary(quanteda_nb)

# With Naive Bayes, features must occur both in the training set and the test sets
# Make the features identical
quanteda_data_matched <- dfm_match(quanteda_data_test, 
  features = featnames(quanteda_data_training))

# Confusion matrix and metrics
actual_class <- quanteda_data_matched$super
predicted_class <- predict(quanteda_nb, newdata = quanteda_data_matched)
tab_class <- table(actual_class, predicted_class)
tab_class

confusionMatrix(tab_class, mode = "everything")
