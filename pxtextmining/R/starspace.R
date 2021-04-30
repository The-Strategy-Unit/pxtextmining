library(tidyverse)
library(janitor)
library(ruimtehol)
library(ggwordcloud)
library(M3C) # T-SNE plot

load('cleanData.RData')


###############################################################################
# Approach 1: supervised learning
###############################################################################

# Data pre-processing
# 1. Clean names and remove NAs in table
data_text <- trustData %>%
  clean_names %>%
  left_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  select(super, improve) %>%
  filter_all(~ !is.na(.)) %>% # Don't want NAs in response variable or text feature
  as_tibble

# 2. Text data in StarSpace-friendly format
data_text$improve <- strsplit(data_text$improve, "\\W")
data_text$improve <- 
  purrr::map_chr(
    data_text$improve, 
    ~ paste(setdiff(.x, ""), collapse = " ")
  )
data_text$improve <- tolower(data_text$improve)
data_text$super <- strsplit(as.character(data_text$super), split = ",")
data_text$super <- purrr::map(data_text$super, ~ gsub(" ", "-", .x))

# Sample for training set
train_id <- data_text %>%
  rownames_to_column %>%
  slice_sample(prop = 0.9) %>%
  pull(rowname) %>%
  as.numeric

data_train <- data_text[train_id, ]
data_test <- data_text[-train_id, ]

# Build model
model_supervised <- embed_tagspace(x = data_train$improve, y = data_train$super,
  early_stopping = 0.8,
  validationPatience = 10,
  dim = 50,
  lr = 0.01, 
  epoch = 60, 
  loss = "softmax", 
  adagrad = TRUE, 
  similarity = "cosine", 
  negSearchLimit = 50,
  ngrams = 5, 
  minCount = 5)

plot(model_supervised)

message("Don't be surprised if model is pretty awful. This is a very early, exploratory stage.")

# Dictionary
dict <- starspace_dictionary(model_supervised)
#str(dict)

## Get embeddings of the dictionary of words as well as the categories
embedding_words <- as.matrix(model_supervised, type = "words")
embedding_labels <- as.matrix(model_supervised, type = "label")

## Find closest labels / predict
embedding_combination <- 
  starspace_embedding(
    model_supervised, 
    data_test$improve, 
    type = "document"
  )

# Predictions table
pr_supervised <- predict(model_supervised, data_test$improve) %>%
  map_dfr(~ {
    .x$prediction %>%
      slice(1) %>%
      select(label)
  }) %>% 
  bind_cols(select(data_test, super)) %>%
  mutate(
    super = as.character(super),
    same = label == super
  )

pr_embeddings <- predict(model_supervised, data_test$improve, type = 'embedding') %>%
  as_tibble(.name_repair = 'universal') %>%
  rename_all(~ paste0('emb_dim_', sub('...', '', .)))

# Assess on test data
message(
  paste0(
    'Accuracy is ', 
    round(sum(pr_supervised$same) / nrow(pr_supervised) * 100), 
    '%')
  )

# Plot model performance with T-SNE plot
tsne(t(distinct(pr_embeddings)), perplex = 15, 
  labels = as.factor(pr_supervised$label[!duplicated(pr_embeddings)]))
message(
  "Clusters in T-SNE plot slightly distinct from each other.
We need much better than that, but at least we can see some elementary structure."
)

# Word clouds
word_clouds <- rownames(embedding_labels) %>%
  purrr::map(
    ~ {
      starspace_knn(model_supervised, .x, k = 11)$prediction %>%
        slice(-1) %>%
        select(-rank) %>%
        ggplot(aes(label = label, size = similarity, color = similarity)) + 
        geom_text_wordcloud_area() +
        scale_size_area(max_size = 16) +
        theme_minimal() +
        scale_color_gradient(low = "orange", high = "blue") + 
        ggtitle(sub('__label__', '', .x))
    }
  )

###############################################################################
# Approach 2: semi-supervised learning
###############################################################################

load('cleanData.RData')

# Data preprocessing
# 1. Clean names and remove NAs in table
data_text <- trustData %>%
  clean_names %>%
  full_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  select(super, improve) %>%
  filter(!is.na(improve)) %>% # Get rid of empty text features but, contrary to Approach 1, keep NAs in response variable, to convert problem from supervised to semi-supervised
  as_tibble

# 2. Text data in StarSpace-friendly format
data_text$improve <- strsplit(data_text$improve, "\\W")
data_text$improve <- lapply(data_text$improve, FUN = function(x) setdiff(x, ""))
data_text$improve <- sapply(data_text$improve, FUN = function(x) paste(x, collapse = " "))
data_text$improve <- tolower(data_text$improve)
data_text$super <- strsplit(data_text$super, split = ",")
data_text$super <- lapply(data_text$super, FUN = function(x) gsub(" ", "-", x))
data_text$improve[1:2]

# Some whitespaces survive. Don't know why. Explicitly remove them here
data_text <- data_text %>%
  filter(!improve %in% c('', ' '))

# For semi-supervised learning, we want a fully-labeled test dataset and a training dataset with some labeled data, rest unlabeled
data_text_no_nas <- data_text %>%
  filter(!is.na(super))
data_text_nas <- data_text %>%
  filter(is.na(super))

# Sample from the labeled dataset and bind this chunk of data with the unlabeled dataset- this willbe our training set
train_id <- data_text_no_nas %>%
  rownames_to_column %>%
  slice_sample(prop = 0.9) %>%
  pull(rowname) %>%
  as.numeric

data_train <- rbind(data_text_no_nas[train_id, ], data_text_nas) # Training set a blend of labeled and unlabeled data
data_test <- data_text_no_nas[-train_id, ] # Test set has labeled data only

model_semisupervised <- embed_tagspace(x = data_train$improve, y = data_train$super,
  early_stopping = 0.8, validationPatience = 10,
  dim = 50,
  lr = 0.01, epoch = 40, loss = "softmax", adagrad = TRUE,
  similarity = "cosine", negSearchLimit = 50,
  ngrams = 2, minCount = 2)

plot(model_semisupervised)

pr_semisupervised <- predict(model_semisupervised, data_test$improve) %>%
  map_dfr(~ {
    .x$prediction %>%
      slice(1) %>%
      select(label)
  }) %>% 
  bind_cols(select(data_test, super)) %>%
  mutate(
    super = as.character(super),
    same = label == super
  )

# Assess on test data
message(
  paste0(
    'Accuracy is ', 
    round(sum(pr_semisupervised$same) / nrow(pr_semisupervised) * 100), 
    '%'
  )
)
message('Semi-supervised learning does not really improve the model...')
