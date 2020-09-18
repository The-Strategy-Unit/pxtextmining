library(tidyverse)
library(janitor)
library(ruimtehol)
library(ggwordcloud)
library(M3C)

load('cleanData.RData')

# Data preprocessing
# 1. Clean names and remove NAs in table
data_text <- trustData %>%
  clean_names %>%
  left_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  select(super, improve) %>%
  filter_all(~ !is.na(.)) %>%
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

train_id <- data_text %>%
  rownames_to_column %>%
  slice_sample(prop = 0.9) %>%
  pull(rowname) %>%
  as.numeric

data_train <- data_text[train_id, ]
data_test <- data_text[-train_id, ]

model <- embed_tagspace(x = data_train$improve, y = data_train$super,
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

plot(model)

message("Don't be surprised if model is pretty awful. This is a very early, exploratory stage.")

# Dictionary
dict <- starspace_dictionary(model)
#str(dict)

## Get embeddings of the dictionary of words as well as the categories
embedding_words <- as.matrix(model, type = "words")
embedding_labels <- as.matrix(model, type = "label")

## Find closest labels / predict
embedding_combination <- 
  starspace_embedding(
    model, 
    data_test$improve, 
    type = "document"
  )

pr <- predict(model, data_test$improve) %>%
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

pr_embeddings <- predict(model, data_test$improve, type = 'embedding') %>%
  as_tibble(.name_repair = 'universal') %>%
  rename_all(~ paste0('emb_dim_', sub('...', '', .)))

# Assess on test data
message(paste0('Accuracy is ', round(sum(pr$same) / nrow(pr) * 100), '%'))

tsne(t(distinct(pr_embeddings)), perplex = 15, 
  labels = as.factor(pr$label[!duplicated(pr_embeddings)]))
message(
  "Clusters in T-SNE plot slightly distinct from each other.
We need much better than that, but at least we can see some elementary structure."
)

# Word clouds
word_clouds<- rownames(embedding_labels) %>%
  purrr::map(
    ~ {
      starspace_knn(model, .x, k = 11)$prediction %>%
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
