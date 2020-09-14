library(tidyverse)
library(janitor)
library(ruimtehol)

load('cleanData.RData')

text_data <- trustData %>%
  clean_names %>%
  left_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  select(super, improve) %>%
  filter_all(~ !is.na(.)) %>%
  as_tibble

train_id <- text_data %>%
  rownames_to_column %>%
  slice_sample(prop = 2 / 3) %>%
  pull(rowname) %>%
  as.numeric

train_data <- text_data[train_id, ]
test_data <- text_data[-train_id, ]

train_data$x <- strsplit(train_data$improve, "\\W")
train_data$x <- sapply(train_data$x, FUN = function(x) paste(setdiff(x, ""), collapse = " "))
train_data$x <- tolower(train_data$x)
train_data$y <- strsplit(as.character(train_data$super), split = ",")
train_data$y <- lapply(train_data$y, FUN=function(x) gsub(" ", "-", x))

model <- embed_tagspace(x = train_data$x, y = train_data$y,
  dim = 50, 
  lr = 0.01, epoch = 40, loss = "softmax", adagrad = TRUE, 
  similarity = "cosine", negSearchLimit = 50,
  ngrams = 2, minCount = 2)

plot(model)                     

text <- test_data$improve
pr <- predict(model, text, k = 3)

acc <- pr %>%
  lapply(
    function(x) {
      x$prediction %>%
        select(label, label_starspace) %>%
        slice(1) %>%
        mutate_at('label_starspace', ~ sub('__label__', '', .))
    }
  ) %>%
  bind_rows %>%
  mutate(same = label == label_starspace)

sum(acc$same) / nrow(acc)