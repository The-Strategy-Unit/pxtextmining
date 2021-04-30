
# set up

library(keras)
library(tidyverse)
library(stringr)

load("cleanData.Rdata")

maxlen <- 400 # cuts off comments after 100 words
max_words <- 5000 # considers only top n words in dataset
size_of_data <- 15000 # number of rows of data to use

# add a 4444 code onto the categoriesTable

categoriesTable <- categoriesTable %>% 
  bind_rows(data.frame("Category" = "General", "Super" = "Couldn't be improved", "Number" = 4444, "type" = "both"))

# reduce categories

# all_data <- trustData %>%
#   left_join(categoriesTable, c("Imp1" = "Number")) %>%
#   filter(!is.na(Super)) %>%
#   filter(!is.na(Improve), !is.na(Imp1), Imp1 != 4444, Division %in% 0) %>%
#   filter(grepl(paste(c("Access to Services", "Care/ Treatment", "Communication", "Environment/Facilities",
#                        "Food", "Service Quality/Outcomes", "Staff/Staff Attitude"),
#                      collapse="|"), Super)) %>%
#   sample_n(size_of_data)

# give the variable names here

# keep_variables <- c("Access to Services", "Care/ Treatment", "Communication", "Couldn't be improved", 
#                     "Environment/Facilities", "Food", "Staff/Staff Attitude")

# keep_variables <- c("Access to Services", "Care/ Treatment", "Communication", "Couldn't be improved", 
#                     "Environment/Facilities", "Food", "Involvement", "Service Quality/Outcomes", 
#                     "Smoking", "Staff/Staff Attitude")

# all variables

keep_variables <- c("Access to Services", "Care/ Treatment", "Communication", "Couldn't be improved", 
                    "Environment/Facilities", "Equality/Diversity", "Food", "Involvement", 
                    "Leave", "MHA", "Physical Health", "Privacy and Dignity", "Record Keeping", 
                    "Safety", "Service Quality/Outcomes", "Smoking", "Staff/Staff Attitude")

all_data <- trustData %>%
  left_join(categoriesTable, c("Imp1" = "Number")) %>%
  filter(!is.na(Super)) %>%
  # filter(str_count(Improve, '\\w+') >= 3) %>%
  filter(!is.na(Improve), !is.na(Imp1), Division %in% 0) %>%
  filter(grepl(paste(keep_variables, collapse="|"), Super)) %>%
  sample_n(size_of_data)

# all_data <- trustData %>% 
#   left_join(categoriesTable, c("Best1" = "Number")) %>% 
#   filter(!is.na(Super)) %>% 
#   filter(!is.na(Best), !is.na(Best1), Division %in% 0 : 1) %>%
#   filter(grepl(paste(c("Access to Services", "Care/ Treatment", "Communication", "Environment/Facilities",
#                        "Food", "Service Quality/Outcomes", "Staff/Staff Attitude"), 
#                      collapse="|"), Super)) %>% 
#   sample_n(size_of_data)

size_of_data = (2 * round(nrow(all_data) / 2)) - 2

# creates the text tokenizer- so create on test and training data

tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(all_data$Improve)

sequences <- texts_to_sequences(tokenizer, all_data$Improve)

data <- pad_sequences(sequences, maxlen = maxlen)

labels <- as.array(as.numeric(as.factor(all_data$Super)))

training_indices <- 1 : (size_of_data / 2)

validation_indices <- ((size_of_data / 2) + 1) : size_of_data

x_train <- data[training_indices, ]
y_train <- labels[training_indices] - 1

x_val <- data[validation_indices, ]
y_val <- labels[validation_indices] - 1

# run the model

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 17, input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>%
  # layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 17, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

# prepdict the nottinghamshire data

n_comments <- trustData %>% 
  filter(!is.na(Improve), Division %in% 0) %>%
  # filter(str_count(Improve, '\\w+') >= 5) %>% 
  sample_n(1000) %>% 
  pull(Improve)

n_tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(n_comments)

n_sequences <- texts_to_sequences(n_tokenizer, n_comments)

n_data <- pad_sequences(n_sequences, maxlen = maxlen)

n_pred <- model %>% predict(n_data, batch_size = 128)
# n_pred = round(n_pred)

# produice a dataframe of predictions

n_predict_df <- data.frame("Improve" = n_comments) %>% 
  mutate(prediction = apply(n_pred, 1, which.max)) %>% 
  mutate(prediction = keep_variables[prediction])

write.csv(n_predict_df, file = "predict_nottingham.csv")

## validation dataset 

pred <- model %>% predict(x_val, batch_size = 128)
Y_pred = round(pred)

# Confusion matrix

(CM = table(apply(pred, 1, which.max) -1, y_val))

cm <- as.matrix(as.data.frame.matrix(CM))
cat(paste0('Accurary is ', round(sum(diag(cm)) / sum(cm) * 100), '%'))

# produce a dataframe of predictions

predict_df <- all_data[validation_indices, ]
predict_df <- predict_df %>% 
  mutate(prediction = apply(pred, 1, which.max)) %>% 
  mutate(prediction = keep_variables[prediction])

final_prediction = predict_df %>% 
  select(Improve, prediction)

write.csv(final_prediction, file = "predict.csv")

# now repeat with Leicester data

leicester = read_csv("leicester.csv")

l_tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(leicester$Improve)

l_sequences <- texts_to_sequences(l_tokenizer, leicester$Improve)

l_data <- pad_sequences(l_sequences, maxlen = maxlen)

l_pred <- model %>% predict(l_data, batch_size = 128)
l_pred = round(l_pred)

# produice a dataframe of predictions

l_predict_df <- leicester %>% 
  mutate(prediction = apply(l_pred, 1, which.max)) %>% 
  mutate(prediction = keep_variables[prediction])

write.csv(l_predict_df, file = "predict_leicester.csv")