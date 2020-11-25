Sys.setenv(RETICULATE_PYTHON = "C:/Users/andreas.soteriades/Anaconda3/envs/textminingpy38/python.exe")

library(reticulate)
use_python("C:/Users/andreas.soteriades/Anaconda3/envs/textminingpy38/python.exe")
use_condaenv("textminingpy38", required = TRUE)
py_config()

py_run_file('text_mining_import_libraries.py')
py_run_file('text_mining_custom_functions_and_classes.py')
py_run_file('text_mining_load_and_prepare_data.py')
py_run_file('text_mining_tfidf_matrix.py')
py_run_string('stop_words = list(nlp.Defaults.stop_words)')

py$tfidf_matrix %>%
  mutate(pred = py$y_test) %>%
  select(-any_of(py$stop_words)) %>% # Removes single-word stop word tokens
  filter(pred == "Smoking") %>%
  reshape2::melt() %>%
  filter(value > 0) %>%
  group_by(variable) %>%
  mutate(value = sum(value)) %>%
  ungroup() %>%
  distinct %>%
  slice_max(value, n = 15) %>%
  ggplot(aes(value, reorder(variable, value))) +
  geom_col() +
  labs(x = "tf-idf", y = NULL) + 
  theme_bw()
