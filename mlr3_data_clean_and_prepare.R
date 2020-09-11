load('cleanData.RData')

pipeline_data <- trustData %>%
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
  as_tibble %>%
  slice_sample(prop = 0.1)

names(pipeline_data)