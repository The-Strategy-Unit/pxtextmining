load('cleanData.RData')

# Add "Couldn't be improved" class
categoriesTable <- categoriesTable %>% 
  bind_rows(
    data.frame(Category = "General", Super = "Couldn't be improved", 
      Number = 4444, type = "both")
  )

source('mlr3_prompt_random_stratified_subset.R')

pipeline_data <- trustData %>%
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.Date, as.POSIXct) %>%
  clean_names %>%
  left_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  #select(super, date, division2, directorate2, improve) %>%
  select(super, improve) %>%
  filter_all(~ !is.na(.)) %>%
  #filter(
  #  !super %in% c('Equality/Diversity', 
  #    'Physical Health', 'Record Keeping', 'Safety', 'MHA', 'Smoking', 'Leave')
  #) %>% # Too few in the data (e.g. 2-3 in 10-30% data samples)
  as_tibble %>%
  slice_sample(prop = prop)

names(pipeline_data)

# Take a look at "Couldn't be improved" text. Worth keeping it in?
pipeline_data %>% 
  filter(
    super == "Couldn't be improved",
    !grepl('nothing', improve, ignore.case = TRUE)
  ) %>%
  View
