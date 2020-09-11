load('cleanData.RData')

trustData_andreas <- trustData %>%
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.Date, as.POSIXct) %>%
  clean_names %>%
  left_join(
    select(categoriesTable, Number, Super), 
    by = c('imp1' = 'Number')
  ) %>%
  clean_names %>%
  filter(
    !is.na(super),
    !is.na(improve)
  ) %>%
  select(super, date, division2, directorate2, improve) %>%
  as_tibble %>%
  slice_sample(prop = 0.1)

names(trustData_andreas)

apply(trustData_andreas, 2, function(x) sum(is.na(x)))
