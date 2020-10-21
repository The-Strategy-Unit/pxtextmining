fun <- function() {
  df_name <- readline("Type name of data frame (unquoted) to be used in the exercise: ")
  dataset <- assign('dataset', .GlobalEnv[[df_name]], envir = .GlobalEnv)
  return(dataset)
}
dataset <- if(interactive()) fun()

while (!is.data.frame(dataset)) {
  message('Data must be in a data frame, tibble or data table!\n')
  dataset <- if(interactive()) fun()
}