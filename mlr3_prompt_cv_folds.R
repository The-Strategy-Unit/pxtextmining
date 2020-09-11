fun <- function() {
  cv_folds <- readline(
    "Provide number of folds for the cross-validation: ")
  cv_folds <- suppressWarnings(as.integer(cv_folds))
  return(cv_folds)
}
cv_folds <- if(interactive()) fun()

while (is.na(cv_folds)) {
  cat('Invalid entry. Has to be an integer.\n')
  cv_folds <- if(interactive()) fun()
}