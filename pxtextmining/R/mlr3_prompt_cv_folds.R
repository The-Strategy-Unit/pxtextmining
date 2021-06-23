fun <- function() {
  if (resampling_strategy %in% c('cv', 'repeated_cv')) {
    cv_folds_or_holdout_sample_size <- readline(
      "Provide number of folds for the cross-validation: ")
    cv_folds_or_holdout_sample_size <- 
      suppressWarnings(as.integer(cv_folds_or_holdout_sample_size))
    return(cv_folds_or_holdout_sample_size)
  } else if (resampling_strategy %in% c('holdout')) {
    cv_folds_or_holdout_sample_size <- readline(
      "Provide the holdout sample size (greater than 0 and smaller than 1): ")
    cv_folds_or_holdout_sample_size <- 
      suppressWarnings(as.numeric(cv_folds_or_holdout_sample_size))
    return(cv_folds_or_holdout_sample_size)
  }
}
cv_folds_or_holdout_sample_size <- if(interactive()) fun()

if (resampling_strategy %in% c('cv', 'repeated_cv')) {
  while (is.na(cv_folds_or_holdout_sample_size)) {
    message('Invalid entry. Has to be an integer.\n')
    cv_folds_or_holdout_sample_size <- if(interactive()) fun()
  } 
} else if (resampling_strategy %in% c('holdout')) {
  while (
    is.na(cv_folds_or_holdout_sample_size) | 
    cv_folds_or_holdout_sample_size >= 1 | 
    cv_folds_or_holdout_sample_size <= 0
) {
    message('Invalid entry. Has to be greater than 0 and smaller than 1.\n')
    cv_folds_or_holdout_sample_size <- if(interactive()) fun()
  }
}