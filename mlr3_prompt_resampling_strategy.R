cat(
  "We need a resampling strategy to tune the pipeline. There are several alternatives, which we have here narrowed down to the following two:\n
1. k-fold cross-validation (CV): simple CV that splits the data into 10 folds.\n
2. Repeated k-fold cross-validation: like the above process, although repeated k times.\n"
)
fun <- function() {
  resampling_strategy <- readline(
    "Provide name of resampling strategy to be used in the exercise.
    Type 'cv (unquoted) or 'repeated_cv' (unquoted): ")
  resampling_strategy <- as.character(resampling_strategy)
  return(resampling_strategy)
}
resampling_strategy <- if(interactive()) fun()

while (!resampling_strategy %in% c('cv', 'repeated_cv')) {
  cat("Invalid entry. Possible values are 'cv' or 'repeated_cv' (unquoted)\n")
  resampling_strategy <- if(interactive()) fun()
}