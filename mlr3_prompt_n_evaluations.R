message(
  "Search in the (hyper)parameter space during optimizing will stop after a user-specified number of evaluations.\n
When many hyperparameters need tuning, set to >= 50 for serious data crunching (default is 100).\n
When tuning the class-balancing parameters only, perhaps 45-50 should suffice (so that there are about 15 evaluations for each class-balancing method).\n
For practicing, use a small number, e.g. 2 or 3.\n
To avoid waiting and waiting, tune heavier runs overnight."
)
fun <- function() {
  n_evaluations <- readline(
    "Provide number of evaluations: ")
  n_evaluations <- suppressWarnings(as.integer(n_evaluations))
  return(n_evaluations)
}
n_evaluations <- if(interactive()) fun()

while (is.na(n_evaluations)) {
  message('Invalid entry. Has to be an integer.\n')
  n_evaluations <- if(interactive()) fun()
}