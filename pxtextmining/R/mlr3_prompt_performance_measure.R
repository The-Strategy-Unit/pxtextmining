message(
  "Benchmarking of different learners will be done based on an appropriate\n
performance measure. All available performance measures can be found in\n
as.data.table(mlr_measures) (e.g. classif.bbrier, classif.sensitivity etc.).\n"
)
View(as.data.table(mlr_measures))
fun <- function() {
  msr_name <- readline(
    "Type name of the measure (unquoted) to be used in the exercise: "
  )
  
  measure_classif <- c() # Create empty measure because, although tryCatch() will suppress an incorrect entry, no value would otherwise be assigned to measure_classif, causing the while() loop below to throw an error.
  measure_classif$id <- NA
  tryCatch(
    {
      measure_classif <- msr(msr_name)
    }, error = function(e){}
  )
  return(measure_classif)
}
measure_classif <- if(interactive()) fun()

while (
  is.na(measure_classif$id) |
  !measure_classif$id %in% as.data.table(mlr_measures)$key |
  !grepl('classif', measure_classif$id) |
  !(is_empty(unlist(as.data.table(mlr_measures)[measure_classif$id, 'task_properties'])) &
    'multiclass' %in% task$properties)
) {
  message('Invalid entry. Possible issues:\n
1. Typo/spelling mistake.\n
2. Measure provided is for regression. Remember this is a classification exercise!\n
3. Measure provided is for binary classification while task is multiclass.\n'
  )
  measure_classif <- if(interactive()) fun()
}
