message(
  "When the dataset is too big, it may be more efficient to run the pipeline on a sample.
The sample should be stratified to ensure even representation of classes in the training and test sets."
)
fun <- function() {
  prop <- readline(
    "Provide a number greater than 0 and at most 1: ")
  prop <- suppressWarnings(as.numeric(prop))
  return(prop)
}
prop <- if(interactive()) fun()

while (
  is.na(prop)  | 
  prop > 1 | 
  prop <= 0
) {
  cat('Invalid entry. Please provide a number greater than 0 and at most 1.\n')
  prop <- if(interactive()) fun()
}