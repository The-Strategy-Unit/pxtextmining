cat(
  "The dataset will be split into training and test sets.
The training set will be used to tune the pipeline.
The test set will be used to assess the model's performance on unseen data.
The training set is normally 66-75% the size of the original dataset.\n"
)
fun <- function() {
  training_frac <- readline(
    "Please specify the size of the training set as a proportion (e.g. 0.67): "
  )
  training_frac <- as.numeric(as.character(training_frac))
  return(training_frac)
}
training_frac <- if(interactive()) fun()

while (
  is.na(training_frac) | 
  training_frac >= 1 | 
  training_frac <= 0
) {
  cat('Invalid training set size value! Please provide a number greater than 0 and smaller than 1. \n')
  training_frac <- if(interactive()) fun()
}