message(
  "The name of the resonse variable needs to be specified.\n
Note that, because the column names in the data have now been formatted to an R-friendly format, said names may now be different from the original ones.\n
For example, if the original name of the response variable was 'Response.Variable',\n
it has now been formatted to 'response_variable'.\n
To see how the response variable's name looks like now, refer to the data frame\n
that is now being displayed as 'Dataset - first 6 rows'.\n"
)
fun <- function() {
  response_name <- readline(
    "Please specify the name (unquoted) of the response variable: "
  )
  response_name <- as.character(response_name)
  return(response_name)
}
response_name <- if(interactive()) fun()

while (!response_name %in% names(dataset)) {
  message('Response variable provided is not in the dataset! Please provide a valid name (unquoted).\n')
  response_name <- if(interactive()) fun()
}