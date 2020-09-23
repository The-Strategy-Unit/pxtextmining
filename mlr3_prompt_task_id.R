fun <- function() {
  task_id <- readline(
    "Please provide a name for the mlr3 task.
    E.g. if using NHS patient feedback data, the task could be called 'nhspf'"
  )
  task_id <- as.character(task_id)
  return(task_id)
}
task_id <- if(interactive()) fun()

while (
  grepl("'", task_id) | # If user provides a string, e.g. 'a' or "a", readline() will store it as "'a'" or '"a"' respectively.
  grepl('"', task_id)
) {
  message(
    'Name must be an unquoted string!
Have you accidentally placed the name inside quotes or supplied a numeric?\n')
  task_id <- if(interactive()) fun()
}