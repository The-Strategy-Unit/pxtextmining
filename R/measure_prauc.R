PRAUC = R6::R6Class("PRAUC",
  inherit = mlr3::MeasureClassif,
    public = list(
      initialize = function() {
        super$initialize(
          # custom id for the measure
          id = "classif.prauc",
                                     
          # additional packages required to calculate this measure
          packages = c('PRROC'),
          
          # properties, see below
          properties = character(),
          
          # required predict type of the learner
          predict_type = "prob",
          
          # feasible range of values
          range = c(0, 1),
          
          # minimize during tuning?
          minimize = FALSE
        )
      }
    ),
                             
    private = list(
      # custom scoring function operating on the prediction object
      .score = function(prediction, ...) {
        
        truth1 <- as.integer(prediction$truth == levels(prediction$truth)[1])
        PRROC::pr.curve(
          scores.class0 = prediction$prob[, 1], 
          weights.class0 = truth1
        )[[2]]
        
      }
    )
)

mlr3::mlr_measures$add("classif.prauc", PRAUC)

#task_sonar <- tsk('sonar')
#task_sonar$positive <- 'R'
#learner <- lrn('classif.rpart', predict_type = 'prob')
#learner$train(task_sonar)
#pred <- learner$predict(task_sonar)
#pred$score()
#pred$score(msr('classif.auc'))
#pred$score(msr('classif.prauc'))