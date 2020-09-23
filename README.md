# positive_about_change_text_mining

This GitHub project is _still experimental_ and, at this stage, aims to build a number of different Machine Learning pipelines for text data. A few avenues that will be explored are the following:
1. Benchmarking of different algorithms with R package [`mlr3`](https://github.com/mlr-org])
2. Facebook's [StarSpace](https://github.com/facebookresearch/StarSpace) with R package [`ruimtehol`](https://github.com/bnosac/ruimtehol)
3. [`Quanteda`'s](https://quanteda.io/index.html) implementation of multinomial Naive Bayes (https://tutorials.quanteda.io/machine-learning/nb/) 
4. Some other relevant package that is available in R (e.g. a Keras package)
4. Python!

The data is [here](https://github.com/ChrisBeeley/naturallanguageprocessing/blob/master/cleanData.Rdata) and that's where it will stay until GitHub stops crashing when I try to upload them to this project!

At this experimental stage, don't be surprised if the chunks of the code contain errors, are cryptic or don't work at all, or if the models aren't appropriate or don't perform _that_ great. Consider this repo as an interface for sharing work with my colleagues, until we make some big announcement along the lines "World, look what a great pipeline we've built!".

## `mlr3` (R)
The pipeline performs data pre-processing (e.g. one-hot encode dates, if a date column is used; tokenize text and get word frequencies; etc.) and then benchmarks a number of classification algorithms. Five algorithms were considered:

| Model                                                 | Issues      | Verdict     |
| :-------------                                        | :---------- | ----------- |
| Generalized Linear Models with Elastic Net (GLM NET) | Something goes wrong inside the pipeline. It seems like it gets confused because GLM NET drops out irrelevant features during training, so the pipeline throws an error when it finds these variables in the test set.  | Investigate issue and consider implementing the model.    |
| Naive Bayes | `mlr3learners` implements `e1071::naiveBayes` which can be [terribly slow](https://stackoverflow.com/questions/54427001/naive-bayes-in-quanteda-vs-caret-wildly-different-results) with sparse data like text data. I may try to add `quanteda.textmodels::textmodel_nb` to `mlr3extralearners`, because it is a freakishly fast multinomial Naive Bayes model that is designed for text data.  | Don't implement, **unless** I manage to add `quanteda.textmodels::textmodel_nb` to `mlr3extralearners`. Alternatively, some fast implementation of a multinomial or kernel-based Naive Bayes model may be a reasonable alternative? |
| Random Forest | It appears that it the use of Random Forest with sparse data can be problematic. See [this video](https://www.youtube.com/watch?v=Sz8RB_fPYOk) (54' 10'') and [this resource](https://stats.stackexchange.com/questions/28828/is-there-a-random-forest-implementation-that-works-well-with-very-sparse-data).   | Don't implement.    |
| XGBoost | The most popular boosted tree algorithm nowadays. Can handle sparse data.   | Implement.    |

The script that runs the whole process (from data loading and prep to model benchmarking and results evaluation) is `mlr3_run_pipeline.R` and consists of four lines of code. Run each line of this code individually to familiarize yourselves with the process.

As a starter, the answers to the prompts in `mlr3_prepare_test_and_training_tasks.R` should be as follows:

1. pipeline_data
2. nfspf
3. super
4. 0.67

The answers to the prompts in `mlr3_pipeline_optimal_defaults.R` should be as follows:

1. cv
2. 2
3. classif.mbrier
4. 3

You can always change these values, but note that more CV folds and evaluations would mean more computation time and memory usage.

### Preliminary findings


## StarSpace `ruimtehol` (R)
As a starter, script `starspace.R` prepares the data in the appropriate format and builds a simple supervised model from which embeddings and other useful information (e.g. word clouds for each tag) can be extracted. The script also produces a rough model accuracy metric with the test data, as well as a T-SNE plot to visually assess how well the model performs on unseen data.
