# positive_about_change_text_mining

This GitHub project is _still experimental_ and, at this stage, aims to build a number of different Machine Learning pipelines for text data. A few avenues that will be explored are the following:
1. Benchmarking of different algorithms with R package [`mlr3`](https://github.com/mlr-org])
2. Facebook's [StarSpace](https://github.com/facebookresearch/StarSpace) with R package [`ruimtehol`](https://github.com/bnosac/ruimtehol)
3. Some other relevant package that is available in R (e.g. a Keras package)
4. Python

The data is [here](https://github.com/ChrisBeeley/naturallanguageprocessing/blob/master/cleanData.Rdata) and that's where it will stay until GitHub stops crashing when I try to upload them to this project!

At this experimental stage, don't be surprised if the chunks of the code contain errors, are cryptic or don't work at all, or if the models don't perform _that_ great. Consider this repo as an interface for sharing work with my colleagues, until we make some big announcement along the lines "World, look what a great pipeline we've built!".

## `mlr3` (R)
The pipeline performs data preprocessing (e.g. one-hot encode dates, if a date column is used; tokenize text and get word frequencies; etc.) and then benchmarks a number of classification algorithms. Five algorithms are currently implemented, namely, Generalized Linear Models with Elastic Net (GLM NET), Naive Bayes, Random Forest, Support Vector Machines (SVM) and XGBoost. For effiency, all but Naive Bayes and Random Forest have been switched off. Both can deal with high-dimensionality, so it is worth exploring them. 

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

## StarSpace `ruimtehol` (R)
As a starter, script `starspace.R` prepares the data in the appropriate format and builds a simple supervised model from which embeddings and other useful information (e.g. word clouds for each tag) can be extracted. The script also produces a rough model accuracy metric with the test data, as well as a T-SNE plot to visually assess how well the model performs on unseen data.
