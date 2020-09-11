# positive_about_change_text_mining

This GitHub project is still experimental and, at this stage, aims to build a number of different Machine Learning pipelines for text data. A few avenues that will be explored are the following:
1. Benchmarking of different algorithms with R package [`mlr3`](https://github.com/mlr-org])
2. Facebook's [StarSpace](https://github.com/facebookresearch/StarSpace) with R package [`ruimtehol`](https://github.com/bnosac/ruimtehol)
3. Some other relavant package that is available in R (e.g. a Keras package)
4. Python

## `mlr3`
Working code is already available. The script that runs the whole process (from data loading and prep to model benchmarking and results evaluation) is `mlr3_run_pipeline.R` and consists of four lines of code. Run each line of this code individually to familiarize yourselves with the process. 

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
