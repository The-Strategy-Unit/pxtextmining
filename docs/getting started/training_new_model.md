# Training a new model

To train a new model to categorise patient feedback text, labelled data is required. Discussions are currently underway to enable the release of the data that the multilabel models in `pxtextmining` are trained on.

This page breaks down the steps in the function `pxtextmining.pipelines.run_sklearn_pipeline`, which outputs trained sklearn models. This is a high-level explanation of the processes; for more detailed technical information please see the relevant code reference pages for each function.


```python

# Step 1: Generate a random_state which is used for the train_test_split.
# This means that the pipeline and evaluation should be reproducible.
random_state = random.randint(1,999)

# Step 2: Load the data and isolate the target columns from the dataframe.
df = load_multilabel_data(filename = 'datasets/hidden/multilabeldata_2.csv',
                          target = 'major_categories')

# Step 3: Conduct preprocessing: remove punctuation and numbers, clean whitespace and drop empty lines.
# Split into train and test using the random_state above.
X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(
                                        df, target = target,
                                        random_state = random_state)

# Step 4: Instantiate a pipeline and hyperparamter grid for each estimator to be tried.
# Conduct a cross-validated randomized search to identify the hyperparameters
  # producing the best results on the validation set.
# For each estimator, returns the pipeline with the best hyperparameters,
  # together with the time taken to search the pipeline.
models, training_times = search_sklearn_pipelines(X_train, Y_train,
                                        models_to_try = models_to_try,
                                        additional_features = additional_features)

# Step 5: Evaluate each pipeline using the test set, comparing predicted values with real values.
# Performance metrics are recorded together with the time taken to search the pipeline.
model_metrics = []
for i in range(len(models)):
    m = models[i]
    t = training_times[i]
    model_metrics.append(get_multilabel_metrics(X_test, Y_test,
                                        random_state = random_state,
                                        labels = target, model_type = 'sklearn',
                                        model = m, training_time = t))

# Step 6: Save the models and performance metrics to the path specified
write_multilabel_models_and_metrics(models,model_metrics,path=path)
```
