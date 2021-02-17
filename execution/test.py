from pipelines.text_classification_pipeline import text_classification_pipeline

pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_train, index_test = \
    text_classification_pipeline(filename="text_data_4444.csv", target="super", predictor="improve",
                                 test_size=0.33,
                                 tknz="spacy",
                                 metric="class_balance_accuracy",
                                 cv=5, n_iter=2, n_jobs=5, verbose=3,
                                 learners=["SGDClassifier", "Perceptron"],
                                 objects_to_save=[
                                     "pipeline",
                                     "tuning results",
                                     "predictions",
                                     "accuracy per class",
                                     "index - training data",
                                     "index - test data",
                                     "bar plot"
                                 ],
                                 save_pipeline_as="test_pipeline",
                                 results_folder_name="results for Super")
