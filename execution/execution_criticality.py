from pxtextmining.pipelines.text_classification_pipeline import text_classification_pipeline

pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_train, index_test = \
    text_classification_pipeline(filename='datasets/text_data.csv', target="criticality", predictor="feedback",
                                 test_size=0.33,
                                 ordinal=True,
                                 tknz="spacy",
                                 metric="class_balance_accuracy",
                                 cv=2, n_iter=1, n_jobs=5, verbose=1,
                                 learners=[
                                     "SGDClassifier",
                                     "RidgeClassifier",
                                     "Perceptron",
                                     "PassiveAggressiveClassifier",
                                     "BernoulliNB",
                                     "ComplementNB",
                                     "MultinomialNB",
                                     "KNeighborsClassifier",
                                     "NearestCentroid",
                                     "RandomForestClassifier"
                                     ],
                                 objects_to_save=[
                                     "pipeline",
                                     "tuning results",
                                     "predictions",
                                     "accuracy per class",
                                     "index - training data",
                                     "index - test data",
                                     "bar plot"
                                 ],
                                 save_objects_to_server=False,
                                 save_objects_to_disk=True,
                                 save_pipeline_as="test_pipeline_criticality",
                                 results_folder_name="test_results_criticality",
                                 reduce_criticality=True,
                                 theme="label")
