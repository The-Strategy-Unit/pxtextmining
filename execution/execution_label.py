from pxtextmining.pipelines.text_classification_pipeline import text_classification_pipeline

"""
This is an example of how to train a model that predicts 'label' categories using the labelled dataset
'datasets/text_data.csv'.
The trained model is saved in the folder results_label.
"""

pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_train, index_test = \
    text_classification_pipeline(filename='datasets/text_data.csv', target="label", predictor="feedback",
                                 test_size=0.33,
                                 ordinal=False,
                                 tknz="spacy",
                                 metric="class_balance_accuracy",
                                 cv=5, n_iter=100, n_jobs=5, verbose=3,
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
                                 results_folder_name="results_label",
                                 reduce_criticality=False,
                                 theme=None)
