import numpy as np
import re
import os
from os import path
import pandas as pd
import shutil
import pickle
# import feather
# import mysql.connector
from sqlalchemy import create_engine
from pxtextmining.factories.factory_data_load_and_split import factory_data_load_and_split
from pxtextmining.factories.factory_pipeline import factory_pipeline
from pxtextmining.factories.factory_model_performance import factory_model_performance
from pxtextmining.factories.factory_write_results import factory_write_results


def text_classification_pipeline(filename, target, predictor, test_size=0.33,
                                 ordinal=False,
                                 tknz="spacy",
                                 metric="class_balance_accuracy_score",
                                 cv=5, n_iter=100, n_jobs=5, verbose=3,
                                 learners=["SGDClassifier"],
                                 objects_to_save=[
                                     "pipeline",
                                     "tuning results",
                                     "predictions",
                                     "accuracy per class",
                                     "index - training data",
                                     "index - test data",
                                     "bar plot"
                                 ],
                                 save_objects_to_server=True,
                                 save_objects_to_disk=False,
                                 save_pipeline_as="default",
                                 results_folder_name="results"):

    """

    :param str filename: Dataset name (CSV), including the data type suffix. If None, data is read from the database.
    :param str target: Name of the response variable.
    :param str predictor: Name of the predictor variable.
    :param float test_size: Proportion of data that will form the test dataset.
    :param bool ordinal: Whether to fit an ordinal classification model. The ordinal model is the implementation of
        `Frank and Hall (2001) <https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf>`_ that can use any standard classification model.
    :param str tknz: Tokenizer to use ("spacy" or "wordnet").
    :param str metric: Scorer to use during pipeline tuning ("accuracy_score", "balanced_accuracy_score",
        "matthews_corrcoef", "class_balance_accuracy_score").
    :param int cv: Number of cross-validation folds.
    :param int n_iter: Number of parameter settings that are sampled
        (see `sklearn.model_selection.RandomizedSearchCV
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_).
    :param int n_jobs: Number of jobs to run in parallel (see ``sklearn.model_selection.RandomizedSearchCV``).
    :param int verbose: Controls the verbosity (see ``sklearn.model_selection.RandomizedSearchCV``).
    :param list[str] learners: A list of ``Scikit-learn`` names of the learners to tune. Must be one or more of
        "SGDClassifier", "RidgeClassifier", "Perceptron", "PassiveAggressiveClassifier", "BernoulliNB", "ComplementNB",
        "MultinomialNB", "KNeighborsClassifier", "NearestCentroid", "RandomForestClassifier".
    :param list[str] objects_to_save: Objects to save following pipeline fitting and assessment. These are:
        the pipeline (SAV file);
        table with all (hyper)parameter values tried out and performance indicators on the cross-validation data;
        table with predictions on the test set;
        table with accuracy and counts per class;
        row indices for the training set;
        row indices for the test set;
        bar plot with the best-performing models- plotted values are the mean scores from a k-fold CV on the training
        set, for the best (hyper)parameter values for each learner.
    :param bool save_objects_to_server:
    :param bool save_objects_to_disk:
    :param str save_pipeline_as: Save the pipeline as "save_pipeline_as.sav".
    :param str results_folder_name: Name of folder in which to save the results. It will create a new folder or
        overwrite and existing one of the same name.
    """

    x_train, x_test, y_train, y_test = factory_data_load_and_split(filename, target, predictor, test_size)

    pipe = factory_pipeline(ordinal, x_train, y_train, tknz, metric, cv, n_iter, n_jobs, verbose, learners)

    pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar = \
        factory_model_performance(pipe, x_train, y_train, x_test, y_test, metric)

    pred, index_training_data, index_test_data = factory_write_results(pipe, tuning_results, pred,
                                                                       accuracy_per_class, p_compare_models_bar,
                                                                       target, x_train, x_test, metric,
                                                                       objects_to_save,
                                                                       save_objects_to_server,
                                                                       save_objects_to_disk, save_pipeline_as,
                                                                       results_folder_name)

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_training_data, index_test_data
