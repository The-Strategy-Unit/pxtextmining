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
from factories.factory_data_load_and_split import factory_data_load_and_split
from factories.factory_pipeline import factory_pipeline
from factories.factory_model_performance import factory_model_performance
from factories.factory_write_results import factory_write_results


def text_classification_pipeline(filename, target, predictor, test_size=0.33,
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
                                 save_objects_to_disk=False,
                                 save_pipeline_as="default",
                                 results_folder_name="results"):

    x_train, x_test, y_train, y_test = factory_data_load_and_split(filename, target, predictor, test_size)

    pipe = factory_pipeline(x_train, y_train, tknz, metric, cv, n_iter, n_jobs, verbose, learners)

    tuning_results, pred, accuracy_per_class, p_compare_models_bar = \
        factory_model_performance(pipe, x_train, y_train, x_test, y_test, metric)

    pred, index_training_data, index_test_data = factory_write_results(pipe, tuning_results, pred,
                                                                       accuracy_per_class, p_compare_models_bar,
                                                                       target, x_train, x_test, metric,
                                                                       objects_to_save,
                                                                       save_objects_to_disk, save_pipeline_as,
                                                                       results_folder_name)

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_training_data, index_test_data
