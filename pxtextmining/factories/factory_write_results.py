import numpy as np
import re
import os
from os import path
import pandas as pd
import shutil
import pickle
# import feather
from sqlalchemy import create_engine


def write_multilabel_models_and_metrics(models, model_metrics, path):
    for i in range(len(models)):
        model_name = f'model_{i}'
        pickle.dump(models[i], open(f'{path}/{model_name}.sav', 'wb'))
        with open(f'{path}/{model_name}.txt', 'w') as file:
            file.write(model_metrics[i])
    print(f'{len(models)} models have been written to {path}')


def write_model_summary(results_file, model_summary):
    """Function to write a .txt file containing the model summary information.

    Args:
        results_file (str): Filepath for the output of the pipeline to be saved.
        model_summary (dict): Model metadata to be written.
    """
    model_summary_file = path.join(results_file, "model_summary.txt")
    with open(model_summary_file, 'w') as f:
        for k, v in model_summary.items():
            f.write(f'{k}: \n {v} \n\n')

def factory_write_results(pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar,
                          target, index_training_data, index_test_data, model_summary,
                          save_objects_to_server=False,
                          save_objects_to_disk=True,
                          results_folder_name="results"):

    """
    Write the fitted pipeline and associated files. Writes 7 files:

    - The fitted pipeline (SAV);
    - All (hyper)parameters tried during fitting and the associated pipeline performance metrics (CSV);
    - The predictions on the test set (CSV);
    - Accuracies per class (CSV);
    - The row indices of the training data (CSV);
    - The row indices of the test data (CSV);
    - A bar plot comparing the mean scores of performance metrics from the cross-validation on
      the training set, for the best (hyper)parameter values for each learner (PNG);
    - The model summary (txt)

    :param estimator pipe: Fitted model or pipeline
    :param pandas.core.frame.DataFrame tuning_results: All (hyper)parameter values and models tried during fitting.
    :param pandas.core.frame.DataFrame pred: The predictions on the test set.
    :param pandas.core.frame.DataFrame accuracy_per_class: Accuracies per class.
    :param png p_compare_models_bar: A bar plot comparing the mean scores of performance metrics from the
        cross-validation on the training set, for the best hyperparameter values for each learner.
    :param str target: Name of the response variable.
    :param pandas.core.frame.DataFrame x_train: The training dataset.
    :param pandas.core.frame.DataFrame x_test: The test dataset.
    :param bool save_objects_to_server: Whether to save the results to the server. **NOTE:** The feature that writes
        results to the database is for internal use only. Experienced users who would like to write the data to their
        own databases can, of course, achieve that by slightly modifying the relevant lines in the script.
    :param bool save_objects_to_disk: Whether to save the results to disk. See ``results_folder_name``.
    :param str results_folder_name: Name of the folder that will contain all saved results. If the folder already exists, it will be overwritten.
    :return: A ``tuple`` of length 3, containing:
        The fitted pipeline (SAV); All (hyper)parameters tried during fitting and the associated pipeline performance
        metrics (CSV); The predictions on the test set (CSV); Accuracies per class (CSV);
        The row indices of the training data (CSV); The row indices of the test data (CSV);
        A bar plot comparing the mean scores of performance metrics from the cross-validation on
        the training set, for the best (hyper)parameter values for each learner (PNG)
    :rtype: tuple
    """

    index_training_data = pd.DataFrame(index_training_data, columns=["row_index"])
    index_test_data = pd.DataFrame(index_test_data, columns=["row_index"])

    pred = pd.DataFrame(pred, columns=[target + "_pred"])
    pred["row_index"] = index_test_data

    # ====== Write results to database ====== #
    if save_objects_to_server:
        # Pull database name & host and user credentials from my.conf file
        conf = open('my.conf').readlines()
        conf.pop(0)
        for i in range(len(conf)):
            match = re.search('=(.*)', conf[i])
            conf[i] = match.group(1).strip()

        # Connect to mysql by providing a sqlachemy engine
        engine = create_engine(
            "mysql+mysqlconnector://" + conf[2] + ":" + conf[3] + "@" + conf[0] + "/" + conf[1],
            echo=False)

        # Write results to database
        print("Writing to database...")

        tuning_results.to_sql(name="tuning_results_" + target, con=engine, if_exists="replace", index=False)

        index_training_data.to_sql(name="index_training_data_" + target, con=engine, if_exists="replace",
                                       index=False)

        index_test_data.to_sql(name="index_test_data_" + target, con=engine, if_exists="replace", index=False)

        # aux = pd.DataFrame(index_training_data, columns=["row_index"])
        # pred = pd.concat([pred.reset_index(drop=True), aux], axis=0)
        pred.to_sql(name="predictions_test_" + target, con=engine, if_exists="replace", index=False)

        accuracy_per_class.to_sql(name="accuracy_per_class_" + target, con=engine, if_exists="replace", index=False)

    # ====== Write results to disk ====== #
    if save_objects_to_disk:
        print("Writing to disk...")

        results_file = results_folder_name
        if os.path.exists(results_file):
            shutil.rmtree(results_file)
        os.makedirs(results_file)

        aux = "pipeline_" + target + ".sav"
        save_pipeline_as = path.join(results_file, aux)
        pickle.dump(pipe, open(save_pipeline_as, "wb"))

        aux = path.join(results_file, "tuning_results_" + target + ".csv")
        tuning_results.to_csv(aux, index=False)

        aux = path.join(results_file, "predictions_" + target + ".csv")
        pd.DataFrame(pred).to_csv(aux, index=False)

        aux = path.join(results_file, "accuracy_per_class_" + target + ".csv")
        accuracy_per_class.to_csv(aux, index=False)

        aux = path.join(results_file, "index_training_data_" + target + ".csv")
        index_training_data.to_csv(aux, index=False)

        aux = path.join(results_file, "index_test_data_" + target + ".csv")
        index_test_data.to_csv(aux, index=False)

        aux = path.join(results_file, "p_compare_models_bar.png")
        p_compare_models_bar.figure.savefig(aux)

    # Write performance metrics
    write_model_summary(results_file, model_summary)

    # db = mysql.connector.connect(option_files="my.conf", use_pure=True)
    # sql_query = [
    #    """CREATE TABLE result20 AS
    #            (SELECT textData.*,""",
    #            "predictions." + target + "_pred",
    #    """FROM textData
    #            LEFT JOIN predictions
    #            ON predictions.row_index = textData.row_index)"""
    # ]
    # with db.cursor() as cursor:
    #    cursor.execute(" ".join(sql_query))

    return pred, index_training_data, index_test_data
