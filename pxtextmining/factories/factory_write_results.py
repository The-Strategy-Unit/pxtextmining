import numpy as np
import re
import os
from os import path
import pandas as pd
import shutil
import pickle
# import feather
from sqlalchemy import create_engine


def factory_write_results(pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar,
                          target, x_train, x_test, index_training_data, index_test_data, metric,
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
                          save_pipeline_as="default",
                          results_folder_name="results"):

    """
    Write the fitted pipeline and associated files. Writes between 1 to 7 files, depending on the value of argument
    ``objects_to_save``:

    - The fitted pipeline (SAV);
    - All (hyper)parameters tried during fitting and the associated pipeline performance metrics (CSV);
    - The predictions on the test set (CSV);
    - Accuracies per class (CSV);
    - The row indices of the training data (CSV);
    - The row indices of the test data (CSV);
    - A bar plot comparing the mean scores (of the user-supplied metric parameter) from the cross-validation on
      the training set, for the best (hyper)parameter values for each learner (PNG);

    :param pipe: Fitted `sklearn.pipeline.Pipeline
        <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_/
        `imblearn.pipeline.Pipeline
        <https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html#imblearn.pipeline.Pipeline>`_
    :param pandas.core.frame.DataFrame tuning_results: All (hyper)parameter values and models tried during fitting.
    :param pandas.core.frame.DataFrame pred: The predictions on the test set.
    :param pandas.core.frame.DataFrame accuracy_per_class: Accuracies per class.
    :param p_compare_models_bar: A bar plot comparing the mean scores (of the user-supplied metric parameter) from the
        cross-validation on the training set, for the best hyperparameter values for each learner.
    :param str target: Name of the response variable.
    :param pandas.core.frame.DataFrame x_train: The training dataset.
    :param pandas.core.frame.DataFrame x_test: The test dataset.
    :param str metric: Scorer that was used in pipeline tuning ("accuracy_score",
        "balanced_accuracy_score", "matthews_corrcoef" or "class_balance_accuracy_score").
    :param list[str] objects_to_save: The objects to save. Should be one or more of "pipeline", "tuning results",
        "predictions", "accuracy per class", "index - training data", "index - test data", "bar plot".
    :param bool save_objects_to_server: Whether to save the results to the server. **NOTE:** The feature that writes
        results to the database is for internal use only. Experienced users who would like to write the data to their
        own databases can, of course, achieve that by slightly modifying the relevant lines in the script. A "my.conf"
        file will need to be placed in the root, with five lines, as follows (without the ";", "<" and ">"):

        - [connector_python];
        - host = <host_name>;
        - database = <database_name>;
        - user = <username>;
        - password = <password>;
    :param bool save_objects_to_disk: Whether to save the results to disk. See ``results_folder_name``.
    :param str save_pipeline_as: Name of saved pipeline. If "default", then it will be saved as
        ``'pipeline_' + target + '.sav'``.
    :param str results_folder_name: Name of the folder that will contain all saved results specified in
        ``objects_to_save``. If the folder already exists, it will be overwritten.
    :return: A ``tuple`` of length 3 with the following ``pandas.DataFrame`` objects:

            - The predictions on the test set;
            - The row indices of the training data;
            - The row indices of the test data;
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

        if "tuning results" in objects_to_save:
            tuning_results.to_sql(name="tuning_results_" + target, con=engine, if_exists="replace", index=False)

        if "index - training data" in objects_to_save:
            index_training_data.to_sql(name="index_training_data_" + target, con=engine, if_exists="replace",
                                       index=False)

        if "index - test data" in objects_to_save:
            index_test_data.to_sql(name="index_test_data_" + target, con=engine, if_exists="replace", index=False)

        # aux = pd.DataFrame(index_training_data, columns=["row_index"])
        # pred = pd.concat([pred.reset_index(drop=True), aux], axis=0)
        if "predictions" in objects_to_save:
            pred.to_sql(name="predictions_test_" + target, con=engine, if_exists="replace", index=False)

        if "accuracy per class" in objects_to_save:
            accuracy_per_class.to_sql(name="accuracy_per_class_" + target, con=engine, if_exists="replace", index=False)

    # ====== Write results to disk ====== #
    if objects_to_save:
        print("Writing to disk...")

    results_file = results_folder_name
    if os.path.exists(results_file):
        shutil.rmtree(results_file)
    os.makedirs(results_file)

    if "pipeline" in objects_to_save:
        if save_pipeline_as == "default":
            aux = "pipeline_" + target + ".sav"
            save_pipeline_as = path.join(results_file, aux)
        else:
            aux = save_pipeline_as + ".sav"
            save_pipeline_as = path.join(results_file, aux)
        pickle.dump(pipe, open(save_pipeline_as, "wb"))

    if "tuning results" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "tuning_results_" + target + ".csv")
        tuning_results.to_csv(aux, index=False)
        # aux = path.join(results_file, "tuning_results")
        # feather.write_dataframe(tuning_results, aux)

    if "predictions" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "predictions_" + target + ".csv")
        pd.DataFrame(pred).to_csv(aux, index=False)
        # aux = path.join(results_file, "predictions")
        # feather.write_dataframe(pd.DataFrame(pred), aux)

    if "accuracy per class" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "accuracy_per_class_" + target + ".csv")
        accuracy_per_class.to_csv(aux, index=False)
        # aux = path.join(results_file, "accuracy_per_class")
        # feather.write_dataframe(accuracy_per_class, aux)

    if "index - training data" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "index_training_data_" + target + ".csv")
        index_training_data.to_csv(aux, index=False)
        # aux = path.join(results_file, "index_training_data")
        # feather.write_dataframe(pd.DataFrame(index_training_data), aux)

    if "index - test data" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "index_test_data_" + target + ".csv")
        index_test_data.to_csv(aux, index=False)
        # aux = path.join(results_file, "index_test_data")
        # feather.write_dataframe(pd.DataFrame(index_test_data), aux)

    if "bar plot" in objects_to_save:
        aux = path.join(results_file, "p_compare_models_bar_" + metric + "_" + target + ".png")
        p_compare_models_bar.figure.savefig(aux)

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
