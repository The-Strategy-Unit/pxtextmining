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
from factories.factory_data_prepros import factory_data_prepros
from factories.factory_pipeline import factory_pipeline
from factories.factory_model_performance import factory_model_performance


def text_classification_pipeline(filename, target, predictor, test_size=0.33,
                                 keep_emojis=True,
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

    x_train, x_test, y_train, y_test = factory_data_prepros(filename, target, predictor, test_size, keep_emojis)

    pipe = factory_pipeline(x_train, y_train, tknz=tknz, metric=metric,
                            cv=cv, n_iter=n_iter, n_jobs=n_jobs,
                            verbose=verbose,
                            learners=learners)

    tuning_results, pred, accuracy_per_class, p_compare_models_bar = \
        factory_model_performance(pipe=pipe, x_train=x_train, y_train=y_train,
                                  x_test=x_test, y_test=y_test,
                                  metric=metric)

    # ====== Write results to database ====== #
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
    if "tuning results" in objects_to_save:
        tuning_results.to_sql(name="tuning_results", con=engine, if_exists="replace", index=False)

    index_training_data = x_train.index
    if "index - training data" in objects_to_save:
        pd.DataFrame(index_training_data).to_sql(name="index_training_data", con=engine,
                                                 if_exists="replace", index=False)
    index_test_data = x_test.index
    if "index - test data" in objects_to_save:
        pd.DataFrame(index_test_data).to_sql(name="index_test_data", con=engine, if_exists="replace", index=False)

    pred = pd.DataFrame(pred, columns=[target + "_pred"])
    pred["row_index"] = index_test_data
    # aux = pd.DataFrame(index_training_data, columns=["row_index"])
    # pred = pd.concat([pred.reset_index(drop=True), aux], axis=0)
    if "predictions" in objects_to_save:
        pd.DataFrame(pred).to_sql(name="predictions", con=engine, if_exists="replace", index=False)

    if "accuracy per class" in objects_to_save:
        accuracy_per_class.to_sql(name="accuracy_per_class", con=engine, if_exists="replace", index=False)

    # Write results to disk
    results_file = results_folder_name
    if os.path.exists(results_file):
        shutil.rmtree(results_file)
    os.makedirs(results_file)
    
    if "pipeline" in objects_to_save:
        if save_pipeline_as == "default":
            aux = "finalized_model_" + ".sav"
            save_pipeline_as = path.join(results_file, aux)
        else:
            aux = save_pipeline_as + ".sav"
            save_pipeline_as = path.join(results_file, aux)
        pickle.dump(pipe, open(save_pipeline_as, "wb"))

    if "tuning results" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "tuning_results.csv")
        tuning_results.to_csv(aux, index=False)
        # aux = path.join(results_file, "tuning_results")
        # feather.write_dataframe(tuning_results, aux)

    if "predictions" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "predictions.csv")
        pred.to_csv(aux, index=False)
        # aux = path.join(results_file, "predictions")
        # feather.write_dataframe(pd.DataFrame(pred), aux)

    if "accuracy per class" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "accuracy_per_class.csv")
        accuracy_per_class.to_csv(aux, index=False)
        # aux = path.join(results_file, "accuracy_per_class")
        # feather.write_dataframe(accuracy_per_class, aux)

    if "index - training data" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "index_training_data.txt")
        np.savetxt(aux, index_training_data, fmt="%d")
        # aux = path.join(results_file, "index_training_data")
        # feather.write_dataframe(pd.DataFrame(index_training_data), aux)

    if "index - test data" in objects_to_save and save_objects_to_disk:
        aux = path.join(results_file, "index_test_data.txt")
        np.savetxt(aux, index_test_data, fmt="%d")
        # aux = path.join(results_file, "index_test_data")
        # feather.write_dataframe(pd.DataFrame(index_test_data), aux)

    if "bar plot" in objects_to_save:
        aux = path.join(results_file, "p_compare_models_bar_" + metric + ".png")
        p_compare_models_bar.figure.savefig(aux)

    #db = mysql.connector.connect(option_files="my.conf", use_pure=True)
    #sql_query = [
    #    """CREATE TABLE result20 AS
    #            (SELECT textData.*,""",
    #            "predictions." + target + "_pred",
    #    """FROM textData
    #            LEFT JOIN predictions
    #            ON predictions.row_index = textData.row_index)"""
    #]
    #with db.cursor() as cursor:
    #    cursor.execute(" ".join(sql_query))

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_training_data, index_test_data
