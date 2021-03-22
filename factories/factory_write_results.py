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
                          target, x_train, x_test, metric,
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
    print("Writing to database...")
    if "tuning results" in objects_to_save:
        tuning_results.to_sql(name="tuning_results_" + target, con=engine, if_exists="replace", index=False)

    index_training_data = pd.DataFrame(x_train.index, columns=["row_index"])
    if "index - training data" in objects_to_save:
        index_training_data.to_sql(name="index_training_data_" + target, con=engine, if_exists="replace", index=False)
    
    index_test_data = pd.DataFrame(x_test.index, columns=["row_index"])
    if "index - test data" in objects_to_save:
        index_test_data.to_sql(name="index_test_data_" + target, con=engine, if_exists="replace", index=False)

    pred = pd.DataFrame(pred, columns=[target + "_pred"])
    pred["row_index"] = index_test_data
    # aux = pd.DataFrame(index_training_data, columns=["row_index"])
    # pred = pd.concat([pred.reset_index(drop=True), aux], axis=0)
    if "predictions" in objects_to_save:
        pd.DataFrame(pred).to_sql(name="predictions_test_" + target, con=engine, if_exists="replace", index=False)

    if "accuracy per class" in objects_to_save:
        accuracy_per_class.to_sql(name="accuracy_per_class_" + target, con=engine, if_exists="replace", index=False)

    # Write results to disk
    if objects_to_save:
        print("Writing to disk...")

    results_file = results_folder_name
    if os.path.exists(results_file):
        shutil.rmtree(results_file)
    os.makedirs(results_file)

    if "pipeline" in objects_to_save:
        if save_pipeline_as == "default":
            aux = "finalized_model_" + target + ".sav"
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
        pred.to_csv(aux, index=False)
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
