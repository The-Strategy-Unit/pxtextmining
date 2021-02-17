import numpy as np
import os
from os import path
import shutil
import pickle
from factories.factory_data_prepros import factory_data_prepros
from factories.factory_pipeline import factory_pipeline
from factories.factory_model_performance import factory_model_performance


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
                                 save_pipeline_as="default",
                                 results_folder_name="results"):

    x_train, x_test, y_train, y_test = factory_data_prepros(filename, target, predictor, test_size)

    pipe = factory_pipeline(x_train, y_train, tknz=tknz, metric=metric,
                            cv=cv, n_iter=n_iter, n_jobs=n_jobs,
                            verbose=verbose,
                            learners=learners)

    tuning_results, pred, accuracy_per_class, p_compare_models_bar = \
        factory_model_performance(pipe=pipe, x_train=x_train, y_train=y_train,
                                  x_test=x_test, y_test=y_test,
                                  metric=metric)

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
    if "tuning results" in objects_to_save:
        aux = path.join(results_file, "tuning_results.csv")
        tuning_results.to_csv(aux, index=False)
    if "predictions" in objects_to_save:
        aux = path.join(results_file, "predictions.txt")
        np.savetxt(aux, pred, fmt="%s")
    if "accuracy per class" in objects_to_save:
        aux = path.join(results_file, "accuracy_per_class.csv")
        accuracy_per_class.to_csv(aux, index=False)
    if "index - training data" in objects_to_save:
        aux = path.join(results_file, "index_training_data.txt")
        np.savetxt(aux, x_train.index + 1, fmt="%d")
    if "index - test data" in objects_to_save:
        aux = path.join(results_file, "index_test_data.txt")
        np.savetxt(aux, x_test.index + 1, fmt="%d")
    if "bar plot" in objects_to_save:
        aux = path.join(results_file, "p_compare_models_bar_" + metric + ".png")
        p_compare_models_bar.figure.savefig(aux)

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, x_train.index, x_test.index
