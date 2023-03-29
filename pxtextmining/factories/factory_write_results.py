import pickle
import os

from tensorflow.keras import Model, Sequential


def write_multilabel_models_and_metrics(models, model_metrics, path):
    """Saves models and their associated performance metrics into a specified folder

    Args:
        models (list): List containing the trained tf.keras or sklearn models to be saved.
        model_metrics (list): List containing the model metrics in `str` format
        path (str): Path where model is to be saved.
    """
    for i in range(len(models)):
        model_name = f"model_{i}"
        if not os.path.exists(path):
            os.makedirs(path)
        fullpath = os.path.join(path, model_name)
        if isinstance(models[i], (Sequential, Model)):
            models[i].save(fullpath)
        else:
            modelpath = os.path.join(path, model_name + '.sav')
            pickle.dump(models[i], open(modelpath, "wb"))
        # Write performance metrics file
        txtpath = os.path.join(path, model_name + '.txt')
        with open(txtpath, "w") as file:
            file.write(model_metrics[i])
    print(f"{len(models)} models have been written to {path}")
