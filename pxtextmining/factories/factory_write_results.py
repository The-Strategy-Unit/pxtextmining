import numpy as np
import re
import os
from os import path
import pandas as pd
import shutil
import pickle
# import feather
from sqlalchemy import create_engine
from tensorflow.keras import Sequential, Model

def write_multilabel_models_and_metrics(models, model_metrics, path, dummy=False):
    for i in range(len(models)):
        model_name = f'model_{i}'
        if isinstance(models[i], (Sequential, Model)):
            models[i].save(f'{path}/{model_name}')
        else:
            pickle.dump(models[i], open(f'{path}/{model_name}.sav', 'wb'))
        with open(f'{path}/{model_name}.txt', 'w') as file:
            file.write(model_metrics[i])
    if len(model_metrics) != len(models):
        with open(f'{path}/dummy_metrics.txt', 'w') as file:
            file.write(model_metrics[-1])
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
