import os
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential

from pxtextmining.factories.factory_model_performance import (
    additional_analysis,
    parse_metrics_file,
)
from pxtextmining.factories.factory_predict_unlabelled_text import (
    get_labels,
    get_probabilities,
)


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
            modelpath = os.path.join(path, model_name + ".sav")
            pickle.dump(models[i], open(modelpath, "wb"))
        # Write performance metrics file
        txtpath = os.path.join(path, model_name + ".txt")
        with open(txtpath, "w") as file:
            file.write(model_metrics[i])
    print(f"{len(models)} models have been written to {path}")


def write_model_preds(x, y_true, preds_df, labels, path="labels.xlsx", return_df=False):
    """Writes an Excel file to enable easier analysis of model outputs using the test set. Columns of the Excel file are: comment_id, actual_labels, predicted_labels, actual_label_probs, and predicted_label_probs.

    Currently only works with sklearn models.

    Args:
    """
    assert len(x) == len(y_true) == len(preds_df)
    actual_labels = pd.DataFrame(y_true, columns=labels).apply(
        get_labels, args=(labels,), axis=1
    )
    actual_labels.name = "actual_labels"
    predicted_labels = preds_df["labels"]
    predicted_labels.name = "predicted_labels"
    df = x.reset_index()
    probabilities = np.array(preds_df.filter(like="Probability", axis=1))
    probs_actual = get_probabilities(actual_labels, labels, probabilities)
    probs_predicted = get_probabilities(predicted_labels, labels, probabilities)
    df = df.merge(actual_labels, left_index=True, right_index=True)
    df = df.merge(predicted_labels, left_on="Comment ID", right_index=True)
    df = df.merge(probs_actual, left_index=True, right_index=True)
    df = df.merge(probs_predicted, left_on="Comment ID", right_index=True)
    # Deal with any rogue characters
    df.applymap(
        lambda x: x.encode("unicode_escape").decode("utf-8")
        if isinstance(x, str)
        else x
    )
    df.to_excel(path, index=False)
    if return_df is True:
        return df
    print(f"Successfully completed, written to {path}")


def write_model_analysis(
    model_name,
    labels,
    dataset,
    path,
    preds_df=None,
    y_true=None,
    custom_threshold_dict=None,
):
    """Writes an Excel file with the performance metrics of each label, as well as the counts of samples for each label.

    Args:
        model_name (str): Model name used in the performance metrics file
        labels (list): List of labels for the categories to be predicted.
        dataset (pd.DataFrame): Original dataset before train test split
        path (str): Filepath where model and performance metrics file are saved.
    """
    metrics_df = parse_metrics_file(f"{path}/{model_name}.txt", labels=labels)
    label_counts = pd.DataFrame(dataset[labels].sum())
    label_counts = label_counts.reset_index()
    label_counts = label_counts.rename(
        columns={"index": "label", 0: "label_count_in_full_dataset"}
    )
    metrics_df = metrics_df.merge(label_counts, on="label").set_index("label")
    if preds_df is not None and y_true is not None:
        more_metrics = additional_analysis(
            preds_df, y_true, labels, custom_threshold_dict
        )
        metrics_df = pd.concat([metrics_df, more_metrics], axis=1)
    metrics_df.to_excel(f"{path}/{model_name}_perf.xlsx", index=True)
