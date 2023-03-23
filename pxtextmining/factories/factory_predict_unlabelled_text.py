import numpy as np
import pandas as pd

from pxtextmining.factories.factory_data_load_and_split import (
    bert_data_to_dataset,
    remove_punc_and_nums,
)
from pxtextmining.params import major_cats


def predict_multilabel_sklearn(
    text: pd.Series,
    model,
    labels=major_cats,
):
    """Conducts basic preprocessing to remove punctuation and numbers.
    Utilises a pretrained sklearn machine learning model to make multilabel predictions on the cleaned text.
    Also takes the class with the highest predicted probability as the predicted class in cases where no class has
    been predicted. Currently used in API, cannot take additional features. Superceded by use of pipeline.

    Args:
        text (pd.Series): Series containing text data to be processed and utilised for predictions.
        model (sklearn.base): Trained sklearn estimator able to perform multilabel classification.
        labels (list, optional): List containing target labels. Defaults to major_cats.

    Returns:
        pd.DataFrame: DataFrame containing the labels and predictions.
    """
    text_no_whitespace = text.replace(r"^\s*$", np.nan, regex=True)
    text_no_nans = text_no_whitespace.dropna()
    text_cleaned = text_no_nans.astype(str).apply(remove_punc_and_nums)
    binary_preds = model.predict(text_cleaned)
    pred_probs = np.array(model.predict_proba(text_cleaned))
    predictions = fix_no_labels(binary_preds, pred_probs, model_type="sklearn")
    preds_df = pd.DataFrame(predictions, index=text_cleaned.index, columns=labels)
    preds_df["labels"] = preds_df.apply(get_labels, args=(labels,), axis=1)
    return preds_df


def get_labels(row, labels):
    """Given a one-hot encoded row of predictions from a dataframe,
    returns a list containing the actual predicted labels as a `str`.

    Args:
        row (pd.DataFrame): Row in a DataFrame containing one-hot encoded predicted labels.
        labels (list): List containing all the target labels, which should also be columns in the dataframe.

    Returns:
        list: List of the labels that have been predicted for the given text.
    """
    label_list = []
    for c in labels:
        if row[c] == 1:
            label_list.append(c)
    return label_list


def predict_with_bert(
    data, model, max_length=150, additional_features=False, already_encoded=False
):
    """_summary_

    Args:
        data (_type_): _description_
        model (_type_): _description_
        max_length (int, optional): _description_. Defaults to 150.
        additional_features (bool, optional): _description_. Defaults to False.
        already_encoded (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if already_encoded == False:
        encoded_dataset = bert_data_to_dataset(
            data, Y=None, max_length=max_length, additional_features=additional_features
        )
    else:
        encoded_dataset = data
    predictions = model.predict(encoded_dataset)
    return predictions


def fix_no_labels(binary_preds, predicted_probs, model_type="sklearn"):
    for i in range(len(binary_preds)):
        if binary_preds[i].sum() == 0:
            if model_type in ("tf", "bert"):
                # index_max = list(predicted_probs[i]).index(max(predicted_probs[i])
                index_max = np.argmax(predicted_probs[i])
            if model_type == "sklearn":
                index_max = np.argmax(predicted_probs[:, i, 1])
            binary_preds[i][index_max] = 1
    return binary_preds


def turn_probs_into_binary(predicted_probs):
    preds = np.where(predicted_probs > 0.5, 1, 0)
    return preds
