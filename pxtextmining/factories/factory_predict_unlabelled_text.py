import numpy as np
import pandas as pd

from pxtextmining.factories.factory_data_load_and_split import (
    bert_data_to_dataset,
    remove_punc_and_nums,
)
from pxtextmining.params import major_cats

def process_text(text):
    """ Enacts same text preprocessing as is found in factory_data_load_and_split when creating training data. Converts to string, removes trailing whitespaces, null values, punctuation and numbers. Converts to lowercase.

    Args:
        text (pd.Series): Series containing data to be cleaned.

    Returns:
        (pd.Series): Processed text data
    """
    text_as_str = text.astype(str)
    text_stripped = text_as_str.str.strip()
    text_no_whitespace = text_stripped.replace([r"^\s*$", r"(?i)^nan$", r"(?i)^null$", r"(?i)^n\/a$"],
                                             np.nan, regex=True)
    text_no_nans = text_no_whitespace.dropna()
    text_cleaned = text_no_nans.astype(str).apply(remove_punc_and_nums)
    processed_text = text_cleaned.replace(r"^\s*$", np.nan, regex=True).dropna()
    return processed_text


def predict_multilabel_sklearn(
    data,
    model,
    labels=major_cats,
    additional_features = False,
    label_fix = True
):
    """Conducts basic preprocessing to remove punctuation and numbers.
    Utilises a pretrained sklearn machine learning model to make multilabel predictions on the cleaned text.
    Also takes the class with the highest predicted probability as the predicted class in cases where no class has
    been predicted, if fix_no_labels = True.

    Args:
        text (pd.Series OR pd.DataFrame): DataFrame or Series containing data to be processed and utilised for predictions. Must be DataFrame with columns 'FFT answer' and 'FFT_q_standardised' if additional_features = True
        model (sklearn.base): Trained sklearn estimator able to perform multilabel classification.
        labels (list, optional): List containing target labels. Defaults to major_cats.
        additional_features (bool, optional): Whether or not FFT_q_standardised is included in data. Defaults to False.
        label_fix (bool, optional): Whether or not the class with the highest probability is taken as the predicted class in cases where no classes are predicted. Defaults to True.

    Returns:
        (pd.DataFrame): DataFrame containing one hot encoded predictions, and a column with a list of the predicted labels.
    """
    if additional_features == False:
        text = pd.Series(data)
    else:
        text = data['FFT answer']
    processed_text = process_text(text)
    if additional_features == False:
        final_data = processed_text
    else:
        final_data = pd.merge(processed_text, data['FFT_q_standardised'], how='left', on='Comment ID')
    binary_preds = model.predict(final_data)
    if label_fix == True:
        pred_probs = np.array(model.predict_proba(final_data))
        predictions = fix_no_labels(binary_preds, pred_probs, model_type="sklearn")
    else:
        predictions = binary_preds
    preds_df = pd.DataFrame(predictions, index=processed_text.index, columns=labels)
    preds_df["labels"] = preds_df.apply(get_labels, args=(labels,), axis=1)
    return preds_df

def predict_with_probs(x, labels, model):
    # Only works with SKLEARN models for now
    # Get all probs for a given comment in one dict first
    pred_probs = model.predict_proba(x)
    probabilities = []
    for i in range(x.shape[0]):
        label_probs = {}
        for index, l in enumerate(labels):
            prob_of_label = pred_probs[index, i, 1]
            label_probs[l] = round(prob_of_label, 5)
        probabilities.append(label_probs)
    probability_s = pd.Series(probabilities)
    probability_s.index = x.index
    # Parse dict of probabilities into one hot encoded format
    prob_preds = []
    for d in range(len(probability_s)):
        row_preds = [0] * 53
        for k,v in probability_s.iloc[d].items():
            max_val = 0
            if v > max_val:
                max_k = k
            if v > 0.5:
                index_over_5 = labels.index(k)
                row_preds[index_over_5] = 1
        if sum(row_preds) == 0:
            index_max = labels.index(max_k)
            row_preds[index_max] = 1
        prob_preds.append(row_preds)
    np.array(prob_preds).shape
    y_pred = np.array(prob_preds)
    return y_pred

def get_probabilities(label_series, labels, predicted_probabilities, model_type):
    """Given a pd.Series containing labels, the list of labels, and a model's outputted predicted_probabilities for each label,
    create a dictionary containing the label and the predicted probability of that label.

    Args:
        label_series (pd.Series): Series containing labels in the format `['label_one', 'label_two']`
        labels (list): List of the label names
        predicted_probabilities (np.array): Predicted probabilities for each label
        model_type (str): Model architecture, if sklearn or otherwise.

    Returns:
        (pd.Series): Series, each line containing a dict with the predicted probabilities for each label.
    """
    probabilities = []
    for i in range(label_series.shape[0]):
        label_probs = {}
        predicted_labels = label_series.iloc[i]
        for each in predicted_labels:
            index_label = labels.index(each)
            if model_type == 'sklearn':
                prob_of_label = predicted_probabilities[index_label, i, 1]
            elif model_type in ('tf', 'bert'):
                prob_of_label = predicted_probabilities[i][index_label]
            label_probs[each] = round(prob_of_label, 5)
        probabilities.append(label_probs)
    probability_s = pd.Series(probabilities)
    probability_s.index = label_series.index
    probability_s.name = f'{label_series.name}_probabilities'
    return probability_s

def get_labels(row, labels):
    """Given a one-hot encoded row of predictions from a dataframe,
    returns a list containing the actual predicted labels as a `str`.

    Args:
        row (pd.DataFrame): Row in a DataFrame containing one-hot encoded predicted labels.
        labels (list): List containing all the target labels, which should also be columns in the dataframe.

    Returns:
        (list): List of the labels that have been predicted for the given text.
    """
    label_list = []
    for c in labels:
        if row[c] == 1:
            label_list.append(c)
    return label_list


def predict_with_bert(
    data, model, max_length=150, additional_features=False, already_encoded=False
):
    """Makes predictions using a transformer-based model. Can encode the data if not already encoded.

    Args:
        data (pd.DataFrame): DataFrame containing features to be passed through model.
        model (tf.keras.models.Model): Pretrained transformer based model in tensorflow keras.
        max_length (int, optional): If encoding is required, maximum length of input text. Defaults to 150.
        additional_features (bool, optional): Whether or not additional features (e.g. question type) are included. Defaults to False.
        already_encoded (bool, optional): Whether or not the input data needs to be encoded. Defaults to False.

    Returns:
        (np.array): Predicted probabilities for each label.
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
    """Function that takes in the binary predicted labels for a particular input, and the predicted probabilities for
    all the labels classes. Where no labels have been predicted for a particular input, takes the label with the highest predicted probability
    as the predicted label.

    Args:
        binary_preds (np.array): Predicted labels, in a one-hot encoded binary format. Some rows may not have any predicted labels.
        predicted_probs (np.array): Predicted probability of each label.
        model_type (str, optional): Whether the model is a sklearn or tensorflow keras model; options are 'tf', 'bert', or 'sklearn. Defaults to "sklearn".

    Returns:
        (np.array): Predicted labels in one-hot encoded format, with all rows containing at least one predicted label.
    """
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
    """Takes predicted probabilities (floats between 0 and 1) and converts these to binary outcomes.
    Scope to finetune this in later iterations of the project depending on the label and whether recall/precision
    is prioritised for that label.

    Args:
        predicted_probs (np.array): Array containing the predicted probabilities for each class. Shape of array corresponds to the number of inputs and the number of target classes; if shape is (100, 13) then there are 100 datapoints, and 13 target classes. Predicted probabilities should range from 0 to 1.

    Returns:
        (np.array): Array containing binary outcomes for each label. Shape should remain the same as input, but values will be either 0 or 1.
    """
    preds = np.where(predicted_probs > 0.5, 1, 0)
    return preds
