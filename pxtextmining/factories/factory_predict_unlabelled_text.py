import pandas as pd
import joblib
from itertools import chain
from pxtextmining.factories.factory_data_load_and_split import process_data, load_data
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizer
from pxtextmining.helpers.metrics import multi_label_accuracy
import numpy as np

def predict_with_bert(text, model, max_length=150):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    padded_encodings = tokenizer.batch_encode_plus(
                            list(text),
                            max_length=max_length,
                            return_token_type_ids=True,
                            return_attention_mask=True,
                            truncation=True,
                            padding='max_length',
                            return_tensors='tf')
    predictions = model.predict(padded_encodings["input_ids"])
    return predictions

def fix_no_labels(binary_preds, predicted_probs, model_type = 'sklearn'):
    for i in range(len(binary_preds)):
        if binary_preds[i].sum() == 0:
            if model_type in ('tf', 'bert'):
                # index_max = list(predicted_probs[i]).index(max(predicted_probs[i])
                index_max = np.argmax(predicted_probs[i])
            if model_type == 'sklearn':
                index_max = np.argmax(predicted_probs[:,i,1])
            binary_preds[i][index_max] = 1
    return binary_preds

def turn_probs_into_binary(predicted_probs):
    preds = np.where(predicted_probs > 0.5, 1, 0)
    return preds

def factory_predict_unlabelled_text(dataset, predictor, pipe_path_or_object,
                                    columns_to_return='all_cols', theme=None):
    """
    Predict unlabelled text data using a fitted model or pipeline.

    **NOTE:** As described later, argument `theme` is for internal use by Nottinghamshire Healthcare NHS Foundation
    Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) labels. It can otherwise be
    safely ignored.

    :param pd.DataFrame dataset: A ``pandas.DataFrame`` (or an objsect that can be converted into such) with the text data to predict
        classes for.
    :param str predictor: The column name in the dataset containing the text to be processed and used for generating
        predictions.
    :param estimator pipe_path_or_object: A fitted model or pipeline.
    :param str_or_list columns_to_return:  Determines which columns to return once predictions are made. Str options are
        'all_cols' which returns all columns (default) or 'preds_only' which returns only the predictions.
        If a specific selection of columns is required please provide the column names as strings inside a list, e.g.
        ['feedback', 'organization']. The 'predictions' column will always be included in the returned dataframe.
    :param str theme: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that use theme labels ("Access", "Environment/ facilities" etc.). The column name of the theme variable.
        Defaults to `None`. If supplied, the theme variable will be used as a predictor (along with the text predictor)
        in the model that is fitted with criticality as the response variable. The rationale is two-fold. First, to
        help the model improve predictions on criticality when the theme labels are readily available. Second, to force
        the criticality for "Couldn't be improved" to always be "3" in the training and test data, as well as in the
        predictions. This is the only criticality value that "Couldn't be improved" can take, so by forcing it to always
        be "3", we are improving model performance, but are also correcting possible erroneous assignments of values
        other than "3" that are attributed to human error.
    :return: A ``pandas.DataFrame`` with the predictions in a column named 'predictions' and any other columns supplied
        in ``columns_to_return``.
    :rtype: pd.DataFrame
    """

    data_unlabelled = pd.DataFrame(dataset)
    print(f"Shape of dataset before cleaning is {data_unlabelled.shape}")
    # Rename predictor column to names pipeline knows and replace NAs with empty string.
    if theme is None:
        data_unlabelled = data_unlabelled.rename(columns={predictor: 'predictor'})
    else:
        data_unlabelled = data_unlabelled.rename(columns={predictor: 'predictor', theme: 'theme'})

    data_unlabelled = process_data(data_unlabelled)

    # Load pipeline (if not already supplied) and make predictions
    if isinstance(pipe_path_or_object, str):
        pipe = joblib.load(pipe_path_or_object)
    else:
        pipe = pipe_path_or_object
    predictions = pipe.predict(data_unlabelled)

    data_unlabelled['predictions'] = predictions

    # Rename back to original variable names
    data_with_predictions = data_unlabelled.rename(columns={'predictor': predictor}).copy()
    if theme is not None:
        data_with_predictions = data_unlabelled.rename(columns={'theme': theme}).copy()

    # Set columns to return in final data frame depending on columns_to_return
    if columns_to_return == 'all_cols':
        columns_to_return = list(data_with_predictions.columns)
    elif columns_to_return == 'preds_only':
        columns_to_return = ['predictions']
    if 'predictions' not in columns_to_return:
        columns_to_return.append('predictions')

    return data_with_predictions[columns_to_return]


if __name__ == '__main__':
    dataset = pd.read_csv('datasets/multilabeldata_2.csv')
    text_to_predict = dataset['FFT answer'][:2000].dropna().to_list()
    predicted_probs = predict_with_bert(text_to_predict, model_path='test_multilabel/bert/distilbert2')
    preds = turn_probs_into_binary(predicted_probs)
