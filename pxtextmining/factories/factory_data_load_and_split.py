import re
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.data import Dataset
from transformers import AutoTokenizer

from pxtextmining.params import minor_cats,  dataset, major_cat_dict, merged_minor_cats, q_map

def merge_categories(df, new_cat, cats_to_merge):
    """Merges categories together in a dataset. Assumes all categories are all in the right format, one hot encoded with int values.

    Args:
        df (pd.DataFrame): DataFrame with labelled data.
        new_cat (str): Name for new column of merged data.
        cats_to_merge (list): List containing columns to be merged.

    Returns:
        (pd.DataFrame): DataFrame with new columns
    """
    df[new_cat] = np.NaN
    for cat in cats_to_merge:
        print(f"Number of {cat} labels: {df[cat].sum()}")
        df[new_cat] = df[new_cat].mask(df[cat] == 1, other = 1)
    print(f"Number of new label {new_cat}: {df[new_cat].sum()}")
    df = df.drop(columns = cats_to_merge)
    return df


def bert_data_to_dataset(
    X,
    Y=None,
    max_length=150,
    model_name="distilbert-base-uncased",
    additional_features=False,
):
    """This function converts a dataframe into a format that can be utilised by a transformer model.
    If Y is provided then it returns a TensorFlow dataset for training the model.
    If Y is not provided, then it returns a dict which can be used to make predictions by an already trained model.

    Args:
        X (pd.DataFrame): Data to be converted to text data. Text should be in column 'FFT answer',
            FFT question should be in column 'FFT_q_standardised'.
        Y (pd.DataFrame, optional): One Hot Encoded targets. Defaults to None.
        max_length (int, optional): Maximum length of text to be encoded. Defaults to 150.
        model_name (str, optional): Type of transformer model. Defaults to 'distilbert-base-uncased'.
        additional_features (bool, optional): Whether additional features are to be included, currently this is only question type
            in 'FFT_q_standardised' column. Defaults to False.

    Returns:
        (tf.data.Dataset OR dict): `tf.data.Dataset` if Y is provided, `dict` otherwise.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_encoded = dict(
        tokenizer(
            list(X["FFT answer"]),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="tf",
        )
    )
    data_encoded.pop('attention_mask', None)
    if additional_features == True:
        onehotted = onehot(X, "FFT_q_standardised")
        data_encoded["input_cat"] = onehotted.astype(np.float32)
    if Y is not None:
        data_encoded = Dataset.from_tensor_slices((data_encoded, Y))
    return data_encoded


def load_multilabel_data(filename, target="major_categories"):
    """Function for loading the multilabel dataset, converting it from csv to pd.DataFrame. Conducts some basic preprocessing,
    including standardisation of the question types, calculation of text length, and drops rows with no labels. Depending on
    selected `target`, returned dataframe contains different columns.

    Args:
        filename (str): Path to file containing multilabel data, in csv format
        target (str, optional): Options are 'minor_categories', 'major_categories', or 'sentiment'. Defaults to 'major_categories'.

    Returns:
        (pd.DataFrame): DataFrame containing the columns 'FFT categorical answer', 'FFT question', and 'FFT answer'. Also conducts some
    """
    print("Loading multilabel dataset...")
    raw_data = pd.read_csv(
        filename,
        na_values=" ",
    )
    print(f"Shape of raw data is {raw_data.shape}")
    raw_data.columns = raw_data.columns.str.strip()
    raw_data = raw_data.set_index("Comment ID").copy()
    features = ["FFT categorical answer", "FFT question", "FFT answer"]
    # For now the labels are hardcoded, these are subject to change as framework is in progress
    if target in ["minor_categories", "major_categories"]:
        cols = minor_cats
    if target == 'test':
        cols = merged_minor_cats
    elif target == "sentiment":
        cols = ["Comment sentiment"]
    # Sort out the features first
    features_df = raw_data.loc[:, features].copy()
    features_df = clean_empty_features(features_df)
    # Standardize FFT qs
    features_df.loc[:, "FFT_q_standardised"] = (
        features_df.loc[:, "FFT question"].map(q_map).copy()
    )
    if features_df["FFT_q_standardised"].count() != features_df.shape[0]:
        raise ValueError(f'Check q_map is correct. features_df.shape[0] is {features_df.shape[0]}. \n \
                         features_df["FFT_q_standardised"].count()  is {features_df["FFT_q_standardised"].count()}. \n\n\
                         Questions are: {features_df["FFT question"].value_counts()}')
    features_df.loc[:, "text_length"] = features_df.loc[:, "FFT answer"].apply(
        lambda x: len([word for word in str(x).split(" ") if word != ""])
    )
    # Sort out the targets
    targets_df = raw_data.loc[:, cols].copy()
    targets_df = targets_df.replace("1", 1)
    targets_df = targets_df.fillna(value=0)
    if target == "major_categories":
        for maj, min_list in major_cat_dict.items():
            targets_df = merge_categories(targets_df, maj, min_list)
        cols = list(major_cat_dict.keys())
    targets_df.loc[:, "num_labels"] = targets_df.loc[:, cols].sum(axis=1)
    targets_df = targets_df[targets_df["num_labels"] != 0]
    targets_df = targets_df.fillna(value=0)
    # merge two together
    combined_df = pd.merge(features_df, targets_df, left_index=True, right_index=True)
    combined_df = combined_df.reset_index()
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df.set_index('Comment ID')
    print(f"Shape of cleaned data is {combined_df.shape}")
    return combined_df


def clean_empty_features(text_dataframe):
    """Replaces all empty whitespaces in a dataframe with np.NaN.

    Args:
        text_dataframe (pd.DataFrame): DataFrame containing text data with labels.

    Returns:
        (pd.DataFrame): DataFrame with all empty whitespaces replaced with np.NaN
    """
    clean_dataframe = text_dataframe.replace(r"^\s*$", np.nan, regex=True)
    clean_dataframe = clean_dataframe.dropna()
    return clean_dataframe


def onehot(df, col_to_onehot):
    """Function to one-hot encode specified columns in a dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing data to be one-hot encoded
        col_to_onehot (list): List of column names to be one-hot encoded

    Returns:
        (pd.DataFrame): One-hot encoded data
    """
    encoder = OneHotEncoder(sparse=False)
    col_encoded = encoder.fit_transform(df[[col_to_onehot]])
    return col_encoded


def process_data(
    df, target, preprocess_text=True, additional_features=False
):
    """Utilises remove_punc_and_nums and clean_empty_features functions to clean the text data and
    drop any rows that are only whitespace after cleaning. Also fills one-hot encoded columns with
    0s rather than NaNs so that Y target is not sparse.

    Args:
        df (pd.DataFrame): DataFrame containing text data, any additional features, and targets
        target (list): List of column names of targets
        preprocess_text (bool, optional): Whether or not text is to be processed with remove_punc_and_nums. If utilising
            an sklearn model then should be True. If utilising transformer-based BERT model then should be set to False.
            Defaults to True.
        additional_features (bool, optional): Whether or not 'question type' feature should be included. Defaults to False.

    Returns:
        (tuple): Tuple containing two pd.DataFrames. The first contains the X features (text, with or without question type depending on additional_features), the second contains the one-hot encoded Y targets
    """

    if preprocess_text == True:
        X = df["FFT answer"].astype(str).apply(remove_punc_and_nums)
        X = clean_empty_features(X)
        print(f"After preprocessing, shape of X is {X.shape}")
    if preprocess_text == False:
        X = df["FFT answer"].astype(str)
    if additional_features == True:
        X = pd.merge(X, df[['FFT_q_standardised']], left_index = True, right_index = True)
        X = X.reset_index()
        X = X.drop_duplicates()
        X = X.set_index('Comment ID')
    if target == 'sentiment':
        Y = df['Comment sentiment'].astype(int) - 1
    else:
        Y = df[target].fillna(value=0)
    Y = Y.loc[X.index]
    Y = Y.reset_index()
    Y = Y.drop_duplicates()
    Y = Y.set_index('Comment ID')
    if target == 'sentiment':
        Y = Y['Comment sentiment']
    Y = np.array(Y).astype(int)
    return X, Y


def process_and_split_data(
    df,
    target,
    preprocess_text=True,
    additional_features=False,
    random_state=42,
):
    """Combines the process_multilabel_data and train_test_split functions into one function

    Args:
        df (pd.DataFrame): DataFrame containing text data, any additional features, and targets
        target (list): List of column names of targets
        preprocess_text (bool, optional): Whether or not text is to be processed with remove_punc_and_nums. If utilising
            an sklearn model then should be True. If utilising transformer-based BERT model then should be set to False.
            Defaults to True.
        additional_features (bool, optional): Whether or not 'question type' feature should be included. Defaults to False.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Enables reproducible output across multiple function calls. Defaults to 42.

    Returns:
        (list): List containing train-test split of preprocessed X features and Y targets.
    """
    X, Y = process_data(
        df,
        target,
        preprocess_text=preprocess_text,
        additional_features=additional_features,
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, Y_train, Y_test


def remove_punc_and_nums(text):
    """Function to conduct basic preprocessing of text, removing punctuation and numbers, converting
    all text to lowercase, removing trailing whitespace.

    Args:
        text (str): Str containing the text to be cleaned

    Returns:
        (str): Cleaned text, all lowercased with no punctuation, numbers or trailing whitespace.
    """
    text = re.sub("\\n", " ", text)
    text = re.sub("\\r", " ", text)
    text = re.sub("â€™", "'", text)
    text = "".join(char for char in text if not char.isdigit())
    punc_list = string.punctuation
    for punctuation in punc_list:
        if punctuation in [",", ".", "-"]:
            text = text.replace(punctuation, " ")
        else:
            text = text.replace(punctuation, "")
    text_split = [word for word in text.split(" ") if word != ""]
    text_lower = []
    for word in text_split:
        text_lower.append(word.lower())
    cleaned_sentence = " ".join(word for word in text_lower)
    cleaned_sentence = cleaned_sentence.strip()
    return cleaned_sentence


if __name__ == '__main__':
    df = load_multilabel_data(dataset, target = 'major_categories')
    print(df.shape)
    print(df.head())
    for i in df.columns:
        print(f"{i}: {df[i].dtype}")
