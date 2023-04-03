import re
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.data import Dataset
from transformers import AutoTokenizer


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
    if additional_features == True:
        onehotted = onehot(X, "FFT_q_standardised")
        data_encoded["input_cat"] = onehotted
    if isinstance(Y, pd.DataFrame):
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
        dtype={
            "Gratitude/ good experience": "Int64",
            "Negative experience": "Int64",
            "Not assigned": "Int64",
            "Organisation & efficiency": "Int64",
            "Funding & use of financial resources": "Int64",
            "Non-specific praise for staff": "Int64",
            "Non-specific dissatisfaction with staff": "Int64",
            "Staff manner & personal attributes": "Int64",
            "Number & deployment of staff": "Int64",
            "Staff responsiveness": "Int64",
            "Staff continuity": "Int64",
            "Competence & training": "Int64",
            "Unspecified communication": "Int64",
            "Staff listening, understanding & involving patients": "Int64",
            "Information directly from staff during care": "Int64",
            "Information provision & guidance": "Int64",
            "Being kept informed, clarity & consistency of information": "Int64",
            "Service involvement with family/ carers": "Int64",
            "Patient contact with family/ carers": "Int64",
            "Contacting services": "Int64",
            "Appointment arrangements": "Int64",
            "Appointment method": "Int64",
            "Timeliness of care": "Int64",
            "Supplying medication": "Int64",
            "Understanding medication": "Int64",
            "Pain management": "Int64",
            "Diagnosis & triage": "Int64",
            "Referals & continuity of care": "Int64",
            "Length of stay/ duration of care": "Int64",
            "Discharge": "Int64",
            "Care plans": "Int64",
            "Patient records": "Int64",
            "Impact of treatment/ care - physical health": "Int64",
            "Impact of treatment/ care - mental health": "Int64",
            "Impact of treatment/ care - general": "Int64",
            "Links with non-NHS organisations": "Int64",
            "Cleanliness, tidiness & infection control": "Int64",
            "Noise & restful environment": "Int64",
            "Temperature": "Int64",
            "Lighting": "Int64",
            "Decoration": "Int64",
            "Smell": "Int64",
            "Comfort of environment": "Int64",
            "Atmosphere of ward/ environment": "Int64",
            "Access to outside/ fresh air": "Int64",
            "Privacy": "Int64",
            "Safety & security": "Int64",
            "Provision of medical  equipment": "Int64",
            "Food & drink provision": "Int64",
            "Food preparation facilities for patients & visitors": "Int64",
            "Service location": "Int64",
            "Transport to/ from services": "Int64",
            "Parking": "Int64",
            "Provision & range of activities": "Int64",
            "Electronic entertainment": "Int64",
            "Feeling safe": "Int64",
            "Patient appearance & grooming": "Int64",
            "Mental Health Act": "Int64",
            "Psychological therapy arrangements": "Int64",
            "Existence of services": "Int64",
            "Choice of services": "Int64",
            "Respect for diversity": "Int64",
            "Admission": "Int64",
            "Out of hours support (community services)": "Int64",
            "Learning organisation": "Int64",
            "Collecting patients feedback": "Int64",
        },
        na_values=" ",
    )
    print(f"Shape of raw data is {raw_data.shape}")
    raw_data.columns = raw_data.columns.str.strip()
    raw_data = raw_data.set_index("Comment ID").copy()
    features = ["FFT categorical answer", "FFT question", "FFT answer"]
    # For now the labels are hardcoded, these are subject to change as framework is in progress
    if target in ["minor_categories", "major_categories"]:
        cols = [
            "Gratitude/ good experience",
            "Negative experience",
            "Not assigned",
            "Organisation & efficiency",
            "Funding & use of financial resources",
            "Non-specific praise for staff",
            "Non-specific dissatisfaction with staff",
            "Staff manner & personal attributes",
            "Number & deployment of staff",
            "Staff responsiveness",
            "Staff continuity",
            "Competence & training",
            "Unspecified communication",
            "Staff listening, understanding & involving patients",
            "Information directly from staff during care",
            "Information provision & guidance",
            "Being kept informed, clarity & consistency of information",
            "Service involvement with family/ carers",
            "Patient contact with family/ carers",
            "Contacting services",
            "Appointment arrangements",
            "Appointment method",
            "Timeliness of care",
            "Supplying medication",
            "Understanding medication",
            "Pain management",
            "Diagnosis & triage",
            "Referals & continuity of care",
            "Length of stay/ duration of care",
            "Discharge",
            "Care plans",
            "Patient records",
            "Impact of treatment/ care - physical health",
            "Impact of treatment/ care - mental health",
            "Impact of treatment/ care - general",
            "Links with non-NHS organisations",
            "Cleanliness, tidiness & infection control",
            "Noise & restful environment",
            "Temperature",
            "Lighting",
            "Decoration",
            "Smell",
            "Comfort of environment",
            "Atmosphere of ward/ environment",
            "Access to outside/ fresh air",
            "Privacy",
            "Safety & security",
            "Provision of medical  equipment",
            "Food & drink provision",
            "Food preparation facilities for patients & visitors",
            "Service location",
            "Transport to/ from services",
            "Parking",
            "Provision & range of activities",
            "Electronic entertainment",
            "Feeling safe",
            "Patient appearance & grooming",
            "Mental Health Act",
            "Psychological therapy arrangements",
            "Existence of services",
            "Choice of services",
            "Respect for diversity",
            "Admission",
            "Out of hours support (community services)",
            "Learning organisation",
            "Collecting patients feedback",
        ]
    elif target == "sentiment":
        cols = ["Comment sentiment"]
    # Sort out the features first
    features_df = raw_data.loc[:, features].copy()
    features_df = clean_empty_features(features_df)
    # Standardize FFT qs
    q_map = {
        "Please tell us why": "nonspecific",
        "Please tells us why you gave this answer?": "nonspecific",
        "FFT Why?": "nonspecific",
        "What was good?": "what_good",
        "Is there anything we could have done better?": "could_improve",
        "How could we improve?": "could_improve",
        "What could we do better?": "could_improve",
        "Please describe any things about the 111 service that \nyou were particularly satisfied and/or dissatisfied with": "nonspecific",
    }
    features_df.loc[:, "FFT_q_standardised"] = (
        features_df.loc[:, "FFT question"].map(q_map).copy()
    )
    features_df.loc[:, "text_length"] = features_df.loc[:, "FFT answer"].apply(
        lambda x: len([word for word in str(x).split(" ") if word != ""])
    )
    # Sort out the targets
    targets_df = raw_data.loc[:, cols].copy()
    targets_df = targets_df.replace("1", 1)
    targets_df = targets_df.fillna(value=0)
    if target == "major_categories":
        major_categories = {
            "Gratitude/ good experience": "General",
            "Negative experience": "General",
            "Not assigned": "General",
            "Organisation & efficiency": "General",
            "Funding & use of financial resources": "General",
            "Non-specific praise for staff": "Staff",
            "Non-specific dissatisfaction with staff": "Staff",
            "Staff manner & personal attributes": "Staff",
            "Number & deployment of staff": "Staff",
            "Staff responsiveness": "Staff",
            "Staff continuity": "Staff",
            "Competence & training": "Staff",
            "Unspecified communication": "Communication & involvement",
            "Staff listening, understanding & involving patients": "Communication & involvement",
            "Information directly from staff during care": "Communication & involvement",
            "Information provision & guidance": "Communication & involvement",
            "Being kept informed, clarity & consistency of information": "Communication & involvement",
            "Service involvement with family/ carers": "Communication & involvement",
            "Patient contact with family/ carers": "Communication & involvement",
            "Contacting services": "Access to medical care & support",
            "Appointment arrangements": "Access to medical care & support",
            "Appointment method": "Access to medical care & support",
            "Timeliness of care": "Access to medical care & support",
            "Supplying medication": "Medication",
            "Understanding medication": "Medication",
            "Pain management": "Medication",
            "Diagnosis & triage": "Patient journey & service coordination",
            "Referals & continuity of care": "Patient journey & service coordination",
            "Length of stay/ duration of care": "Patient journey & service coordination",
            "Discharge": "Patient journey & service coordination",
            "Care plans": "Patient journey & service coordination",
            "Patient records": "Patient journey & service coordination",
            "Impact of treatment/ care - physical health": "Patient journey & service coordination",
            "Impact of treatment/ care - mental health": "Patient journey & service coordination",
            "Impact of treatment/ care - general": "Patient journey & service coordination",
            "Links with non-NHS organisations": "Patient journey & service coordination",
            "Cleanliness, tidiness & infection control": "Environment & equipment",
            "Noise & restful environment": "Environment & equipment",
            "Temperature": "Environment & equipment",
            "Lighting": "Environment & equipment",
            "Decoration": "Environment & equipment",
            "Smell": "Environment & equipment",
            "Comfort of environment": "Environment & equipment",
            "Atmosphere of ward/ environment": "Environment & equipment",
            "Access to outside/ fresh air": "Environment & equipment",
            "Privacy": "Environment & equipment",
            "Safety & security": "Environment & equipment",
            "Provision of medical  equipment": "Environment & equipment",
            "Food & drink provision": "Food & diet",
            "Food preparation facilities for patients & visitors": "Food & diet",
            "Service location": "Service location, travel & transport",
            "Transport to/ from services": "Service location, travel & transport",
            "Parking": "Service location, travel & transport",
            "Provision & range of activities": "Activities",
            "Electronic entertainment": "Activities",
            "Feeling safe": "Category TBC",
            "Patient appearance & grooming": "Category TBC",
            "Mental Health Act": "Mental Health specifics",
            "Psychological therapy arrangements": "Mental Health specifics",
            "Existence of services": "Additional",
            "Choice of services": "Additional",
            "Respect for diversity": "Additional",
            "Admission": "Additional",
            "Out of hours support (community services)": "Additional",
            "Learning organisation": "Additional",
            "Collecting patients feedback": "Additional",
        }
        new_df = targets_df.copy().drop(columns=cols)
        for i in targets_df[cols].index:
            for label in cols:
                if targets_df.loc[i, label] == 1:
                    new_cat = major_categories[label]
                    new_df.loc[i, new_cat] = 1
        targets_df = new_df.copy()
        cols = list(set(major_categories.values()))
    targets_df.loc[:, "num_labels"] = targets_df.loc[:, cols].sum(axis=1)
    targets_df = targets_df[targets_df["num_labels"] != 0]
    targets_df = targets_df.fillna(value=0)
    # merge two together
    combined_df = pd.merge(features_df, targets_df, left_index=True, right_index=True)
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


def process_multilabel_data(
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
    Y = df[target].fillna(value=0)
    if preprocess_text == True:
            X = df["FFT answer"].astype(str).apply(remove_punc_and_nums)
            X = clean_empty_features(X)
            print(f"After preprocessing, shape of X is {X.shape}")
    if preprocess_text == False:
            X = df["FFT answer"].astype(str)
    if additional_features == True:
        X = pd.merge(
            X,
            df[["FFT_q_standardised", "text_length"]],
            left_index=True,
            right_index=True,
        )
    Y = Y.loc[X.index]
    return X, Y


def process_and_split_multilabel_data(
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
    X, Y = process_multilabel_data(
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
