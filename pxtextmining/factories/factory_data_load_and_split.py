import pandas as pd
from os import path
import mysql.connector
from sklearn.model_selection import train_test_split
import re
import string
import numpy as np
from pxtextmining.helpers import decode_emojis, text_length, sentiment_scores

def load_data(filename, target, predictor, theme = None):
    """
    This function loads the data from a csv, dataframe, or SQL database. It returns a pd.DataFrame with the data
    required for training a machine learning model.

    :param filename: A ``pandas.DataFrame`` with the data (class and text columns), otherwise the
            dataset name (CSV), including full path to the data folder (if not in the project's working directory), and the
            data type suffix (".csv"). If ``filename`` is ``None``, the data is read from the SQL database.
            **NOTE:** The feature that reads data from the database is for internal use only. Experienced users who would
            like to pull their data from their own databases can, of course, achieve that by slightly modifying the
            relevant lines in the script and setting up the connection to the SQL server.
    :type filename: pd.DataFrame
    :param target: Name of the column containing the target to be predicted.
    :type target: str
    :param str predictor: Name of the column containing the text to be used to train the model or make predictions.
    :param theme: Name of the column containing the 'theme' data which can be used to train a model predicting
            'criticality'
    :type theme: str, optional

    :return: a pandas.DataFrame with the columns named in a way that works for the rest of the pipeline
    :rtype: pandas.DataFrame

    """
    print('Loading dataset...')
    # Read CSV if filename provided
    if filename is not None:
        if isinstance(filename, str):
            text_data = pd.read_csv(filename, encoding='utf-8')
        else:
            text_data = filename
    # Else load from mysql database
    # For this to work set my.conf settings to access mysql database
    else:
        db = mysql.connector.connect(option_files="my.conf", use_pure=True)
        if theme is None:
            with db.cursor() as cursor:
                query = f"SELECT id , {target} , {predictor} FROM text_data"
                cursor.execute(query)
                text_data = cursor.fetchall()
                text_data = pd.DataFrame(text_data)
                text_data.columns = cursor.column_names
                text_data = text_data.set_index('id')
        else:
            with db.cursor() as cursor:
                query = f"SELECT id , {target} , {predictor} , {theme} FROM text_data"
                cursor.execute(query)
                text_data = cursor.fetchall()
                text_data = pd.DataFrame(text_data)
                text_data.columns = cursor.column_names
                text_data = text_data.set_index('id')
    text_data = text_data.rename(columns={target: 'target', predictor: 'predictor'})
    if theme is not None:
        text_data = text_data.rename(columns={theme: 'theme'})
        text_data = text_data[['target', 'predictor', 'theme']]
    else:
        text_data = text_data[['target', 'predictor']]
    print(f'Shape of dataset before cleaning is {text_data.shape}')
    return text_data

def remove_punc_and_nums(text):
    """
    This function removes excess punctuation and numbers from the text. Exclamation marks and apostrophes have been
    left in, as have words in allcaps, as these may denote strong sentiment. Returns a string.

    :param str text: Text to be cleaned

    :return: the cleaned text as a str
    :rtype: str
    """
    text = re.sub('\\n', ' ', text)
    text = re.sub('\\r', ' ', text)
    text = ''.join(char for char in text if not char.isdigit())
    punc_list = string.punctuation.replace('!', '')
    punc_list = punc_list.replace("'", '')
    for punctuation in punc_list:
        text = text.replace(punctuation, ' ')
    text = decode_emojis.decode_emojis(text)
    text_split = [word for word in text.split(' ') if word != '']
    text_lower = []
    for word in text_split:
        if word.isupper():
            text_lower.append(word)
        else:
            text_lower.append(word.lower())
    cleaned_sentence = ' '.join(word for word in text_lower)
    cleaned_sentence = cleaned_sentence.strip()
    return cleaned_sentence

def clean_data(text_data, target = False):
    """
    Function to clean and preprocess data, for training a model or for making predictions using a trained model.
    target = True if processing labelled data for training a model. The DataFrame should contain a column named
    'predictor' containing the text to be processed. If processing dataset with no target, i.e. to make predictions
    using unlabelled data, then target = False. This function also drops NaNs.

    :param pd.DataFrame text_data: A ``pandas.DataFrame`` with the data to be cleaned. Essential to have one
    column labelled 'predictor', containing text for training or predictions.
    :param target: A string. If present, then it denotes that the dataset is for training a model and the y 'target'
    column is present in the dataframe. If set to False, then the function is able to clean text data in the 'predictor'
    column for making new predictions using a trained model.
    :type target: str, optional

    :return: a pandas.DataFrame with the 'predictor' column cleaned
    :rtype: pandas.DataFrame
    """
    if target == True:
        text_data_clean = text_data.dropna(subset=['target', 'predictor']).copy()
    else:
        text_data_clean = text_data.dropna(subset=['predictor']).copy()
    for i in ['NULL', 'N/A', 'NA', 'NONE']:
        text_data_clean = text_data_clean[text_data_clean['predictor'].str.upper() != i].copy()
    text_data_clean['original_text'] = text_data_clean['predictor'].copy()
    text_data_clean['predictor'] = text_data_clean['predictor'].apply(remove_punc_and_nums)
    text_data_clean['predictor'] = text_data_clean['predictor'].replace('', np.NaN)
    if target == True:
        text_data_clean = text_data_clean.dropna(subset=['target', 'predictor']).copy()
    else:
        text_data_clean = text_data_clean.dropna(subset=['predictor']).copy()
    # have decided against dropping duplicates for now as this is a natural part of dataset
    # text_data = text_data.drop_duplicates().copy()
    return text_data_clean

def reduce_crit(text_data, theme):
    """
    'Criticality' is an indication of how strongly negative or positive a comment is. A comment with a criticality
    value of '-5' is very strongly critical of the organisation. A comment with a criticality value of '3' is mildly
    positive about the organisation. 'Criticality' labels are specific to data collected by Nottinghamshire
    Healthcare NHS Foundation Trust.
    This function manipulates the criticality levels to account for an imbalanced dataset. There are not enough samples
    belonging to classes '-5' and '5' so these are set to '-4' and '4'. This function also sets the 'criticality'
    value for all comments tagged as 'Couldn't be improved' to '3'.

    :param pd.DataFrame text_data: A ``pandas.DataFrame`` with the data to be cleaned. Essential to have one
        column labelled 'predictor', containing text for training or predictions.
    :param theme: Name of the column containing the 'theme' data which can be used to train a model predicting
        'criticality'
    :type theme: str, optional

    :return: a pandas.DataFrame with the ordinal values in the 'target' column changed from 5 to 4, or -5 to -4.
    :rtype: pandas.DataFrame

    """
    text_data_crit = text_data.query("target in ('-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5')").copy()
    text_data_crit['target'] = text_data_crit['target'].copy().replace('-5', '-4')
    text_data_crit['target'] = text_data_crit['target'].copy().replace('5', '4')
    if theme is not None:
        text_data_crit.loc[text_data_crit['theme'] == "Couldn't be improved", 'target'] = '3'
    return text_data_crit

def process_data(text_data, target = False):
    """
    Function to clean data and add feature engineering including sentiment scores and text length.
    target = True if processing labelled data for training a model. If processing dataset with no target,
    i.e. to make predictions using unlabelled data, then target = False.

    :param pd.DataFrame, text_data: A ``pandas.DataFrame`` with the data to be cleaned. Essential to have one
    column labelled 'predictor', containing text for training or predictions.
    :param target: Name of the column containing the target to be predicted.
    :type target: str, optional

    :return: a pandas.DataFrame with the ordinal values in the 'target' column changed from 5 to 4, or -5 to -4.
    :rtype: pandas.DataFrame


    """
    # Add feature text_length
    text_data['text_length'] = text_data['predictor'].apply(lambda x:
                                len([word for word in str(x).split(' ') if word != '']))
    # Clean data - basic preprocessing, removing punctuation, decode emojis, dropnas
    text_data_cleaned = clean_data(text_data, target)
    # Get sentiment scores
    sentiment = sentiment_scores.sentiment_scores(text_data_cleaned[['original_text']])
    sentiment = sentiment.copy().drop(columns=['vader_neg', 'vader_neu', 'vader_pos'])
    text_data = text_data_cleaned.join(sentiment).drop(columns=['original_text']).copy()
    print(f'Shape of dataset after cleaning and processing is {text_data.shape}')
    return text_data

def factory_data_load_and_split(filename, target, predictor, test_size=0.33, reduce_criticality=False, theme=None):
    """
    This function pulls together all the functions above. It loads the dataset, renames the response and predictor as
    "target" and "predictor" respectively, conducts preprocessing, and splits the dataset into training and test sets.

    **NOTE:** As described later, arguments `reduce_criticality` and `theme` are for internal use by Nottinghamshire
    Healthcare NHS Foundation Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) and
    criticality labels. They can otherwise be safely ignored.

    :param filename: A ``pandas.DataFrame`` with the data (class and text columns), otherwise the
            dataset name (CSV), including full path to the data folder (if not in the project's working directory), and the
            data type suffix (".csv"). If ``filename`` is ``None``, the data is read from the SQL database.
            **NOTE:** The feature that reads data from the database is for internal use only. Experienced users who would
            like to pull their data from their own databases can, of course, achieve that by slightly modifying the
            relevant lines in the script and setting up the connection to the SQL server.
    :type filename: pd.DataFrame
    :param target: Name of the column containing the target to be predicted.
    :type target: str
    :param str predictor: Name of the column containing the text to be used to train the model or make predictions.
    :param test_size: Proportion of data that will form the test dataset.
    :type test_size: float, optional
    :param reduce_criticality: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that hold data on criticality. If `True`, then all records with a criticality of "-5" (respectively, "5") are
        assigned a criticality of "-4" (respectively, "4"). This is to avoid situations where the pipeline breaks due to
        a lack of sufficient data for "-5" and/or "5". This param is only relevant when target = "criticality"
    :type reduce_criticality: bool, optional
    :param theme: Name of the column containing the 'theme' data which can be used to train a model predicting
            'criticality'. If supplied, the theme variable will be used as a predictor (along with the text predictor)
        in the model that is fitted with criticality as the response variable. The rationale is two-fold. First, to
        help the model improve predictions on criticality when the theme labels are readily available. Second, to force
        the criticality for "Couldn't be improved" to always be "3" in the training and test data, as well as in the
        predictions. This is the only criticality value that "Couldn't be improved" can take, so by forcing it to always
        be "3", we are improving model performance, but are also correcting possible erroneous assignments of values
        other than "3" that are attributed to human error.
    :type theme: str, optional

    :return: A tuple containing the following objects in order:
        x_train, a pd.DataFrame containing the training data;
        x_test, a pd.DataFrame containing the test data;
        y_train, a pd.Series containing the targets for the training data;
        y_test, a pd.Series containing the targets for the test data;
        index_training_data, a pd.Series of the indices of the data used in the train set;
        index_test_data, a pd.Series of the indices of the data used in the test set
    :rtype: tuple

    """

    # Get data from CSV if filename provided. Else, load fom SQL server
    text_data = load_data(filename=filename, theme=theme, target=target, predictor=predictor)

    text_data = process_data(text_data, target = True)

    # This is specific to NHS patient feedback data labelled with "criticality" classes
    if reduce_criticality == True:
        text_data = reduce_crit(text_data, theme)

    print('Preparing training and test sets...')
    x = text_data.drop(columns = 'target').copy() # Needs to be an array of a data frame- can't be a pandas Series
    # if theme is not None:
    #     x['theme'] = text_data['theme'].copy()
    y = text_data['target'].to_numpy()
    x_train, x_test, y_train, y_test, index_training_data, index_test_data = \
            train_test_split(x, y, pd.DataFrame(x).index,
                            test_size=test_size,
                            stratify=y,
                            shuffle=True
                            )
    print("Done")

    return x_train, x_test, y_train, y_test, index_training_data, index_test_data


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, index_training_data, index_test_data = \
        factory_data_load_and_split(filename='datasets/text_data.csv', target="criticality", predictor="feedback",
                                 test_size=0.33, reduce_criticality=True,
                                 theme="label")
    print(x_train.columns)
