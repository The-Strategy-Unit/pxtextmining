import pandas as pd
from os import path
import mysql.connector
from sklearn.model_selection import train_test_split


def factory_data_load_and_split(filename, target, predictor, test_size=0.33):
    """
    Function loads the dataset, renames the response and predictor as "target" and "predictor" respectively,
    cleans the text in the predictor, splits the dataset into training and test sets.

    :param str filename: Dataset name (CSV), including the data type suffix. If None, data is read from the database.
    :param str target: Name of the response variable.
    :param str predictor: Name of the predictor variable.
    :param float test_size: Proportion of data that will form the test dataset.
    :return: A tuple of length 4: predictor-train, predictor-test, target-train and target-test datasets.
    """

    print('Loading dataset...')

    # Choose to read CSV from folder or table directly from database
    if filename is not None:
        data_path = path.join('datasets', filename)
        text_data = pd.read_csv(data_path, encoding='utf-8')
    else:
        db = mysql.connector.connect(option_files="my.conf", use_pure=True)
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT  " + target + ", " + predictor + " FROM text_data"
            )
            text_data = cursor.fetchall()
            text_data = pd.DataFrame(text_data)
            text_data.columns = cursor.column_names

    text_data = text_data.rename(columns={target: "target", predictor: "predictor"})
    text_data = text_data.loc[text_data.target.notnull()].copy()

    if target == 'criticality':
        text_data = text_data.query("target in ('-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5')")
        text_data.loc[text_data.target == '-5', 'target'] = '-4'
        text_data.loc[text_data.target == '5', 'target'] = '4'

    print('Preparing training and test sets...')
    x = pd.DataFrame(text_data["predictor"])
    y = text_data["target"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        shuffle=True,
                                                        # random_state=42 # https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
                                                        )

    return x_train, x_test, y_train, y_test
