import pandas as pd
from os import path
import mysql.connector
from sklearn.model_selection import train_test_split


def factory_data_load_and_split(filename, target, predictor, test_size=0.33):
    """
    Function loads the dataset, renames the response and predictor as "target" and "predictor" respectively,
    and splits the dataset into training and test sets.

    :param str filename: Dataset name (CSV), including full path to the data folder (if not in the project's working
        directory), and the data type suffix (".csv"). If ``filename`` is ``None``, the data are read from the database.
        **NOTE:** The feature that reads data from the database is for internal use only. Experienced users who would
        like to pull their data from their own databases can, of course, achieve that by slightly modifying the
        relevant lines in the script. A "my.conf" file will need to be placed in the root, with five lines, as follows
        (without the ";", "<" and ">"):

        - [connector_python];
        - host = <host_name>;
        - database = <database_name>;
        - user = <username>;
        - password = <password>;
    :param str target: Name of the response variable.
    :param str predictor: Name of the predictor variable.
    :param float test_size: Proportion of data that will form the test dataset.
    :return: A tuple of length 4: predictor-train, predictor-test, target-train and target-test datasets.
    """

    print('Loading dataset...')

    # Choose to read CSV from folder or table directly from database
    if filename is not None:
        text_data = pd.read_csv(filename, encoding='utf-8')
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
    text_data = text_data.loc[text_data.target.notna()].copy()
    text_data['predictor'] = text_data.predictor.fillna('__none__')

    # This is specific to NHS patient feedback data labelled with "criticality" classes
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
                                                        shuffle=True
                                                        )

    return x_train, x_test, y_train, y_test
