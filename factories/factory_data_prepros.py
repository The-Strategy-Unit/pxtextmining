import re
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split


def factory_data_prepros(filename, target, predictor, test_size=0.33):

    print('Loading dataset...')

    data_path = path.join('datasets', filename)
    text_data = pd.read_csv(data_path)
    text_data = text_data.rename(columns={target: 'target', predictor: 'predictor'})

    # predictor_raw = text_data.predictor.copy()  # Keep a copy of the original text
    print('Stripping punctuation from text...')
    text_data['predictor'] = text_data['predictor'].str.replace('[^\w\s]', '')
    print('Stripping excess spaces, whitespaces and line breaks from text...')
    for text, index in zip(text_data['predictor'], text_data.index):
        aux = re.sub(' +', ' ', text)
        aux = " ".join(text.splitlines())
        text_data.loc[index, 'predictor'] = aux

    print('Preparing training and test sets...')
    x = pd.DataFrame(text_data['predictor'])
    y = text_data['target'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=42
                                                        )

    return x_train, x_test, y_train, y_test
