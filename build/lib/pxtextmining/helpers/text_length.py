import pandas as pd


def text_length(X):
    """
    Calculate the length of a given text.

    :param X: A dictionary, ``pandas.DataFrame``, tuple or list with the text strings.
        If it is a dictionary (``pandas.DataFrame``), it must have a single key (column).
    :return: A ``pandas.DataFrame`` with the length of each text record. Shape [n_samples, 1].
    """

    X = pd.DataFrame(X).copy().rename(lambda x: 'predictor', axis='columns')
    text_length = []

    for i in X.index:
        text = X.loc[i, 'predictor']
        if text is None or str(text) == 'nan':
            text_length.append(len(''))
        else:
            text_length.append(len(text))

    text_length_df = pd.DataFrame(text_length)
    text_length_df.columns = ['text_length']
    text_length_df.index = X.index

    return text_length_df
