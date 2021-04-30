import pandas as pd


def text_length(X):
    """

    :param X:
    :return:
    """

    X = pd.DataFrame(X).copy()
    text_length = []

    for i in X.index:
        text_length.append(len(X.loc[i, 'predictor']))

    text_length_df = pd.DataFrame(text_length)
    text_length_df.columns = ['text_length']
    text_length_df.index = X.index

    return text_length_df
