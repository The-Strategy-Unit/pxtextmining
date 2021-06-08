import pandas as pd


def text_length(X):
    """

    :param X:
    :return:
    """

    X = pd.DataFrame(X).copy()
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
