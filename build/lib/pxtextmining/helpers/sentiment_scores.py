import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def sentiment_scores(X):
    """
    Calculate sentiment indicators from `TextBlob <https://textblob.readthedocs.io/en/dev/>`_ (polarity and
    subjectivity) and `vaderSentiment <https://pypi.org/project/vaderSentiment/>`_ (positive, negative and neutral
    sentiments and compound score).

    :param X: A dictionary, ``pandas.DataFrame``, tuple or list with the text strings. If it is a dictionary
        (``pandas.DataFrame``), it must have a single key (column).
    :return: A ``pandas.DataFrame`` with the sentiment scores for each text record. Shape [n_samples, 6].
    """

    vader_analyser = SentimentIntensityAnalyzer()
    X = pd.DataFrame(X).copy().rename(lambda x: 'predictor', axis='columns')
    text_blob_scores = []
    vader_scores = []

    for i in X.index:
        text = X.loc[i, 'predictor']
        if text is None or str(text) == 'nan':
            text = ''
        text_blob_scores.append(TextBlob(text).sentiment)
        vader_scores.append(vader_analyser.polarity_scores(text))

    text_blob_scores_df = pd.DataFrame(text_blob_scores)
    text_blob_scores_df.columns = 'text_blob_' + text_blob_scores_df.columns
    text_blob_scores_df.index = X.index

    vader_scores_df = pd.DataFrame.from_dict(vader_scores)
    vader_scores_df.columns = 'vader_' + vader_scores_df.columns
    vader_scores_df.index = X.index

    all_scores = pd.concat([text_blob_scores_df, vader_scores_df], axis=1, ignore_index=False)
    return all_scores
