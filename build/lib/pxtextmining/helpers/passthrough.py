import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Passthrough(BaseEstimator, TransformerMixin):
    """
    Class for passing through features that require no preprocessing.
    https://stackoverflow.com/questions/54592115/appending-the-columntransformer-result-to-the-original-data-within-a-pipeline
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Single-column data frames are Pandas series, which Scikit-learn doesn't know how to deal with. Make sure that
        # result is always a data frame.
        X = pd.DataFrame(X)
        return X
