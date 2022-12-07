from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class ScalerSwitcher(BaseEstimator, TransformerMixin):
    """
    Class for switching between ``Scikit-learn``
    [scalers and preprocessors](https://scikit-learn.org/stable/modules/preprocessing.html#)
    in the randomized search.
    """

    def __init__(self, scaler=MinMaxScaler()):
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.scaler.fit_transform(X)
