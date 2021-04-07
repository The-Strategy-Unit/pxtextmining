from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class ScalerSwitcher(BaseEstimator, TransformerMixin):
    """
    Class for choosing between Scikit-learn scalers.
    """

    def __init__(
                 self,
                 scaler=MinMaxScaler(),
    ): self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.scaler.fit_transform(X)
