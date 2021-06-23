from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile


class FeatureSelectionSwitcher(BaseEstimator, TransformerMixin):
    """
    Class for choosing between ``Scikit-learn`` `feature selection tests
    <https://scikit-learn.org/stable/modules/feature_selection.html#>`_ for use with
    `sklearn.feature_selection.SelectPercentile
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html>`_.
    """

    def __init__(self, selector=SelectPercentile()):
        self.selector = selector

    def fit(self, X, y, **kwargs):
        self.selector.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        return self.selector.transform(X)
