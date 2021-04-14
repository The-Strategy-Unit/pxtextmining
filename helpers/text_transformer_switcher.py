from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TextTransformerSwitcher(BaseEstimator, TransformerMixin):
    """
    Class for choosing between Bag-of-Words and embeddings transformers.
    """

    def __init__(self, transformer=TfidfVectorizer()):
        self.transformer = transformer

    def fit(self, X, y=None, **kwargs):
        self.transformer.fit(X)
        return self

    def transform(self, X, y=None, **kwargs):
        return self.transformer.transform(X)
