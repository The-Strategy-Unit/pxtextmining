# https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import spacy
from helpers.text_preprocessor import text_preprocessor
nlp = spacy.load("en_core_web_lg")


class EmbeddingsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_processed = [text_preprocessor(doc) for doc in X]
        return np.concatenate([nlp(doc,
                                   disable=["tagger", "parser", "ner"]).vector.reshape(1, -1) for doc in X_processed])
