from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import spacy
from pxtextmining.helpers.text_preprocessor import text_preprocessor
nlp = spacy.load("en_core_web_lg") # Don't put this inside the function- loading it in every CV iteration would tremendously slow down the pipeline.


class EmbeddingsTransformer(TransformerMixin, BaseEstimator):
    """
    Class for converting text into `GloVe <https://nlp.stanford.edu/projects/glove/>`_ word vectors with
    `spaCy <https://spacy.io/>`_. Helpful resource `here
    <https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/>`_.
    """

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_processed = [text_preprocessor(doc) for doc in X]
        return np.concatenate([nlp(doc,
                                   disable=["tagger", "parser", "ner"]).vector.reshape(1, -1) for doc in X_processed])
