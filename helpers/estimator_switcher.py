#############################################################################
# Function for automating pipeline for > 1 learners
# ------------------------------------
# https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier


class ClfSwitcher(BaseEstimator):
    def __init__(
                 self,
                 estimator=SGDClassifier(max_iter=10000),
    ): self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)