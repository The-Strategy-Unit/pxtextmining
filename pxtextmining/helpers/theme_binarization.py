import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ThemeBinarizer(BaseEstimator, TransformerMixin):
    """
    Class for passing through features that require no preprocessing.
    """

    def __init__(self, target='target', theme='theme',
                 # target_class_value='3',
                 theme_class_value="Couldn't be improved",
                 set_class_to=1, set_rest_to=0):
        self.target = target
        self.theme = theme
        # self.target_class_value = target_class_value
        self.theme_class_value = theme_class_value
        self.set_class_to = set_class_to
        self.set_rest_to = set_rest_to

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X.loc[X[self.theme] == self.theme_class_value, self.target] = self.target_class_value
        X.loc[X[self.theme] == self.theme_class_value, self.theme] = str(self.set_class_to)
        X.loc[X[self.theme] != str(self.set_class_to), self.theme] = str(self.set_rest_to)
        X[self.theme] = X[self.theme].apply(pd.to_numeric, errors='coerce').copy()
        return X
