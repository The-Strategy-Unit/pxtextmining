import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ThemeBinarizer(BaseEstimator, TransformerMixin):
    """
    Class for binarizing categories.

    Sets a selected category to 1 and the rest to 0.

    **NOTE:** As described later, argument `theme` is for internal use by Nottinghamshire Healthcare NHS Foundation
    Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) labels. It can otherwise be
    safely ignored.

    :param str class_col: The name of the column with the classes to binarize.
    :param target_class: The name (if a string) or value (if numeric) of the class that will be set to `set_class_to`.
    :param int set_class_to: The value to set the `target_class` to. Defaults to 1.
    :param int set_rest_to: The value to set all classes but `target_class` to. Defaults to 0.
    """

    def __init__(self, class_col=None,
                 target_class=None,
                 set_class_to=1, set_rest_to=0):
        self.class_col = class_col
        self.target_class = target_class
        self.set_class_to = set_class_to
        self.set_rest_to = set_rest_to

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[X[self.class_col] == self.target_class, self.class_col] = self.set_class_to # Seems like it's okay to have some rows with numbers and some with strings
        X.loc[X[self.class_col] != self.set_class_to, self.class_col] = self.set_rest_to
        # X[self.class_col] = X[self.class_col].apply(pd.to_numeric, errors='coerce', downcast='integer').copy()
        return X
