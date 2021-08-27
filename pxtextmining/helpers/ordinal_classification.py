import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class OrdinalClassifier(BaseEstimator):

    """
    Estimator class for building an ordinal classification model using the method of
    `Frank and Hall (2001) <https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf>`_ The code in this class is
    based on code published online in `this post
    <https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c>`_.

    **NOTE:** As described later, argument `theme` is for internal use by Nottinghamshire Healthcare NHS Foundation
    Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) labels. It can otherwise be
    safely ignored.

    :param estimator: A Scikit-learn classifier.
    :param dict clfs: Helper variable. Defined inside the class.
    :param y_factorized: Helper variable. Defined inside the class.
    :param unique_class: Helper variable. Defined inside the class.
    :param dict class_dict: Helper variable. Defined inside the class.
    :param str theme: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that use theme labels ("Access", "Environment/ facilities" etc.). The column name of the theme variable.
        Defaults to `None`. If supplied, the theme variable will be used as a predictor (along with the text predictor)
        in the model that is fitted with criticality as the response variable. The rationale is two-fold. First, to
        help the model improve predictions on criticality when the theme labels are readily available. Second, to force
        the criticality for "Couldn't be improved" to always be "3" in the training and test data, as well as in the
        predictions. This is the only criticality value that "Couldn't be improved" can take, so by forcing it to always
        be "3", we are improving model performance, but are also correcting possible erroneous assignments of values
        other than "3" that are attributed to human error.
    :param str target_class_value: The criticality value to assign to "Couldn't be improved".
    :param int theme_class_value: The value of "Couldn't be improved" in the transformed (e.g. one-hot encoded) theme
        column.
    """

    def __init__(self, estimator=LogisticRegression(), clfs={}, y_factorized=None, unique_class=None, class_dict=None,
                 theme=None, target_class_value='3', theme_class_value=1):
        self.estimator = estimator
        self.clfs = clfs
        self.y_factorized = y_factorized
        self.unique_class = unique_class
        self.class_dict = class_dict
        self.theme = theme
        self.target_class_value = target_class_value
        self.theme_class_value = theme_class_value

    def fit(self, X, y=None, **kwargs):
        self.y_factorized = pd.Series(y.astype('int64')).factorize(sort=True)[0]
        self.unique_class = np.sort(np.unique(self.y_factorized))
        self.class_dict = dict(zip(self.y_factorized, y))

        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                y_binary = (self.y_factorized > self.unique_class[i]).astype(np.uint8)
                estimator = clone(self.estimator)
                estimator.fit(X, y_binary)
                self.clfs[i] = estimator
        return self

    def predict_proba_all(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []

        if self.unique_class.shape[0] > 2:
            for i in self.unique_class:
                if i == 0:
                    # V1 = 1 - Pr(y > V1)
                    predicted.append(1 - clfs_predict[i][:, 1])
                elif i in clfs_predict:
                    # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                    predicted.append(clfs_predict[i - 1][:, 1] - clfs_predict[i][:, 1])
                else:
                    # Vk = Pr(y > Vk-1)
                    predicted.append(clfs_predict[i - 1][:, 1])
        return np.vstack(predicted).T

    def predict_proba(self, X):
        return np.max(self.predict_proba_all(X), axis=1)

    def predict(self, X):
        y_pred = np.argmax(self.predict_proba_all(X), axis=1)
        y_pred_orig_class_names = []
        for i in y_pred:
            y_pred_orig_class_names.append(self.class_dict[i])
        re = np.array(y_pred_orig_class_names)

        # This is for internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts that use theme
        # labels ("Access", "Environment/ facilities" etc.). We want the criticality for "Couldn't be improved" to
        # always be "3" (or theme_class_value). The theme label is passed as a one-hot encoded set of columns, of
        # which the first is for "Couldn't be improved". The one-hot encoded columns are actually the first columns of
        # the whole sparse matrix that has the TF-IDFs, sentiment features etc. So we want to find the records
        # with "Couldn't be improved" (i.e. records with a value of 1) in the first, one-hot encoded, column and replace
        # the predicted criticality values with "3".
        if self.theme is not None:
            if isinstance(X[:, 0], np.ndarray):
                theme_col = pd.DataFrame(X[:, 0])
            else:
                theme_col = pd.DataFrame(X[:, 0].todense())
            no_improvements_index = theme_col.loc[theme_col.iloc[:, 0] == self.theme_class_value].index
            re = pd.DataFrame(re, columns=['aux'], index=theme_col.index)
            re.loc[no_improvements_index] = self.target_class_value
            re = np.array(re.aux)
        return re

    def score(self, X, y):
        return self.estimator.score(X, y)

