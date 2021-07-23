import numpy as np
import pandas as pd
import scipy
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class OrdinalClassifier(BaseEstimator):

    """
    Estimator class for building an ordinal classification model using the method of
    `Frank and Hall (2001) <https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf>`_ The code in this class is
    based on code published online in `this post
    <https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c>`_.
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

        if self.theme is not None:
            # In what follows, the theme column is set to zero, as we don't want it to be a predictor. The only reason
            # we have carried it over is to force criticality for "Couldn't be improved" to be "3" post-prediction.
            # Removing the theme column completely raises a Scikit-learn error that X has one column less than the
            # expected number. It must be an assertion process in the background that ensures that the data passed into
            # the classifier and the data used for fitting have the same number of features. A hacky way to avoid this
            # is to assign zeros to the theme column and keep it in.
            # Assigning a value to a CSR matrix like below will throw an efficiency warning. See more here
            # https://stackoverflow.com/questions/33091397/sparse-efficiency-warning-while-changing-the-column
            # My knowledge of Scipy is basic at the time of writing, so I will just have to live with it. I do not think
            # that it has much impact or any impact at all in terms of processing times with our 10k records.
            X[:, 0] = 0

            # The above replaces the ifelse statement below. Here's why:
            # I don't think that the data will ever be passed as pd.DataFrame, so it's safe to comment out the
            # "elif isinstance(X, pd.DataFrame):" part below. In fact, I don't think that the data format will ever be
            # anything other than scipy.sparse.csr.csr_matrix, so I can safely comment out the
            # "if isinstance(X, scipy.sparse.csr.csr_matrix):" part too. I'm not simply deleting it, just in case I need
            # to bring it back at some point.

            # if isinstance(X, scipy.sparse.csr.csr_matrix):
            #     X[:, 0] = 0
            #     indx = range(1, X.shape[1])
            #     X = X[:, indx]
            # elif isinstance(X, pd.DataFrame):
            #     X.iloc[:, 0] = 0
            #     X = pd.DataFrame(X.drop([0], axis=1))

        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                y_binary = (self.y_factorized > self.unique_class[i]).astype(np.uint8)
                estimator = clone(self.estimator)
                estimator.fit(X, y_binary)
                self.clfs[i] = estimator
        return self

    def predict_proba_all(self, X):
        if self.theme is not None:
            X[:, 0] = 0

            # if isinstance(X, scipy.sparse.csr.csr_matrix):
            #     X[:, 0] = 0
            #     indx = range(1, X.shape[1])
            #     X = X[:, indx]
            # elif isinstance(X, pd.DataFrame):
            #     X.iloc[:, 0] = 0
            #     X = pd.DataFrame(X.drop([0], axis=1))

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
        if self.theme is not None:
            X[:, 0] = 0

            # if isinstance(X, scipy.sparse.csr.csr_matrix):
            #     X[:, 0] = 0
            #     indx = range(1, X.shape[1])
            #     X = X[:, indx]
            # elif isinstance(X, pd.DataFrame):
            #     X.iloc[:, 0] = 0
            #     X = pd.DataFrame(X.drop([0], axis=1))
        return np.max(self.predict_proba_all(X), axis=1)

    def predict(self, X):
        if self.theme is not None:
            theme_col = pd.DataFrame(X[:, 0].todense())
            X[:, 0] = 0

            # if isinstance(X, scipy.sparse.csr.csr_matrix):
            #     X[:, 0] = 0
            #     indx = range(1, X.shape[1])
            #     X = X[:, indx]
            # elif isinstance(X, pd.DataFrame):
            #     X.iloc[:, 0] = 0
            #     X = pd.DataFrame(X.drop([0], axis=1))

        y_pred = np.argmax(self.predict_proba_all(X), axis=1)
        y_pred_orig_class_names = []
        for i in y_pred:
            y_pred_orig_class_names.append(self.class_dict[i])
        re = np.array(y_pred_orig_class_names)

        if self.theme is not None:
            no_improvements_index = theme_col.loc[theme_col.iloc[:, 0] == self.theme_class_value].index
            re = pd.DataFrame(re, columns=['aux'], index=theme_col.index)
            re.loc[no_improvements_index] = self.target_class_value
            re = np.array(re.aux)
        return re

    def score(self, X, y):
        if self.theme is not None:
            X[:, 0] = 0

            # if isinstance(X, scipy.sparse.csr.csr_matrix):
            #     X[:, 0] = 0
            #     indx = range(1, X.shape[1])
            #     X = X[:, indx]
            # elif isinstance(X, pd.DataFrame):
            #     X.iloc[:, 0] = 0
            #     X = pd.DataFrame(X.drop([0], axis=1))
        return self.estimator.score(X, y)

