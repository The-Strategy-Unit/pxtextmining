"""
================================================================================
Classification of patient feedback using algoritmhs that can efficiently
handle sparse matrices
================================================================================

Classify documents by label using a bag-of-words approach.

See example in https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html?highlight=text%20classification%20sparse

"""


import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

RandomForestClassifier().get_params()
MultinomialNB().get_params()


##############################################################################
# Load data from the training set
# ------------------------------------
filename = "text_data.csv"
text_data = pd.read_csv(filename)
text_data = text_data.rename(columns={'super': 'target'})
type(text_data)

# split a training set and a test set
X = text_data['improve']  # This way it's a series. Don't do text_data.drop...
y = text_data['target'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )

learners = [RidgeClassifier(),
            LinearSVC(),
            SGDClassifier(),
            Perceptron(),
            PassiveAggressiveClassifier(),
            BernoulliNB(),
            ComplementNB(),
            MultinomialNB(),
            KNeighborsClassifier(),
            NearestCentroid(),
            # RandomForestClassifier()
            ]


#############################################################################
# Benchmark classifiers
# ------------------------------------
# We train and test the datasets with 15 different classification models
# and get performance results for each model.
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('kbest', SelectKBest(chi2, k=10000)),
        ('clf', clf)
    ])
    param_grid = [
        {'vect__ngram_range': [(1, 1), (2, 2)]}
    ]
    gs = GridSearchCV(pipe, param_grid)
    gs.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred_class = gs.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score_accuracy = metrics.accuracy_score(y_test, pred_class)
    print("accuracy:   %0.3f" % score_accuracy)

    score_f1 = metrics.f1_score(y_test, pred_class, average='weighted')
    print("f1:   %0.3f" % score_f1)

    if hasattr(clf, 'predict_proba'):
        pred_prob = gs.predict_proba(X_test)
        score_roc = metrics.roc_auc_score(y_test, pred_prob,
                                          average='weighted',
                                          multi_class='ovr')
    else:
        score_roc = 0

    print("ROC:   %0.3f" % score_roc)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score_accuracy, score_f1, score_roc, train_time, test_time


results = []
for clf in learners:
    results.append(benchmark(clf))

##############################################################################
# Add plots
# ------------------------------------
# Bar plot with performance metrics, training time (normalized) and test time
# (normalized) of each learner.
df = DataFrame(results).transpose()
df.columns = ['Learner', 'Accuracy', 'F1', 'ROC', 'Training time', 'Test time']
df['Training time'] = df['Training time'] / max(df['Training time'])
df['Test time'] = df['Training time'] / max(df['Test time'])

cmap = ListedColormap(['#0343df', '#e50000', '#ffff14', '#929591', '#0343df'])
ax = df.plot.bar(x='Learner', colormap=cmap)
ax.set_xlabel(None)
plt.show()
