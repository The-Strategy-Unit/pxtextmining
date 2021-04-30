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
filename = "/Users/afsa43/git_projects/positive_about_change_text_mining/text_data.csv"
text_data = pd.read_csv(filename)
text_data = text_data.rename(columns={'super': 'target'})
# text_data = text_data.to_numpy()
type(text_data)

# order of labels in `target_names` can be different from `categories`
target_names = text_data.target  # MONITOR THIS. I took it from the example script but don't know what it does. Should it be defined on the whole dataset or train data only?

# split a training set and a test set
X = text_data['improve']  # This way it's a series. Don't do text_data.drop...
y = text_data['target'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.33, stratify=y, shuffle=True, random_state=42)

# def print_binary_evaluation(y_true, y_pred):
#     results_dict = {'accuracy': accuracy_score(y_true, y_pred),
#                     'f1_score': f1_score(y_true, y_pred)}
#     return results_dict

learners = [#LogisticRegression(),
            RidgeClassifier(),
            LinearSVC(),
            SGDClassifier(),
            Perceptron(),
            PassiveAggressiveClassifier(),
            BernoulliNB(),
            ComplementNB(),
            MultinomialNB(),
            KNeighborsClassifier(),
            NearestCentroid()#,
            #RandomForestClassifier()
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
    pred = gs.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf in learners:
    results.append(benchmark(clf))

##############################################################################
# Add plots
# ------------------------------------
# The bar plot indicates the accuracy, training time (normalized) and test time
# (normalized) of each classifier.
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]
print(DataFrame(results))
DataFrame(results).max(axis=1)

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
