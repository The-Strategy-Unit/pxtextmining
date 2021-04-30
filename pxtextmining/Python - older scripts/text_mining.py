"""
================================================================================
Classification of patient feedback using algoritmhs that can efficiently
handle sparse matrices
================================================================================

Classify documents by label using a bag-of-words approach.

See example in https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html?highlight=text%20classification%20sparse

"""

from copy import deepcopy
import joblib
from textblob import TextBlob
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

##############################################################################
# Load data from the training set
# ------------------------------------
filename = "text_data_4444.csv"
text_data = pd.read_csv(filename)
text_data = text_data.rename(columns={'super': 'target'})
type(text_data)

#############################################################################
# Calculate polarity and subjectivity of feedback and add to data
# ------------------------------------
# Polarity and subjectivity will hopefully add an extra piece of
# useful information in the feature set for the learners to learn from.
text_polarity = []
text_subjectivity = []
for i in range(len(text_data)):
    tb = TextBlob(text_data['improve'][i])
    text_polarity.append(tb.sentiment.polarity)
    text_subjectivity.append(tb.sentiment.subjectivity)

text_data['comment_polarity'] = text_polarity
text_data['comment_subjectivity'] = text_subjectivity

# It may however not work, because TextBlob seems to fail to capture the 
# positive polarity of comments for label "Couldn't be improved".
text_data[text_data['target'] == "Couldn't be improved"]['comment_polarity'].hist()

# Force polarity to always be 1 for "Couldn't be improved"
text_data.loc[text_data['target'] == "Couldn't be improved", 'comment_polarity'] = 1
text_data[text_data['target'] == "Couldn't be improved"]['comment_polarity'].hist()

#############################################################################
# Split a training set and a test set
# ------------------------------------
#X = text_data['improve']  # This way it's a series. Don't do text_data.drop(['target'], axis=1) as TfidfVectorizer() doesn't like it
X = text_data[['improve', 'comment_polarity', 'comment_subjectivity']]
y = text_data['target'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )

#############################################################################
# Define learners that can handle sparse matrices
# ------------------------------------
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
            RandomForestClassifier()
            ]

#learners = [SGDClassifier(), MultinomialNB()]

#############################################################################
# NLTK-based function for lemmatizing. Will be passed in CountVectorizer()
# ------------------------------------
# https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=stemming
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#############################################################################
# Function for automating pipeline for > 1 learners
# ------------------------------------
# https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python
class ClfSwitcher(BaseEstimator):
    def __init__(
                 self,
                 estimator=SGDClassifier(),
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


#############################################################################
# Prepare pipeline
# ------------------------------------
# Preprocess numerical and text data in different ways.
# Numerical data are min-max normalized because X^2 and Multinomial Naive 
# Bayes can't handle negative data. NOTE: FIND OUT IF IT IS METHODOLOGICALLY 
# SOUND TO DO THIS!
# Pipeline for numeric features
numeric_features = ['comment_polarity', 'comment_subjectivity']
numeric_transformer = Pipeline(steps=[
    ('minmax', MinMaxScaler())])

# Pipeline for text features
text_features = 'improve' # Needs to be a scalar, otherwise TfidfVectorizer() throws an error
text_transformer = Pipeline(steps=[
    ('tfidf', (TfidfVectorizer(max_df=0.95)))]) # https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/

# Pass both pipelines/preprocessors to a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('text', text_transformer, text_features)])

# Pipeline with preprocessors, any other operations and a learner
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('kbest', SelectKBest(chi2)),
                      #('rfecv', RFECV(DecisionTreeClassifier(), cv=5)),
                      ('clf', ClfSwitcher())])

# Parameter grid
param_grid_preproc = {
    'clf__estimator': None,
    'preprocessor__text__tfidf__ngram_range': ((1, 1), (2, 2), (1, 3)),
    'preprocessor__text__tfidf__tokenizer': [LemmaTokenizer(), None],
    'preprocessor__text__tfidf__use_idf': [True, False],
    'kbest__k': (5000, 10000),
    #'kbest__k': (np.array([15, 25, 50]) * X_train.shape[0] / 100).astype(int),
    #'rfecv__estimator': [DecisionTreeClassifier()],
    #'rfecv__step': (0.1, 0.25, 0.5) # Has a scoring argument too. Investigate
}

param_grid = []
for i in learners:
    aux = param_grid_preproc.copy()
    aux['clf__estimator'] = [i]
    if i.__class__.__name__ == LinearSVC().__class__.__name__:
        aux['clf__estimator__class_weight'] = [None, 'balanced']
        aux['clf__estimator__dual'] = [True, False]
    if i.__class__.__name__ == BernoulliNB().__class__.__name__:
        aux['clf__estimator__alpha'] = (0, 0.1, 0.5, 1)
    if i.__class__.__name__ == ComplementNB().__class__.__name__:
        aux['clf__estimator__alpha'] = (0, 0.1, 0.5, 1)
    if i.__class__.__name__ == MultinomialNB().__class__.__name__:
        aux['clf__estimator__alpha'] = (0, 0.1, 0.5, 1)
    if i.__class__.__name__ == SGDClassifier().__class__.__name__:
        aux['clf__estimator__penalty'] = ('l2', 'elasticnet')
    if i.__class__.__name__ == RandomForestClassifier().__class__.__name__:
        aux['clf__estimator__max_features'] = ('sqrt', 0.666)
    param_grid.append(aux)

#############################################################################
# Benchmark classifiers
# ------------------------------------
# Set a scoring measure (other than accuracy) and train several 
# classification models.

# Set Matthews Correlation Coefficient as the scoring measure. Looks promising
# https://towardsdatascience.com/the-best-classification-metric-youve-never-heard-of-the-matthews-correlation-coefficient-3bf50a2f3e9a
# https://towardsdatascience.com/matthews-correlation-coefficient-when-to-use-it-and-when-to-avoid-it-310b3c923f7e
# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0177678&type=printable
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)

# Grid search with 5-fold cross-validation
gscv = GridSearchCV(pipe, param_grid, n_jobs=-1, return_train_score=False, 
                    verbose=3, scoring=matthews_corrcoef_scorer)
gscv.fit(X_train, y_train)
gscv.best_estimator_
gscv.best_params_
gscv.best_score_

#gscv = joblib.load('finalized_model_4444.sav')
estimator_position = len(gscv.best_estimator_) - 1
best_estimator = gscv.best_estimator_[estimator_position].estimator
gscv.best_estimator_.steps.pop(estimator_position)
gscv.best_estimator_.steps.append(('clf', best_estimator))
gscv.best_estimator_.fit(X_train, y_train)
gscv.best_estimator_.score(X_test, y_test)
pred = gscv.best_estimator_.predict(X_test)
metrics.matthews_corrcoef(y_test, pred)
cm = metrics.confusion_matrix(y_test, pred)
DataFrame(cm)
metrics.plot_confusion_matrix(gscv.best_estimator_, X_test, y_test)
#ConfusionMatrixDisplay(cm).plot()

# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# save the model to disk
filename = 'finalized_model_4444.sav'
pickle.dump(gscv, open(filename, 'wb'))

# def benchmark(clf):
#     print('_' * 80)
#     print("Training: ")
#     print(clf)
#     t0 = time()
#     pipe = Pipeline([
#         ('vect', CountVectorizer()),
#         ('tfidf', TfidfTransformer()),
#         #  ('kbest', SelectKBest(chi2, k=10000)),
#         ('kbest', SelectKBest(chi2)),
#         ('clf', clf)
#     ])
#     param_grid = {
#         'vect__ngram_range': ((1, 1), (2, 2), (1, 3)),
#         'kbest__k': (5000, 10000)
#     }
#     gs = GridSearchCV(pipe, param_grid)
#     gs.fit(X_train, y_train)
#     train_time = time() - t0
#     print("train time: %0.3fs" % train_time)
#
#     t0 = time()
#     pred_class = gs.predict(X_test)
#     test_time = time() - t0
#     print("test time:  %0.3fs" % test_time)
#
#     score_accuracy = metrics.accuracy_score(y_test, pred_class)
#     print("accuracy:   %0.3f" % score_accuracy)
#
#     score_f1 = metrics.f1_score(y_test, pred_class, average='weighted')
#     print("f1:   %0.3f" % score_f1)
#
#     if hasattr(clf, 'predict_proba'):
#         pred_prob = gs.predict_proba(X_test)
#         score_roc = metrics.roc_auc_score(y_test, pred_prob,
#                                           average='weighted',
#                                           multi_class='ovr')
#     else:
#         score_roc = 0
#
#     print("ROC:   %0.3f" % score_roc)
#
#     print()
#     clf_descr = str(clf).split('(')[0]
#     return clf_descr, score_accuracy, score_f1, score_roc, train_time, test_time
#
#
# results = []
# for clf in learners:
#     results.append(benchmark(clf))
#
# ##############################################################################
# # Add plots
# # ------------------------------------
# # Bar plot with performance metrics, training time (normalized) and test time
# # (normalized) of each learner.
# df = DataFrame(results)
# df.columns = ['Learner', 'Accuracy', 'F1', 'ROC', 'Training time', 'Test time']
# df['Training time'] = df['Training time'] / max(df['Training time'])
# df['Test time'] = df['Training time'] / max(df['Test time'])
#
# cmap = ListedColormap(['#0343df', '#e50000', '#ffff14', '#929591', '#0343df'])
# ax = df.plot.bar(x='Learner', colormap=cmap)
# ax.set_xlabel(None)
# plt.show()
