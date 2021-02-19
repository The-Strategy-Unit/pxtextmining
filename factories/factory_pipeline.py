from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from helpers.tokenization import LemmaTokenizer
from helpers.oversampling import random_over_sampler_data_generator
from helpers.metrics import class_balance_accuracy_score
from helpers.estimator_switcher import ClfSwitcher


def factory_pipeline(x_train, y_train, tknz,
                     metric="class_balance_accuracy_score",
                     cv=5, n_iter=100, n_jobs=5, verbose=3,
                     learners=[
                         "SGDClassifier",
                         "RidgeClassifier",
                         "Perceptron",
                         "PassiveAggressiveClassifier",
                         "BernoulliNB",
                         "ComplementNB",
                         "MultinomialNB",
                         # "KNeighborsClassifier",
                         # "NearestCentroid",
                         "RandomForestClassifier"
                     ]):
    """
    Function prepares and fits an imblearn.pipeline.Pipeline that consists of the following steps:
        1. Up-sampling of rare classes.
        2. Tokenization and lemmatization of predictor.
        3. Feature selection. Uses sklearn.feature_selection.SelectPercentile with sklearn.feature_selection.chi2.
        4. Fitting and benchmarking of user-supplied Scikit-learn estimators.

    :param x_train: Training data (predictor).
    :param y_train: Training data (response).
    :param str tknz: Tokenizer to use ("spacy" or "wordnet").
    :param str metric: Scorer to use during pipeline tuning ("accuracy_score", "balanced_accuracy_score",
        "matthews_corrcoef", "class_balance_accuracy_score").
    :param int cv: Number of cross-validation folds.
    :param int n_iter: Number of parameter settings that are sampled (see sklearn.model_selection.RandomizedSearchCV).
    :param int n_jobs:Number of jobs to run in parallel (see sklearn.model_selection.RandomizedSearchCV).
    :param int verbose: Controls the verbosity (see sklearn.model_selection.RandomizedSearchCV).
    :param list[str] learners: A list of Sci-kit learner names of the learners to tune.
    :return: A fitted imblearn.pipeline.Pipeline.
    """

    text_features = 'predictor'
    text_transformer = Pipeline(steps=[
        ('tfidf', (TfidfVectorizer(tokenizer=LemmaTokenizer(tknz))))])

    preprocessor = ColumnTransformer(
        transformers=[
            # ('num', numeric_transformer, numeric_features),
            ('text', text_transformer, text_features)])

    oversampler = FunctionSampler(func=random_over_sampler_data_generator,
                                  kw_args={'threshold': 200,
                                           'up_balancing_counts': 300,
                                           'random_state': 0},
                                  validate=False)

    pipe = Pipeline(steps=[('sampling', oversampler),
                           ('preprocessor', preprocessor),
                           # ('rfe', RFE(estimator=LogisticRegression(solver="sag", max_iter=10000), step=0.5)),
                           ('selectperc', SelectPercentile(chi2)),
                           ('clf', ClfSwitcher())])

    param_grid_preproc = {
        'sampling__kw_args': [{'threshold': 100}, {'threshold': 200}],
        'sampling__kw_args': [{'up_balancing_counts': 300}, {'up_balancing_counts': 800}],
        'clf__estimator': None,
        'preprocessor__text__tfidf__ngram_range': ((1, 3), (2, 3), (3, 3)),
        'preprocessor__text__tfidf__max_df': [0.7, 0.95],
        'preprocessor__text__tfidf__min_df': [3, 1],
        'preprocessor__text__tfidf__use_idf': [True, False],
        'selectperc__percentile': [70, 85, 100],
    }

    for i in learners:
        if i in "SGDClassifier":
            learners[learners.index(i)] = SGDClassifier()
        if i in "RidgeClassifier":
            learners[learners.index(i)] = RidgeClassifier()
        if i in "Perceptron":
            learners[learners.index(i)] = Perceptron()
        if i in "PassiveAggressiveClassifier":
            learners[learners.index(i)] = PassiveAggressiveClassifier()
        if i in "BernoulliNB":
            learners[learners.index(i)] = BernoulliNB()
        if i in "ComplementNB":
            learners[learners.index(i)] = ComplementNB()
        if i in "MultinomialNB":
            learners[learners.index(i)] = MultinomialNB()
        if i in "KNeighborsClassifier":
            learners[learners.index(i)] = KNeighborsClassifier()
        if i in "NearestCentroid":
            learners[learners.index(i)] = NearestCentroid()
        if i in "RandomForestClassifier":
            learners[learners.index(i)] = RandomForestClassifier()

    param_grid = []
    for i in learners:
        aux = param_grid_preproc.copy()
        aux['clf__estimator'] = [i]
        aux['preprocessor__text__tfidf__norm'] = ['l2']  # See long comment below
        if i.__class__.__name__ == LinearSVC().__class__.__name__:
            aux['clf__estimator__max_iter'] = [10000]
            aux['clf__estimator__class_weight'] = [None, 'balanced']
            # aux['clf__estimator__dual'] = [True, False] # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
        if i.__class__.__name__ == BernoulliNB().__class__.__name__:
            aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
        if i.__class__.__name__ == ComplementNB().__class__.__name__:
            aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
        if i.__class__.__name__ == MultinomialNB().__class__.__name__:
            aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
        if i.__class__.__name__ == SGDClassifier().__class__.__name__:  # Perhaps try out loss='log' at some point?
            aux['clf__estimator__max_iter'] = [10000]
            aux['clf__estimator__class_weight'] = [None, 'balanced']
            aux['clf__estimator__penalty'] = ('l2', 'elasticnet')
            aux['clf__estimator__loss'] = ['hinge', 'log']
        if i.__class__.__name__ == RidgeClassifier().__class__.__name__:
            aux['clf__estimator__class_weight'] = [None, 'balanced']
            aux['clf__estimator__alpha'] = (0.1, 1.0, 10.0)
        if i.__class__.__name__ == Perceptron().__class__.__name__:
            aux['clf__estimator__class_weight'] = [None, 'balanced']
            aux['clf__estimator__penalty'] = ('l2', 'elasticnet')
        if i.__class__.__name__ == RandomForestClassifier().__class__.__name__:
            aux['clf__estimator__max_features'] = ('sqrt', 0.666)
        param_grid.append(aux)
        aux1 = aux.copy()
        aux1['preprocessor__text__tfidf__use_idf'] = [False]
        aux1['preprocessor__text__tfidf__norm'] = [None]
        param_grid.append(aux1)

    refit = metric.replace('_', ' ').replace(' score', '').title()
    scoring = {'Accuracy': make_scorer(accuracy_score),
               'Balanced Accuracy': make_scorer(balanced_accuracy_score),
               'Matthews Correlation Coefficient': make_scorer(matthews_corrcoef),
               'Class Balance Accuracy': make_scorer(class_balance_accuracy_score)}

    pipe_cv = RandomizedSearchCV(pipe, param_grid, n_jobs=n_jobs, return_train_score=False,
                                 cv=cv, verbose=verbose,
                                 scoring=scoring, refit=refit, n_iter=n_iter)
    pipe_cv.fit(x_train, y_train)

    return pipe_cv
