from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
# from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from pxtextmining.helpers.text_preprocessor import text_preprocessor
from pxtextmining.helpers.sentiment_scores import sentiment_scores
from pxtextmining.helpers.text_length import text_length
from pxtextmining.helpers.tokenization import LemmaTokenizer
from pxtextmining.helpers.word_vectorization import EmbeddingsTransformer
from pxtextmining.helpers.oversampling import random_over_sampler_data_generator
from pxtextmining.helpers.metrics import class_balance_accuracy_score
from pxtextmining.helpers.estimator_switcher import ClfSwitcher
from pxtextmining.helpers.ordinal_classification import OrdinalClassifier
from pxtextmining.helpers.scaler_switcher import ScalerSwitcher
from pxtextmining.helpers.feature_selection_switcher import FeatureSelectionSwitcher
from pxtextmining.helpers.text_transformer_switcher import TextTransformerSwitcher
from pxtextmining.helpers.theme_binarization import ThemeBinarizer


def factory_pipeline(x, y, tknz="spacy",
                     ordinal=False,
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
                     ],
                     theme=None):

    """
    Prepare and fit a text classification pipeline.

    The pipeline's parameter grid switches between two approaches to text classification: Bag-of-Words and Embeddings.
    For the former, both TF-IDF and raw counts are tried out.

    The pipeline does the following:

    - Feature engineering:

      * Converts text into TF-IDFs or `GloVe <https://nlp.stanford.edu/projects/glove/>`_ word vectors with
        `spaCy <https://spacy.io/>`_;
      * Creates a new feature that is the length of the text in each record;
      * Performs sentiment analysis on the text feature and creates new features that are all scores/indicators
        produced by `TextBlob <https://textblob.readthedocs.io/en/dev/>`_
        and `vaderSentiment <https://pypi.org/project/vaderSentiment/>`_.
      * Applies `sklearn.preprocessing.KBinsDiscretizer
        <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html>`_ to the text
        length and sentiment indicator features, and `sklearn.preprocessing.StandardScaler
        <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_ to the
        embeddings (word vectors);
    - Up-sampling of rare classes: uses `imblearn.over_sampling.RandomOverSampler
      <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler>`_
      to up-sample rare classes. Currently the threshold to consider a class as rare and the up-balancing values are
      fixed and cannot be user-defined.
    - Tokenization and lemmatization of the text feature: uses ``spaCy`` (default) or `NLTK <https://www.nltk.org/>`_.
      It also strips punctuation, excess spaces, and metacharacters "r" and "n" from the text. It converts emojis into
      "__text__" (where "text" is the emoji name), and NA/NULL values into "__notext__" (the pipeline does get rid of
      records with no text, but this conversion at least deals with any escaping ones).
    - Feature selection: Uses `sklearn.feature_selection.SelectPercentile
      <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html>`_
      with `sklearn.feature_selection.chi2
      <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2>`_
      for TF-IDFs or `sklearn.feature_selection.f_classif
      <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn-feature-selection-f-classif>`_
      for embeddings.
    - Fitting and benchmarking of user-supplied ``Scikit-learn`` `estimators
      <https://scikit-learn.org/stable/modules/classes.html>`_.

    The numeric values in the grid are currently lists/tuples of values that are defined either empirically or
    are based on the published literature (e.g. for Random Forest, see `Probst et al. 2019
    <https://arxiv.org/abs/1802.09596>`_). Values may be replaced by appropriate distributions in a future release.

     **NOTE:** As described later, argument `theme` is for internal use by Nottinghamshire Healthcare NHS Foundation
     Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) labels. It can otherwise be
     safely ignored.

    :param bool ordinal: Whether to fit an ordinal classification model. The ordinal model is the implementation of
        `Frank and Hall (2001) <https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf>`_ that can use any
        standard classification model that calculates probabilities.
    :param x: The text feature.
    :param y: The response variable.
    :param str tknz: Tokenizer to use ("spacy" or "wordnet").
    :param str metric: Scorer to use during pipeline tuning ("accuracy_score", "balanced_accuracy_score",
        "matthews_corrcoef", "class_balance_accuracy_score").
    :param int cv: Number of cross-validation folds.
    :param int n_iter: Number of parameter settings that are sampled (see `sklearn.model_selection.RandomizedSearchCV
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_).
    :param int n_jobs: Number of jobs to run in parallel (see ``sklearn.model_selection.RandomizedSearchCV``).
    :param int verbose: Controls the verbosity (see ``sklearn.model_selection.RandomizedSearchCV``).
    :param str, list[str] learners: A list of ``Scikit-learn`` names of the learners to tune. Must be one or more of
        "SGDClassifier", "RidgeClassifier", "Perceptron", "PassiveAggressiveClassifier", "BernoulliNB", "ComplementNB",
        "MultinomialNB", "KNeighborsClassifier", "NearestCentroid", "RandomForestClassifier". When a single model is
        used, it can be passed as a string.
    :param str theme: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that use theme labels ("Access", "Environment/ facilities" etc.). The column name of the theme variable.
        Defaults to `None`. If supplied, the theme variable will be used as a predictor (along with the text predictor)
        in the model that is fitted with criticality as the response variable. The rationale is two-fold. First, to
        help the model improve predictions on criticality when the theme labels are readily available. Second, to force
        the criticality for "Couldn't be improved" to always be "3" in the training and test data, as well as in the
        predictions. This is the only criticality value that "Couldn't be improved" can take, so by forcing it to always
        be "3", we are improving model performance, but are also correcting possible erroneous assignments of values
        other than "3" that are attributed to human error.
    :return: A tuned `sklearn.pipeline.Pipeline
        <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_/
        `imblearn.pipeline.Pipeline
        <https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html#imblearn.pipeline.Pipeline>`_.
    """

    features_text = 'predictor'

    # Define transformers for pipeline #
    # Transformer that calculates text length and transforms it.
    transformer_text_length = Pipeline(steps=[
        ('length', (FunctionTransformer(text_length))),
        ('scaler', (ScalerSwitcher()))
    ])

    # Transformer that calculates sentiment indicators (e.g. TextBlob, VADER) and transforms them.
    transformer_sentiment = Pipeline(steps=[
        ('sentiment', (FunctionTransformer(sentiment_scores))),
        ('scaler', (ScalerSwitcher()))
    ])

    # Transformer that converts text to Bag-of-Words or embeddings.
    transformer_text = Pipeline(steps=[
        ('text', (TextTransformerSwitcher()))
    ])

    # Gather transformers.
    preprocessor = ColumnTransformer(
        transformers=[
            ('sentimenttr', transformer_sentiment, features_text),
            ('lengthtr', transformer_text_length, features_text),
            ('texttr', transformer_text, features_text)])

    # Up-sampling step #
    oversampler = FunctionSampler(func=random_over_sampler_data_generator,
                                  kw_args={'threshold': 200,
                                           'up_balancing_counts': 300,
                                           'random_state': 0},
                                  validate=False)

    # Make pipeline #
    if ordinal and theme is not None:
        # This is for internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts that use theme
        # labels ("Access", "Environment/ facilities" etc.). We want the criticality for "Couldn't be improved" to
        # always be "3". The theme label is passed as a one-hot encoded set of columns (or as a "binarized" column where
        # 1 is for "Couldn't be improved" and 0 is for everything else) of which the first is for # "Couldn't be
        # improved". The one-hot encoded columns (or the binarized column) are (is) actually the first column(s) of the
        # whole sparse matrix that has the TF-IDFs, sentiment features etc. that is produced when fitting by the
        # pipeline. When running the ordinal classification model, we want to find the records with "Couldn't be
        # improved" (i.e. records with a value of 1) in the first column and replace the predicted criticality values
        # with "3".
        # When one-hot encoded, we pass all of the theme's columns into the model, so we handle them separately from
        # text predictor to avoid the feature selection step for them. We thus make a separate pipeline with the
        # preprocessor and feature selection steps for the text predictor (pipe_all_but_theme) and one-hot encode the
        # theme column in all_transforms. We want to place "Couldn't be improved" in position 0 (first column) of the
        # thus produced sparse matrix so as to easily access it in the code for the ordinal model (OrdinalClassifier()).
        pipe_all_but_theme = Pipeline([
            ('preprocessor', preprocessor),
            ('featsel', FeatureSelectionSwitcher())
        ])

        all_transforms = ColumnTransformer([
            ('theme', ScalerSwitcher(), ['theme']), # Try out OneHotEncoder() or ThemeBinarizer().
            ('process', pipe_all_but_theme, [features_text])
        ])

        pipe = Pipeline([
            ('sampling', oversampler),
            ('alltrans', all_transforms),
            ('clf', OrdinalClassifier(theme='theme', target_class_value='3', theme_class_value=1))
        ])
    elif ordinal and theme is None:
        pipe = Pipeline([
            ('sampling', oversampler),
            ('preprocessor', preprocessor),
            ('featsel', FeatureSelectionSwitcher()),
            ('clf', OrdinalClassifier())])
    else:
        pipe = Pipeline([
            ('sampling', oversampler),
            ('preprocessor', preprocessor),
            ('featsel', FeatureSelectionSwitcher()),
            ('clf', ClfSwitcher())])

    # Define (hyper)parameter grid #
    # A few initial value ranges for some (hyper)parameters.
    param_grid_preproc = {
        'sampling__kw_args': [{'threshold': 100}, {'threshold': 200}],
        'sampling__kw_args': [{'up_balancing_counts': 300}, {'up_balancing_counts': 800}],
        'clf__estimator': None,
        'preprocessor__sentimenttr__scaler__scaler': None,
        'preprocessor__lengthtr__scaler__scaler': None,
        'preprocessor__texttr__text__transformer': None,
        'featsel__selector': [SelectPercentile()],
        'featsel__selector__percentile': [70, 85, 100]
    }

    if ordinal and theme is not None:
        param_grid_preproc['alltrans__theme__scaler'] = None


    # If a single model is passed as a string, convert to list
    if isinstance(learners, str):
        learners = [learners]

    # Just in case user has supplied the same learner more than once
    learners = list(set(learners))

    # For Frank and Hall's (2001) ordinal method to work, we need models that can calculate probs/scores.
    if ordinal:
        learners = [lrn for lrn in learners if lrn not in ["RidgeClassifier", "Perceptron",
                                                           "PassiveAggressiveClassifier", "NearestCentroid"]]

    # Replace learner name with learner class in 'learners' function argument.
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

    # Further populate (hyper)parameter grid.
    # NOTE ABOUT PROCESS BELOW:
    # Use TfidfVectorizer() as CountVectorizer() also, to determine if raw
    # counts instead of frequencies improves performance. This requires
    # use_idf=False and norm=None. We want to ensure that norm=None
    # will not be combined with use_idf=True inside the grid search, so we
    # create a separate parameter set to prevent this from happening. We do
    # this below with temp list aux1.
    # Meanwhile, we want norm='l2' (the default) for the grid defined by temp
    # list aux. If we don't explicitly set norm='l2' in aux, the
    # norm column in the table of the CV results (following fitting) is
    # always empty. My speculation is that Scikit-learn does consider norm
    # to be 'l2' for aux, but it doesn't print it. That's because unless we
    # explicitly run aux['preprocessor__text__tfidf__norm'] = ['l2'], setting
    # norm as 'l2' in aux is implicit (i.e. it's the default), while setting
    # norm as None in aux1 is explicit (i.e. done by the user). But we want
    # the colum norm in the CV results to clearly state which runs used the
    # 'l2' norm, hence we explicitly run command
    # aux['preprocessor__text__tfidf__norm'] = ['l2'].

    param_grid = []
    for i in learners:
        for j in [TfidfVectorizer(), EmbeddingsTransformer()]:
            aux = param_grid_preproc.copy()
            aux['clf__estimator'] = [i]
            aux['preprocessor__texttr__text__transformer'] = [j]
            if ordinal and theme is not None:
                onehot_categories = [["Couldn't be improved", 'Access', 'Care received', 'Communication', 'Dignity',
                                      'Environment/ facilities', 'Miscellaneous', 'Staff', 'Transition/coordination']]
                aux['alltrans__theme__scaler'] = \
                    [OneHotEncoder(categories=onehot_categories), ThemeBinarizer(class_col='theme',
                                                                                 target_class="Couldn't be improved")]

            # if i.__class__.__name__ == LinearSVC().__class__.__name__:
            #     aux['clf__estimator__max_iter'] = [10000]
            #     aux['clf__estimator__class_weight'] = [None, 'balanced']
            #     # aux['clf__estimator__dual'] = [True, False] # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
            if i.__class__.__name__ == BernoulliNB().__class__.__name__:
                aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
            if i.__class__.__name__ == ComplementNB().__class__.__name__:
                aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
            if i.__class__.__name__ == MultinomialNB().__class__.__name__:
                aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
            if i.__class__.__name__ == SGDClassifier().__class__.__name__:
                aux['clf__estimator__max_iter'] = [10000]
                aux['clf__estimator__class_weight'] = [None, 'balanced']
                aux['clf__estimator__penalty'] = ('l2', 'elasticnet')
                if ordinal:
                    aux['clf__estimator__loss'] = ['log']
                else:
                    aux['clf__estimator__loss'] = ['hinge', 'log']
            if i.__class__.__name__ == RidgeClassifier().__class__.__name__:
                aux['clf__estimator__class_weight'] = [None, 'balanced']
                aux['clf__estimator__alpha'] = (0.1, 1.0, 10.0)
            if i.__class__.__name__ == Perceptron().__class__.__name__:
                aux['clf__estimator__class_weight'] = [None, 'balanced']
                aux['clf__estimator__penalty'] = ('l2', 'elasticnet')
            if i.__class__.__name__ == RandomForestClassifier().__class__.__name__:
                aux['clf__estimator__max_features'] = ('sqrt', 0.666)

            if j.__class__.__name__ == TfidfVectorizer().__class__.__name__:
                aux['featsel__selector__score_func'] = [chi2]
                aux['preprocessor__texttr__text__transformer__tokenizer'] = [LemmaTokenizer(tknz)]
                aux['preprocessor__texttr__text__transformer__preprocessor'] = [text_preprocessor]
                aux['preprocessor__texttr__text__transformer__norm'] = ['l2']
                aux['preprocessor__texttr__text__transformer__ngram_range'] = ((1, 3), (2, 3), (3, 3))
                aux['preprocessor__texttr__text__transformer__max_df'] = [0.7, 0.95]
                aux['preprocessor__texttr__text__transformer__min_df'] = [3, 1]
                aux['preprocessor__texttr__text__transformer__use_idf'] = [True, False]

                # The transformation is a k-means discretizer with 3 bins:
                #   1. The three bins represent short, medium and long text length. Reluctant to make n_bins a tunable
                #      parameter for efficiency reasons;
                #   2. Discretizing and one-hot encoding satisfies the data format requirements for Chi^2-based feature
                #      selection;
                #   3. An added benefit is that this data format is acceptable by different models, some of which may
                #      not be scale-invariant, while others do not accept negative or continuous values other than
                #      TF-IDFs;
                aux['preprocessor__lengthtr__scaler__scaler'] = \
                    [KBinsDiscretizer(n_bins=3, encode='onehot', strategy='kmeans')]

                # The transformation is a k-means discretizer with 4 or 8 bins supplied as a tunable argument later on:
                #   1. The 4 bins represent weak, weak-medium, medium-strong and strong for values in [0, 1];
                #   2. The 8 bins represent weak, weak-medium, medium-strong and strong for values in [-1, 0] and [0, 1]
                #      (i.e. 8 bins for values in [-1, 1]);
                #   3. We also allow for the possibility of 8 bins for [0, 1] and 4 bins for [-1, 1]- no harm in trying;
                #   4. Discretizing and one-hot encoding satisfies the data format requirements for Chi^2-based feature
                #      selection;
                #   5. An added benefit is that this data format is acceptable by different models, some of which may
                #      not be scale-invariant, while others do not accept negative or continuous values other than
                #      TF-IDFs;
                aux['preprocessor__sentimenttr__scaler__scaler'] = [KBinsDiscretizer(encode='onehot', strategy='kmeans')]
                aux['preprocessor__sentimenttr__scaler__scaler__n_bins'] = [4, 8] # Based on the idea of having 4 (8) bins for indicators in [0, 1] ([-1, 1]), but open to trying 8 (4) for [0, 1] ([-1, 1]) too.

                param_grid.append(aux)

                aux1 = aux.copy()
                aux1['preprocessor__texttr__text__transformer__use_idf'] = [False]
                aux1['preprocessor__texttr__text__transformer__norm'] = [None]

                param_grid.append(aux1)

            if j.__class__.__name__ == EmbeddingsTransformer().__class__.__name__:
                aux['featsel__selector__score_func'] = [f_classif]
                aux['preprocessor__lengthtr__scaler__scaler'] = [StandardScaler()]
                aux['preprocessor__sentimenttr__scaler__scaler'] = [StandardScaler()]

                # We don't want learners than can't handle negative data in the embeddings.
                if (i.__class__.__name__ == BernoulliNB().__class__.__name__) or \
                        (i.__class__.__name__ == ComplementNB().__class__.__name__) or \
                        (i.__class__.__name__ == MultinomialNB().__class__.__name__):
                    aux = None

                param_grid.append(aux)

    param_grid = [x for x in param_grid if x is not None]

    # When a theme is supplied for the ordinal model, the pipeline steps are a little different. Step "alltrans"
    # includes the steps for both the preprocessing of the text feature, and the one-hot encoding of the theme feature.
    # So, a parameter such as "featsel__selector" in the pipeline without a theme feature would be
    # "alltrans__process__featsel__selector" in this one. We need to pass these correct names to the tuning grid.
    if ordinal and theme is not None:
        ordinal_with_theme_params = [
            'featsel__selector',
            'featsel__selector__percentile',
            'featsel__selector__score_func',
            'preprocessor__sentimenttr__scaler__scaler',
            'preprocessor__sentimenttr__scaler__scaler__n_bins',
            'preprocessor__lengthtr__scaler__scaler',
            'preprocessor__texttr__text__transformer',
            'preprocessor__texttr__text__transformer__tokenizer',
            'preprocessor__texttr__text__transformer__preprocessor',
            'preprocessor__texttr__text__transformer__norm',
            'preprocessor__texttr__text__transformer__ngram_range',
            'preprocessor__texttr__text__transformer__max_df',
            'preprocessor__texttr__text__transformer__min_df',
            'preprocessor__texttr__text__transformer__use_idf']

        for i in range(len(param_grid)):
            for j in ordinal_with_theme_params:
                if j in param_grid[i].keys():
                    old_key = j
                    new_key = 'alltrans__process__' + old_key
                    param_grid[i][new_key] = param_grid[i].pop(old_key)

    # Define fitting metric (refit) and other useful performance metrics.
    refit = metric.replace('_', ' ').replace(' score', '').title()
    scoring = {'Accuracy': make_scorer(accuracy_score),
               'Balanced Accuracy': make_scorer(balanced_accuracy_score),
               'Matthews Correlation Coefficient': make_scorer(matthews_corrcoef),
               'Class Balance Accuracy': make_scorer(class_balance_accuracy_score)}

    # Define pipeline #
    pipe_cv = RandomizedSearchCV(pipe, param_grid, n_jobs=n_jobs, return_train_score=False,
                                 cv=cv, verbose=verbose,
                                 scoring=scoring, refit=refit, n_iter=n_iter)

    # These messages are for function helpers.text_preprocessor which is used by
    # TfidfVectorizer() and EmbeddingsTransformer(). Having them inside text_preprocessor() prints
    # them in each iteration, which is redundant. Having the here prints them once.
    print('Stripping punctuation from text...')
    print("Stripping excess spaces, whitespaces and line breaks from text...")

    # Fit pipeline #
    pipe_cv.fit(x, y)

    return pipe_cv
