"""Create separate pipeline for preprocessing here. It will be combined with the
model tuning pipeline. No need to RandomSearch best hyperparams for this pipeline
"""

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


def create_preprocessing_pipe(x, y, tknz="spacy", ordinal=False, theme=None):
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

    return pipe
