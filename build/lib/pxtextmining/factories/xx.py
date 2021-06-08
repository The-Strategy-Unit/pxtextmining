
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
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
from helpers.text_preprocessor import text_preprocessor
from helpers.sentiment_scores import sentiment_scores
from helpers.text_length import text_length
from helpers.tokenization import LemmaTokenizer
from helpers.oversampling import random_over_sampler_data_generator
from helpers.metrics import class_balance_accuracy_score
from helpers.estimator_switcher import ClfSwitcher
from helpers.scaler_switcher import ScalerSwitcher
import pandas as pd
from os import path
import mysql.connector

tknz = 'spacy'
target = 'label'
predictor = 'feedback'
db = mysql.connector.connect(option_files="my.conf", use_pure=True)
with db.cursor() as cursor:
    cursor.execute(
        "SELECT  " + target + ", " + predictor + " FROM text_data"
    )
    text_data = cursor.fetchall()
    text_data = pd.DataFrame(text_data)
    text_data.columns = cursor.column_names

text_data = text_data.rename(columns={target: "target", predictor: "predictor"})
text_data = text_data.loc[text_data.target.notnull()].copy()

features_text = 'predictor'
features_in_minus_1_to_1 = ['text_blob_polarity', 'vader_compound']
features_in_0_to_1 = ['text_blob_subjectivity', 'vader_neg', 'vader_neu', 'vader_pos']
features_positive_and_unbounded = ['text_length']

text_length_transformer = Pipeline(steps=[
        ('length', (FunctionTransformer(text_length)))
    ])

sentiment_transformer = Pipeline(steps=[
    ('sentiment', (FunctionTransformer(sentiment_scores)))
])

text_transformer = Pipeline(steps=[
    ('tfidf', (TfidfVectorizer(tokenizer=LemmaTokenizer(tknz),
                               preprocessor=text_preprocessor)))
])

pp = Pipeline(steps=[
    ('a1', ColumnTransformer(
        transformers=[
            ('text', text_transformer, features_text)])),

    ('union', FeatureUnion(
        transformer_list=[
            ('st', Pipeline([
                ('xx1', ColumnTransformer(
                    transformers=[
                        ('st1', sentiment_transformer, features_text)
                    ]
                ))
            ])),

            ('tlt', Pipeline([
                ('xx2', ColumnTransformer(
                    transformers=[
                        ('tlt1', text_length_transformer, features_text)
                    ]
                ))
            ]))

        ])),

    ('a4', ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), features_in_minus_1_to_1)]))
])

pp.fit_transform(text_data)

#######################

pp1 = Pipeline(steps=[
    ('a1', ColumnTransformer(
        transformers=[
            ('text', text_transformer, features_text)]))
])

pp1.fit_transform(text_data)

#####################################
pp2 = Pipeline(steps=[
    ('union', FeatureUnion(
        transformer_list=[
            ('st', Pipeline([
                ('xx1', ColumnTransformer(
                    transformers=[
                        ('st1', sentiment_transformer, features_text)
                    ]
                ))
            ])),

            ('tlt', Pipeline([
                ('xx2', ColumnTransformer(
                    transformers=[
                        ('tlt1', text_length_transformer, features_text)
                    ]
                ))
            ]))

        ]))
])

pp2.fit_transform(text_data)

###############################
# https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html
# https://github.com/marrrcin/pandas-feature-union/blob/master/pandas_feature_union.py

pp22 = Pipeline(steps=[
    ('union', PandasFeatureUnion(
        transformer_list=[
            ('st', Pipeline([
                ('xx1', ColumnTransformer(
                    transformers=[
                        ('st1', sentiment_transformer, [features_text])
                    ]
                ))
            ])),

            ('tlt', Pipeline([
                ('xx2', ColumnTransformer(
                    transformers=[
                        ('tlt1', text_length_transformer, [features_text])
                    ]
                ))
            ]))

        ]))
])

pp22.fit_transform(text_data)

#####################################
# https://github.com/scikit-learn-contrib/sklearn-pandas/blob/master/sklearn_pandas/dataframe_mapper.py

pp23 = DataFrameMapper([
    (features_text, FunctionTransformer(sentiment_scores)),
    (features_text, FunctionTransformer(text_length), {'alias': 'text_length'})],
    input_df=True, df_out=True
)

pp23.fit_transform(text_data)

['text_blob_polarity', 'text_blob_subjectivity', 'vader_neg',
       'vader_neu', 'vader_pos', 'vader_compound']