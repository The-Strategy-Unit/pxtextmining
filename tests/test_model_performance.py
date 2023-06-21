from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from tensorflow.keras import Model

from pxtextmining.factories import factory_model_performance


@pytest.fixture
def grab_test_bert_multiclass():
    predicted_probs = np.array(
        [
            [0.9, 0.01, 0.07, 0.01, 0.01],
            [0.01, 0.07, 0.01, 0.01, 0.9],
            [0.07, 0.9, 0.01, 0.01, 0.01],
            [0.9, 0.01, 0.07, 0.01, 0.01],
            [0.9, 0.01, 0.01, 0.01, 0.07],
        ]
    )
    model = Mock(spec=Model, predict=Mock(return_value=predicted_probs))
    return model


@pytest.fixture
def grab_test_bert_multilabel():
    predicted_probs = np.array(
        [
            [
                6.2770307e-01,
                2.3520987e-02,
                1.3149388e-01,
                2.7835215e-02,
                1.8944685e-01,
            ],
            [
                9.8868138e-01,
                1.9990385e-03,
                5.4453085e-03,
                9.0726715e-04,
                2.9669846e-03,
            ],
            [
                4.2310607e-01,
                5.6546849e-01,
                9.3136989e-03,
                1.3205722e-03,
                7.9117226e-04,
            ],
            [
                2.0081511e-01,
                7.0609129e-04,
                1.1107661e-03,
                7.9677838e-01,
                5.8961433e-04,
            ],
            [
                1.4777037e-03,
                5.1493715e-03,
                2.8268427e-03,
                7.4673461e-04,
                9.8979920e-01,
            ],
        ]
    )
    model = Mock(spec=Model, predict=Mock(return_value=predicted_probs))
    return model


def test_multiclass_metrics_sklearn(grab_test_X_additional_feats):
    x = grab_test_X_additional_feats
    y = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    labels = ["A", "B", "C"]
    model = factory_model_performance.get_dummy_model(x, y)
    random_state = 42
    additional_features = True
    metrics_string = factory_model_performance.get_multiclass_metrics(
        x, y, labels, random_state, model, additional_features
    )
    assert type(metrics_string) == str


def test_multiclass_metrics_bert(
    grab_test_X_additional_feats, grab_test_bert_multiclass
):
    x = grab_test_X_additional_feats
    y = np.array(
        [
            [0],
            [4],
            [1],
            [3],
            [3],
        ]
    )
    labels = ["A", "B", "C", "D"]
    model = grab_test_bert_multiclass
    random_state = 42
    additional_features = True
    metrics_string = factory_model_performance.get_multiclass_metrics(
        x, y, labels, random_state, model, additional_features
    )
    assert type(metrics_string) == str


def test_multilabel_metrics_sklearn(grab_test_X_additional_feats):
    x = grab_test_X_additional_feats
    y = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    labels = ["A", "B", "C", "D", "E"]
    random_state = 42
    model_type = "sklearn"
    additional_features = True
    model = factory_model_performance.get_dummy_model(x, y)
    metrics_string = factory_model_performance.get_multilabel_metrics(
        x,
        y,
        labels,
        random_state,
        model_type,
        model,
        additional_features=additional_features,
    )
    assert type(metrics_string) == str


def test_multilabel_metrics_bert(
    grab_test_X_additional_feats, grab_test_bert_multilabel
):
    x = grab_test_X_additional_feats
    y = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    labels = ["A", "B", "C", "D", "E"]
    random_state = 42
    model_type = "bert"
    additional_features = True
    model = grab_test_bert_multilabel
    metrics_string = factory_model_performance.get_multilabel_metrics(
        x,
        y,
        labels,
        random_state,
        model_type,
        model,
        additional_features=additional_features,
    )
    assert type(metrics_string) == str


def test_accuracy_per_class():
    y_test = pd.Series([0, 1, 0, 2, 1, 0])
    y_pred = pd.Series([0, 1, 0, 1, 1, 2])
    df = factory_model_performance.get_accuracy_per_class(y_test, y_pred)
    assert df.shape == (3, 3)


def test_parse_metrics_file():
    metrics_file = "current_best_multilabel/bert_sentiment.txt"
    labels = ["very positive", "positive", "neutral", "negative", "very negative"]
    metrics_df = factory_model_performance.parse_metrics_file(metrics_file, labels)
    assert metrics_df.shape == (5, 5)
