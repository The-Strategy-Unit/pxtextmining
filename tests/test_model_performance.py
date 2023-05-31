from pxtextmining.factories import factory_model_performance
import pandas as pd
import numpy as np


def test_multiclass_metrics(grab_test_X_additional_feats):
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


def test_multilabel_metrics(grab_test_X_additional_feats):
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
        x, y, labels, random_state, model_type, model, additional_features = additional_features
    )
    assert type(metrics_string) == str
