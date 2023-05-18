from pxtextmining.factories import factory_predict_unlabelled_text
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock

def test_get_probabilities_bert():
    label_series = pd.Series([["label_one"], ["label_two", "label_three"]], name="test")
    labels = ["label_one", "label_two", "label_three"]
    predicted_probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.7]])
    model_type = "bert"
    test_probability_s = factory_predict_unlabelled_text.get_probabilities(
        label_series, labels, predicted_probabilities, model_type
    )
    assert len(test_probability_s.iloc[0]) == 1
    assert test_probability_s.iloc[1]["label_two"] == 0.9
    assert type(test_probability_s) == pd.Series
    assert len(test_probability_s) == len(label_series)


def test_get_probabilities_sklearn():
    label_series = pd.Series([["label_one"], ["label_two", "label_three"]], name="test")
    labels = ["label_one", "label_two", "label_three"]
    predicted_probabilities = np.array(
        [[[0.2, 0.8], [0.9, 0.1]], [[0.9, 0.1], [0.1, 0.9]], [[0.9, 0.1], [0.3, 0.7]]]
    )
    model_type = "sklearn"
    test_probability_s = factory_predict_unlabelled_text.get_probabilities(
        label_series, labels, predicted_probabilities, model_type
    )
    assert len(test_probability_s.iloc[0]) == 1
    assert test_probability_s.iloc[1]["label_two"] == 0.9
    assert type(test_probability_s) == pd.Series
    assert len(test_probability_s) == len(label_series)


def test_predict_with_bert(grab_test_X_additional_feats):
    #arrange
    data = grab_test_X_additional_feats
    model = Mock(predict=Mock(return_value=np.array(
        [
            [6.2770307e-01, 2.3520987e-02, 1.3149388e-01, 2.7835215e-02, 1.8944685e-01],
            [9.8868138e-01, 1.9990385e-03, 5.4453085e-03, 9.0726715e-04, 2.9669846e-03],
            [4.2310607e-01, 5.6546849e-01, 9.3136989e-03, 1.3205722e-03, 7.9117226e-04],
            [2.0081511e-01, 7.0609129e-04, 1.1107661e-03, 7.9677838e-01, 5.8961433e-04],
            [1.4777037e-03, 5.1493715e-03, 2.8268427e-03, 7.4673461e-04, 9.8979920e-01],
        ]
    )))
    #act
    predictions = factory_predict_unlabelled_text.predict_with_bert(data, model, additional_features = True,
                                                      already_encoded=False)
    #assert
    assert type(predictions) == np.ndarray
