from unittest.mock import Mock, patch

import numpy as np
import pytest
from keras.engine.functional import Functional
from sklearn.base import is_classifier

from pxtextmining.factories import factory_pipeline


@pytest.mark.parametrize("model_type", ["svm", "xgb"])
@pytest.mark.parametrize("additional_features", [True, False])
def test_create_sklearn_pipeline_sentiment(model_type, additional_features):
    pipe, params = factory_pipeline.create_sklearn_pipeline_sentiment(
        model_type, 3, tokenizer=None, additional_features=additional_features
    )
    assert type(params) == dict
    assert is_classifier(pipe) is True


@pytest.mark.parametrize("multilabel", [True, False])
def test_create_bert_model(multilabel):
    Y_train = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    model = factory_pipeline.create_bert_model(Y_train, multilabel=multilabel)
    assert type(model) == Functional


@pytest.mark.parametrize("multilabel", [True, False])
def test_create_bert_model_additional_features(multilabel):
    Y_train = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    model = factory_pipeline.create_bert_model_additional_features(
        Y_train, multilabel=multilabel
    )
    assert type(model) == Functional


def test_train_bert_model():
    train_dataset = Mock()
    test_dataset = Mock()
    model = Mock()
    model, training_time = factory_pipeline.train_bert_model(
        train_dataset, test_dataset, model
    )
    model.fit.assert_called_once()
    assert type(training_time) == str


def test_calculating_class_weights():
    Y_train = np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
    )
    class_weights_dict = factory_pipeline.calculating_class_weights(Y_train)
    assert type(class_weights_dict) == dict


@pytest.mark.parametrize("model_type", ["svm", "xgb", "rfc", "mnb", "knn"])
@pytest.mark.parametrize("additional_features", [True, False])
@pytest.mark.parametrize("tokenizer", [None, "spacy"])
def test_create_sklearn_pipeline(model_type, tokenizer, additional_features):
    pipe, params = factory_pipeline.create_sklearn_pipeline(
        model_type, tokenizer, additional_features
    )
    assert is_classifier(pipe) is True
    assert type(params) == dict


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_search_sklearn_pipelines(
    mock_randomized_search,
    grab_test_X_additional_feats,
):
    mock_model = Mock()
    mock_randomized_search.fit.return_value = mock_model
    mock_randomized_search.best_estimator_ = mock_model
    models_to_try = ["knn"]
    X_train = grab_test_X_additional_feats
    Y_train = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    models, training_times = factory_pipeline.search_sklearn_pipelines(
        X_train,
        Y_train,
        models_to_try,
        additional_features=True,
    )
    assert len(models) == len(training_times)


@patch("sklearn.model_selection.RandomizedSearchCV")
def test_search_sklearn_pipelines_addf_false(
    mock_randomized_search,
    grab_test_X_additional_feats,
):
    mock_model = Mock()
    mock_randomized_search.fit.return_value = mock_model
    mock_randomized_search.best_estimator_ = mock_model
    models_to_try = ["knn"]
    X_train = grab_test_X_additional_feats["FFT answer"]
    Y_train = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    models, training_times = factory_pipeline.search_sklearn_pipelines(
        X_train,
        Y_train,
        models_to_try,
        additional_features=False,
    )
    assert len(models) == len(training_times)
