from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from keras.src.engine.functional import Functional
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

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


@pytest.mark.parametrize("target", ["sentiment", None])
@patch("pxtextmining.factories.factory_pipeline.RandomizedSearchCV")
def test_search_sklearn_pipelines(mock_search, target, grab_test_X_additional_feats):
    mock_instance = MagicMock()
    mock_search.return_value = mock_instance
    models_to_try = ["svm"]
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
    mock_instance.best_estimator_ = Pipeline([("dummy", None)])
    mock_instance.best_params_ = {"param1": 10, "param2": 20}

    models, training_times = factory_pipeline.search_sklearn_pipelines(
        X_train, Y_train, models_to_try, target=target, additional_features=True
    )

    mock_instance.fit.assert_called()
    assert len(models) == 1
    assert isinstance(models[0], Pipeline) is True
    assert models[0].steps[0][0] == "dummy"
    assert len(training_times) == 1


@pytest.mark.parametrize("target", ["sentiment", None])
@patch("pxtextmining.factories.factory_pipeline.RandomizedSearchCV")
def test_search_sklearn_pipelines_no_feats(
    mock_search, target, grab_test_X_additional_feats
):
    mock_instance = MagicMock()
    mock_search.return_value = mock_instance
    models_to_try = ["svm"]
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
    mock_instance.best_estimator_ = Pipeline([("dummy", None)])
    mock_instance.best_params_ = {"param1": 10, "param2": 20}

    models, training_times = factory_pipeline.search_sklearn_pipelines(
        X_train, Y_train, models_to_try, target=target, additional_features=False
    )

    mock_instance.fit.assert_called()
    assert len(models) == 1
    assert isinstance(models[0], Pipeline) is True
    assert models[0].steps[0][0] == "dummy"
    assert len(training_times) == 1
