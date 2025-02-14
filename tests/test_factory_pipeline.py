from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from keras.engine.functional import Functional
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from pxtextmining.factories import factory_pipeline


@pytest.mark.parametrize("model_type", ["svm", "xgb"])
@pytest.mark.parametrize("additional_features", [True, False])
def test_create_sklearn_pipeline_sentiment(model_type, additional_features):
    pipe, params = factory_pipeline.create_sklearn_pipeline_sentiment(
        model_type, 3, additional_features=additional_features
    )
    assert isinstance(params, dict) is True
    assert is_classifier(pipe) is True


@pytest.mark.parametrize("multilabel", [True, False])
def test_create_bert_model(multilabel):
    Y_train = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    model = factory_pipeline.create_bert_model(Y_train, multilabel=multilabel)
    assert isinstance(model, Functional) is True


@pytest.mark.parametrize("multilabel", [True, False])
def test_create_bert_model_additional_features(multilabel):
    Y_train = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    model = factory_pipeline.create_bert_model_additional_features(
        Y_train, multilabel=multilabel
    )
    assert isinstance(model, Functional) is True


def test_train_bert_model():
    train_dataset = Mock()
    test_dataset = Mock()
    model = Mock()
    model, training_time = factory_pipeline.train_bert_model(
        train_dataset, test_dataset, model
    )
    model.fit.assert_called_once()
    assert isinstance(training_time, str) is True


def test_calculating_class_weights():
    Y_train = np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
    )
    class_weights_dict = factory_pipeline.calculating_class_weights(Y_train)
    assert isinstance(class_weights_dict, dict) is True


@pytest.mark.parametrize("model_type", ["svm", "xgb", "rfc", "mnb", "knn"])
@pytest.mark.parametrize("additional_features", [True, False])
def test_create_sklearn_pipeline(model_type, additional_features):
    pipe, params = factory_pipeline.create_sklearn_pipeline(
        model_type, additional_features
    )
    assert is_classifier(pipe) is True
    assert isinstance(params, dict) is True


@pytest.mark.parametrize("target", ["sentiment", None])
@pytest.mark.parametrize("model_type", [["svm"], ["xgb"]])
@patch("pxtextmining.factories.factory_pipeline.RandomizedSearchCV")
def test_search_sklearn_pipelines(
    mock_randomsearch, target, model_type, grab_test_X_additional_feats
):
    mock_instance = MagicMock()
    mock_randomsearch.return_value = mock_instance
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
        X_train,
        Y_train,
        models_to_try=model_type,
        target=target,
        additional_features=True,
    )

    mock_instance.fit.assert_called()
    assert len(models) == 1
    assert isinstance(models[0], Pipeline) is True
    assert models[0].steps[0][0] == "dummy"
    assert len(training_times) == 1

    with pytest.raises(ValueError):
        factory_pipeline.search_sklearn_pipelines(
            X_train,
            Y_train,
            models_to_try=["nonsense"],
            target=target,
            additional_features=True,
        )


@pytest.mark.parametrize("target", ["sentiment", None])
@patch("pxtextmining.factories.factory_pipeline.RandomizedSearchCV")
def test_search_sklearn_pipelines_no_feats(
    mock_randomsearch, target, grab_test_X_additional_feats
):
    mock_instance = MagicMock()
    mock_randomsearch.return_value = mock_instance
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


@patch("pxtextmining.factories.factory_pipeline.make_pipeline")
def test_create_and_train_svc_model(mock_pipeline, grab_test_X_additional_feats):
    mock_pipe = Mock()
    mock_pipeline.return_value = mock_pipe
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
    factory_pipeline.create_and_train_svc_model(
        X_train, Y_train, additional_features=True
    )
    mock_pipe.fit.assert_called_with(X_train, Y_train)


@patch("pxtextmining.factories.factory_pipeline.make_pipeline")
def test_create_and_train_svc_model_no_feats(
    mock_pipeline, grab_test_X_additional_feats
):
    mock_pipe = Mock()
    mock_pipeline.return_value = mock_pipe
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
    factory_pipeline.create_and_train_svc_model(
        X_train, Y_train, additional_features=False
    )
    mock_pipe.fit.assert_called_with(X_train, Y_train)
