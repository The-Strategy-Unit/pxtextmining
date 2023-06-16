import pytest
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
