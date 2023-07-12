from unittest.mock import patch

import pytest

from pxtextmining.params import major_cats, merged_minor_cats, minor_cats
from pxtextmining.pipelines import multilabel_pipeline


@pytest.mark.parametrize("target", [major_cats, minor_cats, merged_minor_cats])
@patch("pxtextmining.pipelines.multilabel_pipeline.write_model_analysis")
@patch("pxtextmining.pipelines.multilabel_pipeline.write_model_preds")
@patch("pxtextmining.pipelines.multilabel_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.get_multilabel_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.search_sklearn_pipelines")
@patch("pxtextmining.pipelines.multilabel_pipeline.process_and_split_data", create=True)
@patch("pxtextmining.pipelines.multilabel_pipeline.load_multilabel_data")
def test_sklearn_pipeline(
    mock_dataload,
    mock_datasplit,
    mock_skpipeline,
    mock_metrics,
    mock_write,
    mock_writepreds,
    mock_writeanalysis,
    target,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_skpipeline.return_value = (["model"], ["training_time"])

    # act
    multilabel_pipeline.run_sklearn_pipeline(target=target, include_analysis=True)

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_skpipeline.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
    mock_writepreds.assert_called_once()
    mock_writeanalysis.assert_called_once()


@pytest.mark.parametrize("target", [major_cats, minor_cats, merged_minor_cats])
@patch("pxtextmining.pipelines.multilabel_pipeline.write_model_analysis")
@patch("pxtextmining.pipelines.multilabel_pipeline.write_model_preds")
@patch("pxtextmining.pipelines.multilabel_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.get_multilabel_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.create_and_train_svc_model")
@patch("pxtextmining.pipelines.multilabel_pipeline.process_and_split_data", create=True)
@patch("pxtextmining.pipelines.multilabel_pipeline.load_multilabel_data")
def test_svc_pipeline(
    mock_dataload,
    mock_datasplit,
    mock_skpipeline,
    mock_metrics,
    mock_write,
    mock_writepreds,
    mock_writeanalysis,
    target,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_skpipeline.return_value = ("model", "training_time")

    # act
    multilabel_pipeline.run_svc_pipeline(target=target, include_analysis=True)

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_skpipeline.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
    mock_writepreds.assert_called_once()
    mock_writeanalysis.assert_called_once()


@pytest.mark.parametrize("target", [major_cats, minor_cats, merged_minor_cats])
@patch("pxtextmining.pipelines.multilabel_pipeline.write_model_analysis")
@patch("pxtextmining.pipelines.multilabel_pipeline.write_model_preds")
@patch("pxtextmining.pipelines.multilabel_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.get_multilabel_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.train_bert_model")
@patch("pxtextmining.pipelines.multilabel_pipeline.create_bert_model")
@patch("pxtextmining.pipelines.multilabel_pipeline.calculating_class_weights")
@patch("pxtextmining.pipelines.multilabel_pipeline.bert_data_to_dataset")
@patch("pxtextmining.pipelines.multilabel_pipeline.train_test_split")
@patch("pxtextmining.pipelines.multilabel_pipeline.process_and_split_data")
@patch("pxtextmining.pipelines.multilabel_pipeline.load_multilabel_data")
def test_bert_pipeline(
    mock_dataload,
    mock_datasplit,
    mock_traintest,
    mock_bertdata,
    mock_classweights,
    mock_createbert,
    mock_trainbert,
    mock_metrics,
    mock_write,
    mock_writepreds,
    mock_writeanalysis,
    target,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_traintest.return_value = ("X_train_val", "X_test", "Y_train_val", "Y_test")
    mock_trainbert.return_value = (1, 2)

    # act
    multilabel_pipeline.run_bert_pipeline(target=target, include_analysis=True)

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_traintest.assert_called_once()
    mock_bertdata.assert_called()
    mock_classweights.assert_called_once()
    mock_createbert.assert_called_once()
    mock_trainbert.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
    mock_writepreds.assert_called_once()
    mock_writeanalysis.assert_called_once()


@pytest.mark.parametrize("target", [major_cats, minor_cats, merged_minor_cats])
@patch("pxtextmining.pipelines.multilabel_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.get_multilabel_metrics")
@patch("pxtextmining.pipelines.multilabel_pipeline.train_bert_model")
@patch(
    "pxtextmining.pipelines.multilabel_pipeline.create_bert_model_additional_features"
)
@patch("pxtextmining.pipelines.multilabel_pipeline.calculating_class_weights")
@patch("pxtextmining.pipelines.multilabel_pipeline.bert_data_to_dataset")
@patch("pxtextmining.pipelines.multilabel_pipeline.train_test_split")
@patch("pxtextmining.pipelines.multilabel_pipeline.process_and_split_data")
@patch("pxtextmining.pipelines.multilabel_pipeline.load_multilabel_data")
def test_bert_pipeline_additional_features(
    mock_dataload,
    mock_datasplit,
    mock_traintest,
    mock_bertdata,
    mock_classweights,
    mock_createbert,
    mock_trainbert,
    mock_metrics,
    mock_write,
    target,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_traintest.return_value = ("X_train_val", "X_test", "Y_train_val", "Y_test")
    mock_trainbert.return_value = (1, 2)

    # act
    multilabel_pipeline.run_bert_pipeline(
        target=target, additional_features=True, include_analysis=False
    )

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_traintest.assert_called_once()
    mock_bertdata.assert_called()
    mock_classweights.assert_called_once()
    mock_createbert.assert_called_once()
    mock_trainbert.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
