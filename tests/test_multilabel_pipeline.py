from unittest.mock import patch

from pxtextmining.pipelines import multilabel_pipeline


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
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_skpipeline.return_value = (["model"], ["training_time"])

    # act
    multilabel_pipeline.run_sklearn_pipeline(include_analysis=True)

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_skpipeline.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
    mock_writepreds.assert_called_once()
    mock_writeanalysis.assert_called_once()
