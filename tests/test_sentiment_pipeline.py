from unittest.mock import patch

from pxtextmining.pipelines import sentiment_pipeline


@patch("pxtextmining.pipelines.sentiment_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.sentiment_pipeline.get_multiclass_metrics")
@patch("pxtextmining.pipelines.sentiment_pipeline.search_sklearn_pipelines")
@patch("pxtextmining.pipelines.sentiment_pipeline.process_and_split_data", create=True)
@patch("pxtextmining.pipelines.sentiment_pipeline.load_multilabel_data")
def test_sentiment_pipeline(
    mock_dataload,
    mock_datasplit,
    mock_skpipeline,
    mock_metrics,
    mock_write,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_skpipeline.return_value = (["model"], ["training_time"])

    # act
    sentiment_pipeline.run_sentiment_pipeline()

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_skpipeline.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
