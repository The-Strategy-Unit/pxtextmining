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


@patch("pxtextmining.pipelines.sentiment_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.sentiment_pipeline.get_multiclass_metrics")
@patch("pxtextmining.pipelines.sentiment_pipeline.train_bert_model")
@patch("pxtextmining.pipelines.sentiment_pipeline.create_bert_model")
@patch("pxtextmining.pipelines.sentiment_pipeline.compute_class_weight")
@patch("pxtextmining.pipelines.sentiment_pipeline.bert_data_to_dataset")
@patch("pxtextmining.pipelines.sentiment_pipeline.train_test_split")
@patch("pxtextmining.pipelines.sentiment_pipeline.to_categorical")
@patch("pxtextmining.pipelines.sentiment_pipeline.process_and_split_data")
@patch("pxtextmining.pipelines.sentiment_pipeline.load_multilabel_data")
def test_bert_pipeline(
    mock_dataload,
    mock_datasplit,
    mock_categorical,
    mock_traintest,
    mock_bertdata,
    mock_classweights,
    mock_createbert,
    mock_trainbert,
    mock_metrics,
    mock_write,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_traintest.return_value = ("X_train_val", "X_test", "Y_train_val", "Y_test")
    mock_trainbert.return_value = (1, 2)

    # act
    sentiment_pipeline.run_sentiment_bert_pipeline(additional_features=False)

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_categorical.assert_called_once()
    mock_traintest.assert_called_once()
    mock_bertdata.assert_called()
    mock_classweights.assert_called_once()
    mock_createbert.assert_called_once()
    mock_trainbert.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()


@patch("pxtextmining.pipelines.sentiment_pipeline.write_multilabel_models_and_metrics")
@patch("pxtextmining.pipelines.sentiment_pipeline.get_multiclass_metrics")
@patch("pxtextmining.pipelines.sentiment_pipeline.train_bert_model")
@patch(
    "pxtextmining.pipelines.sentiment_pipeline.create_bert_model_additional_features"
)
@patch("pxtextmining.pipelines.sentiment_pipeline.compute_class_weight")
@patch("pxtextmining.pipelines.sentiment_pipeline.bert_data_to_dataset")
@patch("pxtextmining.pipelines.sentiment_pipeline.train_test_split")
@patch("pxtextmining.pipelines.sentiment_pipeline.to_categorical")
@patch("pxtextmining.pipelines.sentiment_pipeline.process_and_split_data")
@patch("pxtextmining.pipelines.sentiment_pipeline.load_multilabel_data")
def test_bert_pipeline_additional_features(
    mock_dataload,
    mock_datasplit,
    mock_categorical,
    mock_traintest,
    mock_bertdata,
    mock_classweights,
    mock_createbert,
    mock_trainbert,
    mock_metrics,
    mock_write,
):
    # arrange mocks
    mock_datasplit.return_value = (1, 2, 3, 4)
    mock_traintest.return_value = ("X_train_val", "X_test", "Y_train_val", "Y_test")
    mock_trainbert.return_value = (1, 2)
    mock_classweights.return_value = [0.5, 0.2]

    # act
    sentiment_pipeline.run_sentiment_bert_pipeline(additional_features=True)

    # assert
    mock_dataload.assert_called_once()
    mock_datasplit.assert_called_once()
    mock_categorical.assert_called_once()
    mock_traintest.assert_called_once()
    mock_bertdata.assert_called()
    mock_classweights.assert_called_once()
    mock_createbert.assert_called_once()
    mock_trainbert.assert_called_once()
    mock_metrics.assert_called_once()
    mock_write.assert_called_once()
