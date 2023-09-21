import json
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

import docker_run


@patch("docker_run.load_model")
def test_load_sentiment_model(mock_load):
    docker_run.load_sentiment_model()
    mock_load.assert_called_once()


@patch("docker_run.predict_sentiment_bert")
def test_get_sentiment_predictions(mock_predict):
    docker_run.get_sentiment_predictions(
        "text", "model", preprocess_text=True, additional_features=True
    )
    mock_predict.assert_called_with(
        "text", "model", preprocess_text=True, additional_features=True
    )


@patch("docker_run.get_sentiment_predictions")
@patch("docker_run.load_model")
def test_predict_sentiment(mock_load_model, mock_get_predictions):
    input_text = [
        {
            "comment_id": "1",
            "comment_text": "Nurse was great.",
            "question_type": "what_good",
        },
        {"comment_id": "2", "comment_text": "", "question_type": "could_improve"},
    ]
    output = pd.DataFrame(
        [
            {
                "Comment ID": "1",
                "FFT answer": "Nurse was great.",
                "FFT_q_standardised": "what_good",
                "sentiment": 1,
            }
        ]
    ).set_index("Comment ID")
    mock_get_predictions.return_value = output
    docker_run.predict_sentiment(input_text)
    mock_load_model.assert_called_once()
    mock_get_predictions.assert_called()


@pytest.mark.parametrize("args", [["file_01.json"], ["file_01.json", "-l"]])
def test_parse_args(mocker, args):
    mocker.patch("sys.argv", ["docker_run.py"] + args)
    args = docker_run.parse_args()
    assert args.json_file[0] == "file_01.json"
    if args.local_storage:
        assert args.local_storage is True


@patch("docker_run.get_sentiment_predictions")
@patch("docker_run.load_model")
def test_comment_id_error(mock_load_model, mock_get_predictions):
    with pytest.raises(ValueError):
        test_json = [
            {
                "comment_id": "1",
                "comment_text": "I liked all of it",
                "question_type": "nonspecific",
            },
            {
                "comment_id": "1",
                "comment_text": "I liked all of it",
                "question_type": "nonspecific",
            },
        ]
        docker_run.predict_sentiment(test_json)


@patch("docker_run.predict_sentiment", return_value={"text": "ok"})
@patch("docker_run.os.remove")
@patch(
    "builtins.open", new_callable=mock_open, read_data=json.dumps([{"data": "Here"}])
)
@patch("sys.argv", ["docker_run.py"] + ["file_01.json"])
def test_main_not_local(mock_open, mock_remove, mock_predict):
    docker_run.main()
    mock_open.assert_called()
    mock_predict.assert_called()
    mock_remove.assert_called_once()


@patch("docker_run.predict_sentiment", return_value={"text": "ok"})
@patch(
    "builtins.open", new_callable=mock_open, read_data=json.dumps([{"data": "Here"}])
)
@patch("sys.argv", ["docker_run.py"] + ["file_01.json", "-l"])
def test_main_local(mock_open, mock_predict):
    docker_run.main()
    mock_open.assert_called()
    mock_predict.assert_called()
