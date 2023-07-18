from unittest.mock import patch

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
def test_predict_sentiment(mock_get_predictions):
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
    return_dict = docker_run.predict_sentiment(input_text)
    assert len(return_dict) == len(input_text)
    assert "sentiment" in return_dict[0].keys()


@pytest.mark.parametrize("args", [["file_01.json"], ["file_01.json", "-l"]])
def test_parse_args(mocker, args):
    mocker.patch("sys.argv", ["docker_run.py"] + args)
    args = docker_run.parse_args()
    assert args.json_file[0] == "file_01.json"
    if args.local_storage:
        assert args.local_storage is True


def test_main():
    # docker_run.main()
    pass
