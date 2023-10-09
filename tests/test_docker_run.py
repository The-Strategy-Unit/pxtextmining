import json
import random
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

import docker_run
from pxtextmining.params import minor_cats


@pytest.fixture
def sentiment_output():
    s_preds = pd.DataFrame(
        [
            {"comment_id": "1", "sentiment": 1},
            {"comment_id": "2", "sentiment": "Labelling not possible"},
        ]
    )
    return s_preds


@pytest.fixture
def multilabel_output():
    m_preds = pd.DataFrame(
        [
            {"comment_id": "1", "labels": ["Positive experience & gratitude"]},
            {"comment_id": "2", "labels": ["Labelling not possible"]},
        ]
    )
    return m_preds


@pytest.fixture
def output_df():
    indices = ["1"]
    df_list = []
    for _ in range(len(indices)):
        data_dict = {}
        for cat in minor_cats:
            data_dict[cat] = random.randint(0, 1)
            key = f"Probability of '{cat}'"
            data_dict[key] = random.uniform(0.0, 0.99)
        df_list.append(data_dict)
    df = pd.DataFrame(df_list)
    df.index = indices
    assert len(df.columns) == 64
    return df


@pytest.fixture
def input_data():
    input_text = [
        {
            "comment_id": "1",
            "comment_text": "Nurse was great.",
            "question_type": "what_good",
        },
        {"comment_id": "2", "comment_text": "", "question_type": "could_improve"},
    ]
    return input_text


@patch("docker_run.load_model")
def test_load_bert_model(mock_load):
    docker_run.load_bert_model("bert_sentiment")
    mock_load.assert_called_once()


def test_process_text(input_data):
    df, text_to_predict = docker_run.process_text(input_data)
    assert len(df.columns) == 3
    assert len(text_to_predict.columns) == 2
    assert text_to_predict.index.name == "Comment ID"
    assert df.shape[0] == text_to_predict.shape[0]


@patch("docker_run.pickle.load")
def test_load_sklearn_model(mock_pickle_load):
    docker_run.load_sklearn_model("final_svc")
    mock_pickle_load.assert_called_once()


@patch("docker_run.predict_sentiment_bert")
@patch("docker_run.load_model")
def test_predict_sentiment(mock_load_model, mock_get_predictions, input_data):
    input_text = input_data
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
    return_df = docker_run.predict_sentiment(input_text)
    mock_load_model.assert_called_once()
    mock_get_predictions.assert_called()
    assert len(return_df) == len(input_text)


@patch("docker_run.predict_multilabel_sklearn")
@patch("docker_run.predict_multilabel_bert")
@patch("docker_run.pickle.load")
@patch("docker_run.load_model")
def test_predict_multilabel_ensemble(
    mock_load_model,
    mock_pickle_load,
    mock_predict_bert,
    mock_predict_sklearn,
    output_df,
    input_data,
):
    mock_predict_bert.return_value = output_df
    mock_predict_sklearn.return_value = output_df
    input_text = input_data
    return_df = docker_run.predict_multilabel_ensemble(input_text)
    mock_load_model.assert_called_once()
    mock_pickle_load.assert_called()
    mock_predict_bert.assert_called_once()
    mock_predict_sklearn.assert_called()
    assert len(return_df) == len(input_text)


@pytest.mark.parametrize(
    "args",
    [
        ["file_01.json"],
        ["file_01.json", "-l", "--target", "m"],
        ["file_01.json", "-t", "s"],
    ],
)
def test_parse_args(mocker, args):
    mocker.patch("sys.argv", ["docker_run.py"] + args)
    args = docker_run.parse_args()
    assert args.json_file[0] == "file_01.json"
    if args.local_storage:
        assert args.local_storage is True
    assert args.target in "ms"


def test_comment_id_error():
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
        docker_run.process_text(test_json)


@patch("docker_run.predict_multilabel_ensemble")
@patch("docker_run.predict_sentiment")
@patch("docker_run.os.remove")
@patch(
    "builtins.open", new_callable=mock_open, read_data=json.dumps([{"data": "Here"}])
)
@patch("sys.argv", ["docker_run.py"] + ["file_01.json"])
def test_main_not_local(
    mock_open,
    mock_remove,
    mock_predict_sentiment,
    mock_predict_ensemble,
    sentiment_output,
    multilabel_output,
):
    mock_predict_sentiment.return_value = sentiment_output
    mock_predict_ensemble.return_value = multilabel_output
    docker_run.main()
    mock_open.assert_called()
    mock_predict_sentiment.assert_called()
    mock_predict_ensemble.assert_called()
    mock_remove.assert_called_once()


@patch("docker_run.predict_sentiment")
@patch(
    "builtins.open", new_callable=mock_open, read_data=json.dumps([{"data": "Here"}])
)
@patch("sys.argv", ["docker_run.py"] + ["file_01.json", "-l", "-t", "s"])
def test_main_local(mock_open, mock_predict_sentiment, sentiment_output):
    mock_predict_sentiment.return_value = sentiment_output
    docker_run.main()
    mock_open.assert_called()
    mock_predict_sentiment.assert_called()
