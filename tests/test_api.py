from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.api import app

client = TestClient(app)


def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"test": "Hello"}


def test_multilabel_predictions():
    test_json = [
        {
            "comment_id": "99999",
            "comment_text": "I liked all of it",
            "question_type": "nonspecific",
        },
        {"comment_id": "A55", "comment_text": "", "question_type": "nonspecific"},
        {
            "comment_id": "A56",
            "comment_text": "Truly awful time finding parking",
            "question_type": "could_improve",
        },
        {
            "comment_id": "4",
            "comment_text": "I really enjoyed the session",
            "question_type": "what_good",
        },
        {"comment_id": "5", "comment_text": "7482367", "question_type": "nonspecific"},
    ]
    response = client.post("/predict_multilabel", json=test_json).json()
    assert len(test_json) == len(response)
    assert isinstance(response[0]["labels"], list)


@patch(
    "api.api.load_sentiment_model",
    AsyncMock(
        return_value=Mock(
            predict=Mock(
                return_value=np.array(
                    [
                        [2.3520987e-02, 6.2770307e-01, 1.3149388e-01],
                        [9.8868138e-01, 1.9990385e-03, 5.4453085e-03],
                        [2.3520987e-02, 6.2770307e-01, 1.3149388e-01],
                        [9.8868138e-01, 1.9990385e-03, 5.4453085e-03],
                    ]
                )
            )
        ),
    ),
)
def test_sentiment_predictions():
    test_json = [
        {
            "comment_id": "99999",
            "comment_text": "I liked all of it",
            "question_type": "nonspecific",
        },
        {"comment_id": "A55", "comment_text": "", "question_type": "nonspecific"},
        {
            "comment_id": "A56",
            "comment_text": "Truly awful time finding parking",
            "question_type": "could_improve",
        },
        {
            "comment_id": "4",
            "comment_text": "I really enjoyed the session",
            "question_type": "what_good",
        },
        {"comment_id": "5", "comment_text": "7482367", "question_type": "nonspecific"},
    ]
    response = client.post("/predict_sentiment", json=test_json).json()
    assert len(test_json) == len(response)
    assert isinstance(response[0]["sentiment"], int) is True


def test_q_type_error():
    test_json = [
        {
            "comment_id": "99999",
            "comment_text": "I liked all of it",
            "question_type": "NOT A QUESTION",
        }
    ]
    response = client.post("/predict_multilabel", json=test_json).json()
    assert response["detail"][0]["type"] == "value_error"


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
        client.post("/predict_multilabel", json=test_json).json()
