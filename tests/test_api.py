from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)


def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"test": "Hello"}


def test_predictions():
    test_json = [
        {"comment_id": "99999", "comment_text": "I liked all of it", 'question_type': 'nonspecific'},
        {"comment_id": "A55", "comment_text": "", 'question_type': 'nonspecific'},
        {"comment_id": "A56", "comment_text": "Truly awful time finding parking", 'question_type': 'could_improve'},
        {"comment_id": "4", "comment_text": "I really enjoyed the session", 'question_type': 'what_good'},
        {"comment_id": "5", "comment_text": "7482367", 'question_type': 'nonspecific'},
    ]
    response = client.post("/predict_multilabel", json=test_json).json()
    assert len(test_json) == len(response)
