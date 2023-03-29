from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)

def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Test": "Hello"}

def test_predictions():
    test_json = [{'comment_id': '99999', 'comment_text': 'I liked all of it'}, {'comment_id': 'A55', 'comment_text': ""}, {'comment_id': 'A56', 'comment_text': 'All was good'}, {'comment_id': '4', 'comment_text': 'I really enjoyed the session'}, {'comment_id': '5', 'comment_text': '7482367'}]
    response = client.post("/predict_multilabel", json=test_json).json()
    total_responses = 0
    for v in response.values():
        total_responses += len(v)
    assert len(test_json) == total_responses
