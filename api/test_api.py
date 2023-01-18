import requests

"""
To test the API, first in terminal, run this command to launch uvicorn server on http://127.0.0.1:8000
    uvicorn api.api:app --reload
Then you can run this test_api script to check if the API is behaving as it should locally
"""


def test_sql():
    ids = ','.join([str(x) for x in range(1,50)])
    params={"ids": ids, "target": "criticality"}
    response = requests.get("http://127.0.0.1:8000/predict_from_sql", params=params)
    print(response.json())

def test_json():
    json = [{'id': '1', 'feedback': 'What is this all about'},
            {'id': '2', 'feedback': 'Everything about the service was amazing'},
            {'id': '3', 'feedback': 'Food was absolutely disgusting'},
            {'id': '4', 'feedback': 'Everyone is really nice'},
            {'id': '5', 'feedback': 'Loved riding on the rollercoasters'},
            {'id': '6', 'feedback': 'Not sure what to say in this comment'},
            {'id': '7', 'feedback': 'Once upon a time there was an amazing comment'},
            {'id': '8', 'feedback': 'maybe the hospital could consider hiring more horses'},
            {'id': '9', 'feedback': 'Why were all the staff wearing hats'}
            ]
    response = requests.post("http://127.0.0.1:8000/test_json", json=json)
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    test_json()
