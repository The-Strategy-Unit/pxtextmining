import requests
import pandas as pd

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

def test_json(json):
    response = requests.post("http://127.0.0.1:8000/test_json", json=json)
    print(response.status_code)
    print(response.json())

def test_json_predictions(json):
    response = requests.post("http://127.0.0.1:8000/predict_multilabel", json=json)
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    df = pd.read_csv('datasets/text_data.csv')
    df = df.loc[101:200][['feedback']]
    js = []
    for i in range(df.shape[0]):
        js.append({'comment_id': str(i), 'comment_text': df.iloc[i]['feedback']})
    print('The JSON that was sent looks like:')
    print(js[:5])
    print('The JSON that is returned is:')
    test_json_predictions(js)
