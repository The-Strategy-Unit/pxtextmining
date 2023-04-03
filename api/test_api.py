import requests
import pandas as pd
import json

"""
To test the API, first in terminal, run this command to launch uvicorn server on http://127.0.0.1:8000
    uvicorn api.api:app --reload
Then you can run this test_api script to check if the API is behaving as it should locally
"""


def test_json_predictions(json):
    response = requests.post("http://127.0.0.1:8000/predict_multilabel", json=json)
    return response

if __name__ == "__main__":
    df = pd.read_csv('datasets/hidden/API_test.csv')
    df = df[['row_id', 'comment_txt']].copy().set_index('row_id')[:20]
    js = []
    for i in df.index:
        js.append({'comment_id': str(i), 'comment_text': df.loc[i]['comment_txt']})
    print('The JSON that was sent looks like:')
    print(js[:5])
    print('The JSON that is returned is:')
    returned_json = test_json_predictions(js).json()
    print(returned_json)
    # json_object = json.dumps(returned_json, indent=4)
    # with open("predictions.json", "w") as outfile:
    #     outfile.write(json_object)
