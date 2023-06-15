import requests
import pandas as pd
import time

"""
To test the API, first in terminal, run this command to launch uvicorn server on http://127.0.0.1:8000
    uvicorn api.api:app --reload
Then you can run this test_api script to check if the API is behaving as it should locally
"""


def test_json_predictions(json):
    # response = requests.post("http://127.0.0.1:8000/predict_multilabel", json=json)
    response = requests.post("http://127.0.0.1:8000/predict_sentiment", json=json)
    return response


if __name__ == "__main__":
    start = time.time()
    df = pd.read_csv("datasets/hidden/merged_230612.csv")[["Comment ID", "FFT answer"]][
        :2000
    ]
    df = df.rename(
        columns={"Comment ID": "row_id", "FFT answer": "comment_txt"}
    ).dropna()
    df = df[["row_id", "comment_txt"]].copy().set_index("row_id")[:1000]
    js = []
    for i in df.index:
        js.append(
            {
                "comment_id": str(i),
                "comment_text": df.loc[i]["comment_txt"],
                "question_type": "nonspecific",
            }
        )
    print("The JSON that was sent looks like:")
    print(js[:5])
    print("The JSON that is returned is:")
    returned_json = test_json_predictions(js).json()
    finish = time.time()
    total = finish - start
    print(f"Time taken: {total} seconds")
    print(returned_json[:10])
    # json_object = json.dumps(returned_json, indent=4)
    # with open("predictions.json", "w") as outfile:
    #     outfile.write(json_object)
