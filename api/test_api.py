import requests

"""
To test the API, first in terminal, run this command to launch uvicorn server on http://127.0.0.1:8000
    uvicorn api.api:app --reload
Then you can run this test_api script to check if the API is behaving as it should locally
"""


ids = ','.join([str(x) for x in range(1,500)])
params={"ids": ids, "target": "criticality"}
response = requests.get("http://127.0.0.1:8000/predict_from_sql", params=params)
print(response.json())
