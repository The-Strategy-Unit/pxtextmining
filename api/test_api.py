import requests

ids = ','.join([str(x) for x in range(1,100)])
params={"ids": ids, "target": "label"}
response = requests.get("http://127.0.0.1:8000/predict_from_sql", params=params)
print(response.json())
