FROM python:3.9.17-alpine3.18

COPY docker-requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY api/bert_sentiment bert_sentiment
COPY docker_run.py docker_run.py
COPY test_json.json test_json.json

ENTRYPOINT [ "./docker_run.py" ]
