FROM python:3.10.12-bookworm

COPY docker-requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt

COPY api/bert_sentiment bert_sentiment
COPY docker_run.py docker_run.py
COPY test_json.json test_json.json

CMD python docker_run.py
