FROM python:3.10.12-bookworm
VOLUME /data

LABEL org.opencontainers.image.source=https://github.com/cdu-data-science-team/pxtextmining

COPY docker-requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache

COPY api/bert_sentiment bert_sentiment
COPY docker_run.py docker_run.py
RUN chmod +x ./docker_run.py

ENTRYPOINT ["python3", "docker_run.py"]