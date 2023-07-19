FROM python:3.10.12-slim-bookworm
VOLUME /data

COPY docker-requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
  && pip install -r requirements.txt \
  && rm -rf /root/.cache

COPY api/bert_sentiment bert_sentiment
COPY --chmod=755 docker_run.py docker_run.py

LABEL org.opencontainers.image.source=https://github.com/cdu-data-science-team/pxtextmining

ENTRYPOINT ["python3", "docker_run.py"]
