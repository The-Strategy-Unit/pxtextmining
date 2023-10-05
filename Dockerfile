FROM python:3.10.13-slim
VOLUME /data

COPY pxtextmining /pxtextmining
COPY pyproject.toml /pyproject.toml
COPY README.md /README.md
RUN pip install --upgrade pip setuptools \
  && pip install . \
  && rm -rf /root/.cache
COPY current_best_model/sentiment/bert_sentiment bert_sentiment
COPY current_best_model/final_bert/bert_multilabel bert_multilabel
COPY current_best_model/final_svc/final_svc.sav /final_svc.sav
COPY current_best_model/final_xgb/final_xgb.sav /final_xgb.sav
COPY --chmod=755 docker_run.py docker_run.py

LABEL org.opencontainers.image.source=https://github.com/cdu-data-science-team/pxtextmining

ENTRYPOINT ["python3", "docker_run.py"]
