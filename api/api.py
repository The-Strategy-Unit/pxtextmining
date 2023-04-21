import os
import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from pxtextmining.factories.factory_predict_unlabelled_text import \
    predict_multilabel_sklearn

description = """
This API is for classifying patient experience qualitative data,
utilising the models trained as part of the pxtextmining project.
"""

class ItemIn(BaseModel):
    comment_id: str
    comment_text: str
    question_type: str

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "question_type": "nonspecific"
            }
        }

class ItemOut(BaseModel):
    comment_id: str
    comment_text: str
    labels: list

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "labels": ['Staff manner & personal attributes', 'Parking']
            }
        }

app = FastAPI(
    title="pxtextmining API",
    description=description,
    version="0.0.1",
    contact={
        "name": "CDU Data Science",
        "url": "https://cdu-data-science-team.github.io/PatientExperience-QDC/",
        "email": "CDUDataScience@nottshc.nhs.uk",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/CDU-data-science-team/pxtextmining/blob/main/LICENSE",
    }
    )


@app.get("/")
def index():
    return {"Test": "Hello"}


@app.post("/predict_multilabel", response_model=List[ItemOut])
def predict(items: List[ItemIn]):
    """Accepts comment ids, comment text and question type as JSON in a POST request. Makes predictions using trained SVC model.

    Args:
        items (List[ItemIn]): JSON list of dictionaries with the following compulsory keys:
        - `comment_id` (str)
        - `comment_text` (str), and
        - `question_type` (str).
        The 'question_type' must be one of three values: 'nonspecific', 'what_good', and 'could_improve'.
        For example, `[{'comment_id': '1', 'comment_text': 'Thank you', 'question_type': 'what_good'}, {'comment_id': '2', 'comment_text': 'Food was cold', 'question_type': 'could_improve'}]`

    Returns:
        (dict): Keys are: `comment_id`, `comment_text`, and predicted `labels`.
    """

    # Process received data
    df = pd.DataFrame([i.dict() for i in items], dtype=str)
    df_newindex = df.set_index("comment_id")
    if df_newindex.index.duplicated().sum() != 0:
        raise ValueError('comment_id must all be unique values')
    df_newindex.index.rename("Comment ID", inplace=True)
    text_to_predict = df_newindex[["comment_text", "question_type"]]
    text_to_predict = text_to_predict.rename(
        columns={"comment_text": "FFT answer", "question_type": "FFT_q_standardised"}
    )
    # Make predictions
    model_path = os.path.join('current_best_multilabel', 'svc_minorcats_v5.sav')
    with open(model_path, "rb") as model:
        loaded_model = pickle.load(model)
    preds_df = predict_multilabel_sklearn(
        text_to_predict, loaded_model, additional_features=True
    )
    # Join predicted labels with received data
    preds_df["comment_id"] = preds_df.index.astype(str)
    merged = pd.merge(df, preds_df, how="left", on="comment_id")
    merged["labels"] = merged["labels"].fillna("").apply(list)
    for i in merged["labels"].index:
        if len(merged["labels"].loc[i]) < 1:
            merged["labels"].loc[i].append("Labelling not possible")
    return_dict = merged[["comment_id", "comment_text", "labels"]].to_dict(
        orient="records"
    )
    return return_dict
