import os
import pickle
from typing import List, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, validator
from tensorflow.keras.saving import load_model

from pxtextmining.factories.factory_predict_unlabelled_text import (
    predict_multilabel_sklearn,
    predict_sentiment_bert,
)

minor_cats_v5 = [
    "Gratitude/ good experience",
    "Negative experience",
    "Not assigned",
    "Organisation & efficiency",
    "Funding & use of financial resources",
    "Non-specific praise for staff",
    "Non-specific dissatisfaction with staff",
    "Staff manner & personal attributes",
    "Number & deployment of staff",
    "Staff responsiveness",
    "Staff continuity",
    "Competence & training",
    "Unspecified communication",
    "Staff listening, understanding & involving patients",
    "Information directly from staff during care",
    "Information provision & guidance",
    "Being kept informed, clarity & consistency of information",
    "Service involvement with family/ carers",
    "Patient contact with family/ carers",
    "Contacting services",
    "Appointment arrangements",
    "Appointment method",
    "Timeliness of care",
    "Pain management",
    "Diagnosis & triage",
    "Referals & continuity of care",
    "Length of stay/ duration of care",
    "Discharge",
    "Care plans",
    "Patient records",
    "Links with non-NHS organisations",
    "Cleanliness, tidiness & infection control",
    "Safety & security",
    "Provision of medical equipment",
    "Service location",
    "Transport to/ from services",
    "Parking",
    "Electronic entertainment",
    "Feeling safe",
    "Patient appearance & grooming",
    "Mental Health Act",
    "Equality, Diversity & Inclusion",
    "Admission",
    "Collecting patients feedback",
    "Labelling not possible",
    "Environment & Facilities",
    "Supplying & understanding medication",
    "Activities & access to fresh air",
    "Food & drink provision & facilities",
    "Sensory experience",
    "Impact of treatment/ care",
]

description = """
This API is for classifying patient experience qualitative data,
utilising the models trained as part of the pxtextmining project.
"""

tags_metadata = [
    {"name": "index", "description": "Basic page to test if API is working."},
    {
        "name": "multilabel",
        "description": "Generate multilabel predictions for given text.",
    },
    {
        "name": "sentiment",
        "description": "Generate predicted sentiment for given text.",
    },
]


class Test(BaseModel):
    test: str

    class Config:
        schema_extra = {"example": {"test": "Hello"}}


class ItemIn(BaseModel):
    comment_id: str
    comment_text: str
    question_type: str

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "question_type": "nonspecific",
            }
        }

    @validator("question_type")
    def question_type_validation(cls, v):
        if v not in ["what_good", "could_improve", "nonspecific"]:
            raise ValueError(
                "question_type must be one of what_good, could_improve, or nonspecific"
            )
        return v


class MultilabelOut(BaseModel):
    comment_id: str
    comment_text: str
    labels: list

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "labels": ["Staff manner & personal attributes", "Parking"],
            }
        }


class SentimentOut(BaseModel):
    comment_id: str
    comment_text: str
    sentiment: Union[int, str]

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "sentiment": 3,
            }
        }


app = FastAPI(
    title="pxtextmining API",
    description=description,
    version="0.0.1",
    contact={
        "name": "Patient Experience Qualitative Data Categorisation",
        "url": "https://cdu-data-science-team.github.io/PatientExperience-QDC/",
        "email": "CDUDataScience@nottshc.nhs.uk",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/CDU-data-science-team/pxtextmining/blob/main/LICENSE",
    },
    openapi_tags=tags_metadata,
)


@app.get("/", response_model=Test, tags=["index"])
def index():
    return {"test": "Hello"}


@app.post(
    "/predict_multilabel", response_model=List[MultilabelOut], tags=["multilabel"]
)
def predict_multilabel(items: List[ItemIn]):
    """Accepts comment ids, comment text and question type as JSON in a POST request. Makes predictions using trained SVC model.

    Args:
        items (List[ItemIn]): JSON list of dictionaries with the following compulsory keys:
        - `comment_id` (str)
        - `comment_text` (str)
        - `question_type` (str)
        The 'question_type' must be one of three values: 'nonspecific', 'what_good', and 'could_improve'.
        For example, `[{'comment_id': '1', 'comment_text': 'Thank you', 'question_type': 'what_good'},
        {'comment_id': '2', 'comment_text': 'Food was cold', 'question_type': 'could_improve'}]`

    Returns:
        (dict): Keys are: `comment_id`, `comment_text`, and predicted `labels`.
    """

    # Process received data
    df = pd.DataFrame([i.dict() for i in items], dtype=str)
    df_newindex = df.set_index("comment_id")
    if df_newindex.index.duplicated().sum() != 0:
        raise ValueError("comment_id must all be unique values")
    df_newindex.index.rename("Comment ID", inplace=True)
    text_to_predict = df_newindex[["comment_text", "question_type"]]
    text_to_predict = text_to_predict.rename(
        columns={"comment_text": "FFT answer", "question_type": "FFT_q_standardised"}
    )
    # Make predictions
    model_path = "svc_minorcats_v5.sav"
    if not os.path.isfile(model_path):
        model_path = os.path.join("api", model_path)
    with open(model_path, "rb") as model:
        loaded_model = pickle.load(model)
    preds_df = predict_multilabel_sklearn(
        text_to_predict, loaded_model, labels=minor_cats_v5, additional_features=True
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


@app.post("/predict_sentiment", response_model=List[SentimentOut], tags=["sentiment"])
def predict_sentiment(items: List[ItemIn]):
    """Accepts comment ids, comment text and question type as JSON in a POST request. Makes predictions using trained Tensorflow Keras model.

    Args:
        items (List[ItemIn]): JSON list of dictionaries with the following compulsory keys:
        - `comment_id` (str)
        - `comment_text` (str)
        - `question_type` (str)
        The 'question_type' must be one of three values: 'nonspecific', 'what_good', and 'could_improve'.
        For example, `[{'comment_id': '1', 'comment_text': 'Thank you', 'question_type': 'what_good'},
        {'comment_id': '2', 'comment_text': 'Food was cold', 'question_type': 'could_improve'}]`

    Returns:
        (dict): Keys are: `comment_id`, `comment_text`, and predicted `labels`.
    """

    # Process received data
    df = pd.DataFrame([i.dict() for i in items], dtype=str)
    df_newindex = df.set_index("comment_id")
    if df_newindex.index.duplicated().sum() != 0:
        raise ValueError("comment_id must all be unique values")
    df_newindex.index.rename("Comment ID", inplace=True)
    text_to_predict = df_newindex[["comment_text", "question_type"]]
    text_to_predict = text_to_predict.rename(
        columns={"comment_text": "FFT answer", "question_type": "FFT_q_standardised"}
    )
    print(text_to_predict)
    # Make predictions
    model_path = "bert_sentiment"
    if not os.path.exists(model_path):
        model_path = os.path.join("api", model_path)
    loaded_model = load_model(model_path)
    preds_df = predict_sentiment_bert(
        text_to_predict, loaded_model, preprocess_text=True, additional_features=True
    )
    # Join predicted labels with received data
    preds_df["comment_id"] = preds_df.index.astype(str)
    merged = pd.merge(df, preds_df, how="left", on="comment_id")
    merged["sentiment"] = merged["sentiment"].fillna("Labelling not possible")
    return_dict = merged[["comment_id", "comment_text", "sentiment"]].to_dict(
        orient="records"
    )
    return return_dict
