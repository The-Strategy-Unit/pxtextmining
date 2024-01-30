import os
import pickle
from typing import List

import pandas as pd
import schemas
from fastapi import FastAPI

from pxtextmining.factories.factory_predict_unlabelled_text import (
    predict_multilabel_sklearn,
)
from pxtextmining.params import minor_cats

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
]


app = FastAPI(
    title="pxtextmining API",
    description=description,
    version="1.0.0",
    contact={
        "name": "Patient Experience Qualitative Data Categorisation",
        "url": "https://the-strategy-unit.github.io/PatientExperience-QDC/",
        "email": "chris.beeley1@nhs.net",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/the-strategy-unit/pxtextmining/blob/main/LICENSE",
    },
    openapi_tags=tags_metadata,
)


@app.get("/", response_model=schemas.Test, tags=["index"])
def index():
    return {"test": "Hello"}


@app.post(
    "/predict_multilabel",
    response_model=List[schemas.MultilabelOut],
    tags=["multilabel"],
)
async def predict_multilabel(items: List[schemas.ItemIn]):
    """Accepts comment ids and comment text as JSON in a POST request. Makes predictions using trained SVC model.

    Args:
        items (List[ItemIn]): JSON list of dictionaries with the following compulsory keys:
        - `comment_id` (str)
        - `comment_text` (str)

    Returns:
        (dict): Keys are: `comment_id` and predicted `labels`.
    """

    # Process received data
    df = pd.DataFrame([i.dict() for i in items], dtype=str)
    df_for_preds = df.copy().rename(
        columns={"comment_id": "Comment ID", "comment_text": "FFT answer"}
    )
    df_for_preds = df_for_preds.set_index("Comment ID")
    if df_for_preds.index.duplicated().sum() != 0:
        raise ValueError("comment_id must all be unique values")
    text_to_predict = df_for_preds["FFT answer"]
    # Make predictions
    model_path = "final_svc.sav"
    if not os.path.isfile(model_path):
        model_path = os.path.join("api", model_path)
    with open(model_path, "rb") as model:
        loaded_model = pickle.load(model)
    preds_df = predict_multilabel_sklearn(
        text_to_predict, loaded_model, labels=minor_cats, additional_features=False
    )
    # Join predicted labels with received data
    preds_df["comment_id"] = preds_df.index.astype(str)
    merged = pd.merge(df, preds_df, how="left", on="comment_id")
    merged["labels"] = merged["labels"].fillna("").apply(list)
    for i in merged["labels"].index:
        label_list = merged.loc[i, "labels"]
        if len(label_list) < 1:
            merged.loc[i, "labels"].append("Labelling not possible")
    return_dict = merged[["comment_id", "labels"]].to_dict(orient="records")
    return return_dict
