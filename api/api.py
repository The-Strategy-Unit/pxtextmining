import pickle
from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from pxtextmining.factories.factory_predict_unlabelled_text import \
    predict_multilabel_sklearn


class ItemIn(BaseModel):
    comment_id: str
    comment_text: str

app = FastAPI()

@app.get('/')
def index():
    return {'Test': 'Hello'}

@app.post('/predict_multilabel')
def predict(items: List[ItemIn]):
    """Accepts comment ids and comment text as JSON in a POST request. Makes predictions using

    Args:
        items (List[ItemIn]): JSON list of dictionaries with the following compulsory keys: `comment_id` (str) and `comment_text` (str). For example, `[{'comment_id': '1', 'comment_text': 'Thank you'}, {'comment_id': '2', 'comment_text': 'Food was cold'}]`

    Returns:
        (dict): Dict containing two keys. `comments_labelled` is a list of dictionaries containing the received comment ids, comment text, and predicted labels. `comment_ids_failed` is a list of the comment_ids where the text was unable to be labelled, for example due to being an empty string, or a null value.
    """
    with open('current_best_multilabel/svc_text_only.sav', 'rb') as model:
        loaded_model = pickle.load(model)
    df = pd.DataFrame([i.dict() for i in items], dtype=str)
    df_newindex = df.set_index('comment_id')
    df_newindex.index.rename('Index', inplace = True)
    text_to_predict = df_newindex['comment_text']
    preds_df = predict_multilabel_sklearn(text_to_predict, loaded_model)
    preds_df['comment_id'] = preds_df.index.astype(str)
    merged = pd.merge(df, preds_df, how='left', on='comment_id')
    merged['labels'] = merged['labels'].fillna('').apply(list)
    for i in merged['labels'].index:
        if len(merged['labels'].loc[i]) < 1:
            merged['labels'].loc[i].append('Labelling not possible')
    return_dict = merged[['comment_id', 'comment_text', 'labels']].to_dict(orient='records')
    return return_dict
