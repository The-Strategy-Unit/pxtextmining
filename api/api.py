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
    print('loading model')
    with open('current_best_multilabel/svc_text_only.sav', 'rb') as model:
        loaded_model = pickle.load(model)
    print('processing df')
    df = pd.DataFrame([i.dict() for i in items], dtype=str)
    df_newindex = df.set_index('comment_id')
    df_newindex.index.rename('Index', inplace = True)
    text_to_predict = df_newindex['comment_text']
    print('making predictions')
    preds_df = predict_multilabel_sklearn(text_to_predict, loaded_model)
    preds_df['comment_id'] = preds_df.index.astype(str)
    print('merging predictions with input')
    merged = pd.merge(df, preds_df, how='inner', on='comment_id')
    return_dict = {'comments_labelled': merged[['comment_id', 'comment_text', 'labels']].to_dict(orient='records')}
    print('not_processed step')
    not_processed = [i for i in df['comment_id'] if i not in preds_df['comment_id']]
    return_dict['comment_ids_failed'] = not_processed
    return return_dict
