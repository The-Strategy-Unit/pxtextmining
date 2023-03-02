from fastapi import FastAPI
from pxtextmining.factories.factory_predict_unlabelled_text import factory_predict_unlabelled_text, predict_multilabel_sklearn
import pandas as pd
import mysql.connector
from typing import Union, List
from pydantic import BaseModel
import pickle

class ItemIn(BaseModel):
    comment_id: str
    comment_text: str

class TestJson(BaseModel):
    id: str
    feedback: str

app = FastAPI()

@app.get('/')
def index():
    return {'Test': 'Hello'}

@app.post('/predict_multilabel')
def predict(items: List[ItemIn]):
    loaded_model = pickle.load(open('current_best_multilabel/model_0.sav', 'rb'))
    df = pd.DataFrame([i.dict() for i in items])
    df['comment_id'] = df['comment_id'].astype(int)
    text_to_predict = df['comment_text']
    preds_df = predict_multilabel_sklearn(text_to_predict, loaded_model)
    ## Is it necessary to merge with original df? maybe to check they still line up?
    preds_df['comment_id'] = preds_df.index.astype(int)
    merged = pd.merge(df, preds_df, how='inner', on='comment_id')
    return merged[['comment_id', 'labels']].to_dict(orient='records')
    # return preds_df.to_dict(orient='records')

@app.post('/test_json', response_model=List[TestJson])
def accept(items: List[ItemIn]):
    return [i.dict() for i in items]

@app.post('/predict_from_json')
def predict(items: List[ItemIn]):
    df = pd.DataFrame([i.dict() for i in items])
    model = 'results_label/pipeline_label.sav'
    text_data = df.rename(columns = {'comment_text': 'predictor'})
    predictions = factory_predict_unlabelled_text(dataset=text_data, predictor="predictor",
                                                    theme = 'label', pipe_path_or_object=model,
                                                    columns_to_return='all_cols')
    return predictions.to_dict(orient='records')
