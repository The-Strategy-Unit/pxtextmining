from fastapi import FastAPI
from pxtextmining.factories.factory_predict_unlabelled_text import factory_predict_unlabelled_text
import pandas as pd
import mysql.connector
from typing import Union, List
from pydantic import BaseModel

class ItemIn(BaseModel):
    id: str
    comment_text: str

class TestJson(BaseModel):
    id: str
    feedback: str

app = FastAPI()

@app.get('/')
def index():
    return {'Test': 'Hello'}

@app.get('/predict_from_sql')
def predict(ids: str,
            target: str):
    """
    This function creates an SQL query string based on the 'ids' param. It then obtains the text data from the SQL
    database using an SQL connector. This will need to be configured in a my.conf file. The text data is then converted
    to a pandas dataframe and this is used to formulate the prediction using the model of choice. The model used to generate
    predictions is dependent on the 'target' parameter.

    Args:
        ids (str): ids of the text data to be used for predictions, in the format '1,2,3,4,5,6'. Can take up to 5000 ids
        target (str): type of prediction to be chosen. Can either be 'label' or 'criticality'.

    Returns:
        list: List of the predictions, each in dictionary format, containing 'id' and 'predictions'. e.g.
        [{'id': 1, 'predictions': '3'}, {'id': 2, 'predictions': '1'}]. This is converted to JSON by FastAPI.
    """
    q = ids.split(',')
    placeholders= ', '.join(['%s']*len(q))  # "%s, %s, %s, ... %s"
    if target == 'label':
        model = 'results_label/pipeline_label.sav'
        query = "SELECT id , feedback FROM text_data WHERE id IN ({})".format(placeholders)
    elif target == 'criticality':
        model = 'results_criticality_with_theme/pipeline_criticality_with_theme.sav'
        query = "SELECT id , label , feedback FROM text_data WHERE id IN ({})".format(placeholders)
    else:
        return {'error': 'invalid target'}

    db = mysql.connector.connect(option_files="my.conf", use_pure=True)
    with db.cursor() as cursor:
        cursor.execute(query, tuple(q))
        text_data = cursor.fetchall()
        text_data = pd.DataFrame(text_data)
        text_data.columns = cursor.column_names
    if target == 'label':
        predictions = factory_predict_unlabelled_text(dataset=text_data, predictor="feedback",
                                    pipe_path_or_object=model, columns_to_return=['id', 'predictions'])
    elif target == 'criticality':
        text_data = text_data.rename(columns = {'feedback': 'predictor'})
        predictions = factory_predict_unlabelled_text(dataset=text_data, predictor="predictor",
                                                    theme = 'label', pipe_path_or_object=model,
                                                    columns_to_return=['id', 'predictions'])
    return predictions.to_dict(orient='records')

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
