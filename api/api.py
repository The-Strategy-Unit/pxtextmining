from fastapi import FastAPI, File, UploadFile
from pxtextmining.factories.factory_predict_unlabelled_text import factory_predict_unlabelled_text
from pxtextmining.factories.factory_data_load_and_split import process_data
import pandas as pd
import mysql.connector
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Test': 'Hello'}

@app.get('/predict_from_sql')
def predict(ids: str,
            target: str):
    if target == 'label':
        model = 'results_label/pipeline_label.sav'
    elif target == 'criticality':
        model = 'test_results_criticality/test_pipeline_criticality.sav'
    else:
        return {'error': 'invalid target'}
    q = ids.split(',')
    placeholders= ', '.join(['%s']*len(q))  # "%s, %s, %s, ... %s"
    query = "SELECT id , feedback FROM text_data WHERE id IN ({})".format(placeholders)
    db = mysql.connector.connect(option_files="my.conf", use_pure=True)
    with db.cursor() as cursor:
        cursor.execute(query, tuple(q))
        text_data = cursor.fetchall()
        text_data = pd.DataFrame(text_data)
        text_data.columns = cursor.column_names
    if target == 'label':
        predictions = factory_predict_unlabelled_text(dataset=text_data, predictor="feedback",
                                    pipe_path_or_object=model)
    elif target == 'criticality':
        # text_data = text_data.rename(columns = {'feedback': 'predictor'})
        # data_processed = process_data(text_data)
        # predictions = factory_predict_unlabelled_text(dataset=data_processed, predictor="predictor",
        #                             pipe_path_or_object=model)
        return {'error': 'not complete'}
    return predictions.to_dict(orient='records')
