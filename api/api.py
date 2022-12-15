from fastapi import FastAPI
from pxtextmining.factories.factory_predict_unlabelled_text import factory_predict_unlabelled_text
import pandas as pd
import mysql.connector


app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Test': 'Hello'}

@app.get('/predict_from_sql')
def predict(ids: str,
            target: str):
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
