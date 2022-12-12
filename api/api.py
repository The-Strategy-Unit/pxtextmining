from fastapi import FastAPI
from pxtextmining.factories.factory_predict_unlabelled_text import factory_predict_unlabelled_text
from pxtextmining.factories.factory_data_load_and_split import process_data
import pandas as pd

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict')
def predict(data, prediction_target):
    """
    predict unlabelled text. data should be received in JSON format with the
    column containing the text labelled "feedback").
    prediction_target can either be 'label' or 'criticality'
    """
    data_df = pd.DataFrame.read_json(data)
    if prediction_target == 'label':
        predictions = factory_predict_unlabelled_text(dataset = data_df,
                                                  predictor = "feedback",
                                                  pipe_path_or_object="results_label/pipeline_label.sav"
                                                  )
    elif prediction_target == 'criticality':
        predictions = factory_predict_unlabelled_text(dataset = data_df,
                                                  predictor = "feedback",
                                                  pipe_path_or_object="results_criticality/pipeline_criticality.sav"
                                                  )

    return predictions.to_dict(orient="records")
