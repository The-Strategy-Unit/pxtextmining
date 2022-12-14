import pandas as pd
from pxtextmining.factories.factory_predict_unlabelled_text import factory_predict_unlabelled_text

"""
This is an example of how to predict unlabelled text using a pretrained model,
'results_label/pipeline_label.sav', which has been trained to predict categorical labels.
"""

dataset = pd.read_csv('datasets/text_data.csv')
predictions = factory_predict_unlabelled_text(dataset=dataset, predictor="feedback",
                                    pipe_path_or_object="results_label/pipeline_label.sav",
                                    columns_to_return='all_cols')
print(predictions.head())
