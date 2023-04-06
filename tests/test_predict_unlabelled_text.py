from pxtextmining.factories import factory_predict_unlabelled_text
import pandas as pd
import numpy as np

def test_get_probabilities():
    label_series = pd.Series([['label_one'],['label_two', 'label_three']], name='test')
    labels = ['label_one', 'label_two', 'label_three']
    predicted_probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.7]])
    model_type = 'bert'
    test_probability_s = factory_predict_unlabelled_text.get_probabilities(label_series,
                                                                           labels, predicted_probabilities,
                                                                           model_type)
    assert len(test_probability_s.iloc[0]) == 1
    assert test_probability_s.iloc[1]['label_two'] == 0.9
