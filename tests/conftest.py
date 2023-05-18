import pytest
import pandas as pd

@pytest.fixture
def grab_test_X_additional_feats():
    data_dict = {'FFT answer': {'Q1': 'Nurses were great',
                    'Q2': 'Communication was fantastic',
                    'Q3': 'Impossible to find parking, but pleased to get an appointment close to home',
                    'Q4': 'Food and drink selection very limited',
                    'Q5': 'The ward was boiling hot, although staff were great at explaining details'},
                    'FFT_q_standardised': {'Q1': 'what_good',
                    'Q2': 'what_good',
                    'Q3': 'could_improve',
                    'Q4': 'could_improve',
                    'Q5': 'could_improve'}}
    text_X_additional_feats = pd.DataFrame(data_dict)
    return text_X_additional_feats
