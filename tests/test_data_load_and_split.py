from pxtextmining.factories import factory_data_load_and_split
from pxtextmining.params import major_cats, minor_cats
import pandas as pd
import numpy as np
import pytest

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

@pytest.fixture
def grab_test_Y():
    Y_feats = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    return Y_feats

def test_clean_empty_features():
    df_with_empty_lines = pd.DataFrame({'text': ['Some text', '', ' ', 'More text']})
    clean_df = factory_data_load_and_split.clean_empty_features(df_with_empty_lines)
    assert clean_df.shape == (2,1)

def test_onehot():
    df_to_onehot = pd.DataFrame({'Categories': ['A', 'B', 'C', 'A', 'A', 'B']})
    df_onehotted = factory_data_load_and_split.onehot(df_to_onehot, 'Categories')
    assert df_onehotted.shape == (6,3)

# df = factory_data_load_and_split.load_multilabel_data(filename = 'datasets/testing/test_data.csv', target = 'major_categories')[:500]
# X_train_val, X_test, Y_train_val, Y_test = factory_data_load_and_split.process_and_split_multilabel_data(df, target = major_cats, preprocess_text = False, additional_features = True)

def test_bert_data_to_dataset_with_Y(grab_test_X_additional_feats, grab_test_Y):
    train_dataset = factory_data_load_and_split.bert_data_to_dataset(grab_test_X_additional_feats, grab_test_Y, additional_features = True)
    assert type(train_dataset._structure) == tuple

def test_bert_data_to_dataset_without_Y(grab_test_X_additional_feats):
    test_dataset = factory_data_load_and_split.bert_data_to_dataset(grab_test_X_additional_feats, Y = None, additional_features = True)
    assert type(test_dataset) == dict
