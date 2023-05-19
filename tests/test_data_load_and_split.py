from pxtextmining.factories import factory_data_load_and_split
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def grab_test_Y():
    Y_feats = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    return Y_feats

def test_remove_punc_and_nums():
    text = 'Here is.some TEXT?!?!?! 12345 :)'
    cleaned_text = factory_data_load_and_split.remove_punc_and_nums(text)
    assert cleaned_text == 'here is some text'

def test_clean_empty_features():
    df_with_empty_lines = pd.DataFrame({'text': ['Some text', '', ' ', 'More text']})
    clean_df = factory_data_load_and_split.clean_empty_features(df_with_empty_lines)
    assert clean_df.shape == (2,1)

def test_onehot():
    df_to_onehot = pd.DataFrame({'Categories': ['A', 'B', 'C', 'A', 'A', 'B']})
    df_onehotted = factory_data_load_and_split.onehot(df_to_onehot, 'Categories')
    assert df_onehotted.shape == (6,3)

def test_bert_data_to_dataset_with_Y(grab_test_X_additional_feats, grab_test_Y):
    train_dataset = factory_data_load_and_split.bert_data_to_dataset(grab_test_X_additional_feats, grab_test_Y, additional_features = True)
    assert type(train_dataset._structure) == tuple

def test_bert_data_to_dataset_without_Y(grab_test_X_additional_feats):
    test_dataset = factory_data_load_and_split.bert_data_to_dataset(grab_test_X_additional_feats, Y = None, additional_features = True)
    assert type(test_dataset) == dict
