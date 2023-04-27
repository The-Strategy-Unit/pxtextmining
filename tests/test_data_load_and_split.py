from pxtextmining.factories import factory_data_load_and_split
from pxtextmining.params import major_cats, minor_cats
import pandas as pd

def test_clean_empty_features():
    df_with_empty_lines = pd.DataFrame({'text': ['Some text', '', ' ', 'More text']})
    clean_df = factory_data_load_and_split.clean_empty_features(df_with_empty_lines)
    assert clean_df.shape == (2,1)

def test_onehot():
    df_to_onehot = pd.DataFrame({'Categories': ['A', 'B', 'C', 'A', 'A', 'B']})
    df_onehotted = factory_data_load_and_split.onehot(df_to_onehot, 'Categories')
    assert df_onehotted.shape == (6,3)

df = factory_data_load_and_split.load_multilabel_data(filename = 'datasets/testing/test_data.csv', target = 'major_categories')[:500]
X_train_val, X_test, Y_train_val, Y_test = factory_data_load_and_split.process_and_split_data(df, target = major_cats, preprocess_text = False, additional_features = True)

def test_bert_data_to_dataset_with_Y(X_train_val = X_train_val, Y_train_val = Y_train_val):
    train_dataset = factory_data_load_and_split.bert_data_to_dataset(X_train_val, Y_train_val, additional_features = True)
    assert type(train_dataset._structure) == tuple

def test_bert_data_to_dataset_without_Y():
    test_dataset = factory_data_load_and_split.bert_data_to_dataset(X_test, Y = None, additional_features = True)
    assert type(test_dataset) == dict
