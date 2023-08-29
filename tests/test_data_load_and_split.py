import random

import numpy as np
import pandas as pd
import pytest

from pxtextmining.factories import factory_data_load_and_split
from pxtextmining.params import minor_cats


@pytest.fixture
def grab_test_Y():
    Y_feats = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    return Y_feats


@pytest.mark.parametrize(
    "target", ["major_categories", "minor_categories", "sentiment"]
)
def test_load_multilabel_data(mock_read_csv, target):
    filename = "None"
    df = factory_data_load_and_split.load_multilabel_data(filename, target)
    assert type(df) == pd.DataFrame


def test_merge_categories():
    test_df = pd.DataFrame(
        {"col_1": [0, 0, 0, 0, 1], "col_2": [0, 1, 0, 0, 1], "col_3": [1, 0, 0, 0, 0]}
    )
    new_cat = "new_cat"
    cats_to_merge = ["col_1", "col_2"]
    merged_df = factory_data_load_and_split.merge_categories(
        test_df, new_cat, cats_to_merge
    )
    assert list(merged_df.columns) == ["col_3", "new_cat"]
    assert merged_df["new_cat"].sum() == 2


def test_remove_punc_and_nums():
    text = "Here is.some TEXT?!?!?! 12345 :)"
    cleaned_text = factory_data_load_and_split.remove_punc_and_nums(text)
    assert cleaned_text == "here is some text"


def test_clean_empty_features():
    df_with_empty_lines = pd.DataFrame({"text": ["Some text", "", " ", "More text"]})
    clean_df = factory_data_load_and_split.clean_empty_features(df_with_empty_lines)
    assert clean_df.shape == (2, 1)


def test_onehot():
    df_to_onehot = pd.DataFrame({"Categories": ["A", "B", "C", "A", "A", "B"]})
    df_onehotted = factory_data_load_and_split.onehot(df_to_onehot, "Categories")
    assert df_onehotted.shape == (6, 3)


def test_bert_data_to_dataset_with_Y(grab_test_X_additional_feats, grab_test_Y):
    train_dataset = factory_data_load_and_split.bert_data_to_dataset(
        grab_test_X_additional_feats, grab_test_Y, additional_features=True
    )
    assert isinstance(train_dataset._structure, tuple)


def test_bert_data_to_dataset_without_Y(grab_test_X_additional_feats):
    test_dataset = factory_data_load_and_split.bert_data_to_dataset(
        grab_test_X_additional_feats, Y=None, additional_features=True
    )
    assert isinstance(test_dataset, dict)


@pytest.mark.parametrize("target", [minor_cats, "sentiment"])
@pytest.mark.parametrize("additional_features", [True, False])
@pytest.mark.parametrize("preprocess_text", [True, False])
def test_process_data(
    grab_test_X_additional_feats, target, preprocess_text, additional_features
):
    df = grab_test_X_additional_feats
    df["Comment sentiment"] = 0
    df[minor_cats] = 0
    for i in range(df.shape[0]):
        df.loc[i, "Comment sentiment"] = random.randint(1, 5)
        for cat in minor_cats:
            df.loc[i, cat] = random.randint(0, 1)
    X, Y = factory_data_load_and_split.process_data(
        df, target, preprocess_text, additional_features
    )
    assert X.shape[0] == Y.shape[0]
