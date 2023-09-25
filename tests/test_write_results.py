import os
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from tensorflow.keras import Model

from pxtextmining.factories import factory_write_results


@patch("pxtextmining.factories.factory_write_results.pickle.dump", Mock())
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="somestr",
)
@pytest.mark.parametrize("models", [[Mock(spec=Model)], [Mock(spec=DummyClassifier)]])
def test_write_multilabel_models_and_metrics(mock_file, tmp_path_factory, models):
    # arrange
    models = models
    model_metrics = ["somestr"]
    path = tmp_path_factory.mktemp("somepath")
    # act
    factory_write_results.write_multilabel_models_and_metrics(
        models, model_metrics, path
    )
    # assert
    if isinstance(models[0], Model):
        models[0].save.assert_called_once()
    mock_file.assert_called_with(os.path.join(path, "model_0.txt"), "w")
    assert open(os.path.join("somepath", "model_0.txt")).read() == "somestr"


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="somestr",
)
@patch("pxtextmining.factories.factory_write_results.os.makedirs")
def test_write_multilabel_models_and_metrics_nopath(
    mock_makedirs, mock_file_open, tmp_path
):
    # arrange
    models = [Mock(spec=Model)]
    model_metrics = ["somestr"]
    path = "somepath"
    # act
    factory_write_results.write_multilabel_models_and_metrics(
        models, model_metrics, path
    )
    # assert
    mock_makedirs.assert_called_once_with(path)


@patch("pxtextmining.factories.factory_write_results.pd.DataFrame.to_excel")
def test_write_model_preds_sklearn(mock_toexcel, grab_test_X_additional_feats):
    x = grab_test_X_additional_feats["FFT answer"]
    # arrange
    y_true = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    labels = ["one", "two", "three", "four", "five"]
    probs_labels = ["Probability of " + x for x in labels]
    preds_df = pd.DataFrame(
        np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.1, 0.6, 0.2, 0.7, 0.05],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.55, 0.2, 0.3, 0.8, 0.4],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.3, 0.2, 0.3, 0.1],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.7, 0.2, 0.8, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.4, 0.2, 0.1, 0.6],
            ]
        ),
        columns=labels + probs_labels,
        index=grab_test_X_additional_feats.index,
    )
    preds_df["labels"] = [
        ["two", "four"],
        ["one", "four"],
        ["one"],
        ["one", "three", "four"],
        ["five"],
    ]
    path = "somepath.xlsx"
    # act
    df = factory_write_results.write_model_preds(
        x, y_true, preds_df, labels, path=path, return_df=True
    )
    # assert
    assert df.shape[0] == len(x)
    mock_toexcel.assert_called()


@patch("pxtextmining.factories.factory_write_results.pd.DataFrame.to_excel")
@patch("pxtextmining.factories.factory_write_results.parse_metrics_file")
def test_write_model_analysis(
    mock_parsemetrics,
    mock_toexcel,
    grab_preds_df,
):
    mock_parsemetrics.return_value = pd.DataFrame(
        {
            "label": {0: "one", 1: "two", 2: "three", 3: "four", 4: "five"},
            "precision": {0: 0.46, 1: 0.54, 2: 0.52, 3: 0.54, 4: 0.52},
            "recall": {0: 0.43, 1: 0.82, 2: 0.65, 3: 0.82, 4: 0.65},
            "f1_score": {0: 0.44, 1: 0.65, 2: 0.58, 3: 0.65, 4: 0.58},
            "support (label count in test data)": {
                0: 129,
                1: 115,
                2: 20,
                3: 115,
                4: 20,
            },
        }
    )
    labels = ["one", "two", "three", "four", "five"]
    dataset = grab_preds_df.copy()
    preds_df = grab_preds_df
    y_true = np.array(grab_preds_df[labels])

    factory_write_results.write_model_analysis(
        "model_name",
        labels=labels,
        dataset=dataset,
        path="somepath",
        preds_df=preds_df,
        y_true=y_true,
        custom_threshold_dict=None,
    )
    mock_toexcel.assert_called_once()
