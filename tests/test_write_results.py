import os
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
from tensorflow.keras import Model

from pxtextmining.factories import factory_write_results


@patch("pickle.dump", Mock())
@patch("builtins.open", new_callable=mock_open, read_data="somestr")
def test_write_multilabel_models_and_metrics(mock_file, tmp_path_factory):
    # arrange
    mock_model = Mock(spec=Model)
    models = [mock_model]
    model_metrics = ["somestr"]
    path = tmp_path_factory.mktemp("somepath")
    # act
    factory_write_results.write_multilabel_models_and_metrics(
        models, model_metrics, path
    )
    # assert
    mock_model.save.assert_called_once()
    mock_file.assert_called_with(os.path.join(path, "model_0.txt"), "w")
    assert open(os.path.join("somepath", "model_0.txt")).read() == "somestr"


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
    factory_write_results.write_model_preds(x, y_true, preds_df, labels, path=path)
    # assert
    mock_toexcel.assert_called()
