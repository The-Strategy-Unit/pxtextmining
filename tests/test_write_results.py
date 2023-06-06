from pxtextmining.factories import factory_write_results
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, mock_open, patch
from tensorflow.keras import Model
import os

@patch("pickle.dump", Mock())
@patch('builtins.open', new_callable=mock_open, read_data='somestr')
def test_write_multilabel_models_and_metrics(mock_file):
    # arrange
    mock_model = Mock(spec=Model)
    models = [mock_model]
    model_metrics = ['somestr']
    path = 'somepath'
    # act
    factory_write_results.write_multilabel_models_and_metrics(models, model_metrics, path)
    # assert
    mock_model.save.assert_called_once()
    mock_file.assert_called_with(os.path.join('somepath', 'model_0.txt'), 'w')
    assert open(os.path.join('somepath', 'model_0.txt')).read() == "somestr"
