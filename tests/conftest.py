import random
import string
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pxtextmining.params import minor_cats, q_map


@pytest.fixture
def grab_test_X_additional_feats():
    data_dict = {
        "FFT answer": {
            "Q1": "Nurses were great",
            "Q2": "Communication was fantastic",
            "Q3": "Impossible to find parking, but pleased to get an appointment close to home",
            "Q4": "Food and drink selection very limited",
            "Q5": "The ward was boiling hot, although staff were great at explaining details",
        },
        "FFT_q_standardised": {
            "Q1": "what_good",
            "Q2": "what_good",
            "Q3": "could_improve",
            "Q4": "could_improve",
            "Q5": "could_improve",
        },
    }
    text_X_additional_feats = pd.DataFrame(data_dict)
    text_X_additional_feats.index.name = "Comment ID"
    return text_X_additional_feats


@pytest.fixture
def mock_read_csv(mocker, test_raw_data):
    mock = Mock()
    mocker.patch("pandas.read_csv", return_value=test_raw_data)
    return mock


@pytest.fixture
def test_raw_data():
    cols = [
        "Comment ID",
        "Trust",
        "Respondent ID",
        "Date",
        "Service Type 1",
        "Service type 2",
        "FFT categorical answer",
        "FFT question",
        "FFT answer",
        "Comment sentiment",
    ]
    cols.extend(minor_cats)
    data_dict = {}
    for col in cols:
        row = []
        if col not in minor_cats:
            if col in ["FFT categorical answer", "Comment sentiment"]:
                for _i in range(5):
                    row.append(random.randint(1, 5))
            elif col == "FFT question":
                for _i in range(5):
                    row.append(random.choice(list(q_map.keys())))
            else:
                for _i in range(5):
                    row.append(
                        "".join(
                            random.choices(string.ascii_uppercase + string.digits, k=5)
                        )
                    )
        else:
            for _i in range(5):
                row.append(random.choice([np.NaN, 1]))
        data_dict[col] = row
    data = pd.DataFrame(data_dict)
    return data


@pytest.fixture
def grab_preds_df():
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
    )
    preds_df["labels"] = [
        ["two", "four"],
        ["one", "four"],
        ["one"],
        ["one", "three", "four"],
        ["five"],
    ]
    return preds_df
