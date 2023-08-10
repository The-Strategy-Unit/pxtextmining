import argparse
import json
import os

import pandas as pd
from tensorflow.keras.saving import load_model

from pxtextmining.factories.factory_predict_unlabelled_text import (
    predict_sentiment_bert,
)


def load_sentiment_model():
    model_path = "bert_sentiment"
    if not os.path.exists(model_path):
        model_path = os.path.join("api", model_path)
    loaded_model = load_model(model_path)
    return loaded_model


def get_sentiment_predictions(
    text_to_predict, loaded_model, preprocess_text, additional_features
):
    predictions = predict_sentiment_bert(
        text_to_predict,
        loaded_model,
        preprocess_text=preprocess_text,
        additional_features=additional_features,
    )
    return predictions


def predict_sentiment(items):
    """Accepts comment ids, comment text and question type as JSON in a POST request. Makes predictions using trained Tensorflow Keras model.

    Args:
        items (List[ItemIn]): JSON list of dictionaries with the following compulsory keys:
        - `comment_id` (str)
        - `comment_text` (str)
        - `question_type` (str)
        The 'question_type' must be one of three values: 'nonspecific', 'what_good', and 'could_improve'.
        For example, `[{'comment_id': '1', 'comment_text': 'Thank you', 'question_type': 'what_good'},
        {'comment_id': '2', 'comment_text': 'Food was cold', 'question_type': 'could_improve'}]`

    Returns:
        (dict): Keys are: `comment_id`, `comment_text`, and predicted `labels`.
    """

    # Process received data
    loaded_model = load_sentiment_model()
    df = pd.DataFrame([i for i in items], dtype=str)
    df_newindex = df.set_index("comment_id")
    if df_newindex.index.duplicated().sum() != 0:
        raise ValueError("comment_id must all be unique values")
    df_newindex.index.rename("Comment ID", inplace=True)
    text_to_predict = df_newindex[["comment_text", "question_type"]]
    text_to_predict = text_to_predict.rename(
        columns={"comment_text": "FFT answer", "question_type": "FFT_q_standardised"}
    )
    # Make predictions
    preds_df = get_sentiment_predictions(
        text_to_predict, loaded_model, preprocess_text=False, additional_features=True
    )
    # Join predicted labels with received data
    preds_df["comment_id"] = preds_df.index.astype(str)
    merged = pd.merge(df, preds_df, how="left", on="comment_id")
    merged["sentiment"] = merged["sentiment"].fillna("Labelling not possible")
    return_dict = merged[["comment_id", "sentiment"]].to_dict(orient="records")
    return return_dict


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_file",
        nargs=1,
        help="Name of the json file",
    )
    parser.add_argument(
        "--local-storage",
        "-l",
        action="store_true",
        help="Use local storage (instead of Azure)",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    json_file = os.path.join("data", "data_in", args.json_file[0])
    with open(json_file, "r") as jf:
        json_in = json.load(jf)
    if not args.local_storage:
        os.remove(json_file)
    json_out = predict_sentiment(json_in)
    out_path = os.path.join("data", "data_out", args.json_file[0])
    with open(out_path, "w+") as jf:
        json.dump(json_out, jf)


if __name__ == "__main__":
    main()
