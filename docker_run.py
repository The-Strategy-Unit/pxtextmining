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
    return_dict = merged[["comment_id", "comment_text", "sentiment"]].to_dict(
        orient="records"
    )
    return return_dict


def main():
    with open("test_json.json", "r") as json_file:
        json_in = json.load(json_file)
        predictions = predict_sentiment(json_in)
        print(predictions)


if __name__ == "__main__":
    main()
