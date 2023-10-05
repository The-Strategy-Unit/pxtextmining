import argparse
import json
import os
import pickle

import pandas as pd
from tensorflow.keras.saving import load_model

from pxtextmining.factories.factory_predict_unlabelled_text import (
    combine_predictions,
    predict_multilabel_bert,
    predict_multilabel_sklearn,
    predict_sentiment_bert,
)
from pxtextmining.params import minor_cats


def load_bert_model(model_path):
    if not os.path.exists(f"bert_{model_path}"):
        if model_path == "sentiment":
            model_path = os.path.join(
                "current_best_model", model_path, f"bert_{model_path}"
            )
        elif model_path == "multilabel":
            model_path = os.path.join(
                "current_best_model", "final_bert", f"bert_{model_path}"
            )
    loaded_model = load_model(f"bert_{model_path}")
    return loaded_model


def load_sklearn_model(model_name):
    model_path = f"{model_name}.sav"
    if not os.path.exists(model_name):
        model_path = os.path.join("current_best_model", model_name, model_path)
    with open(model_path, "rb") as model:
        loaded_model = pickle.load(model)
    return loaded_model


def process_text(items):
    df = pd.DataFrame([i for i in items], dtype=str)
    df_newindex = df.set_index("comment_id")
    if df_newindex.index.duplicated().sum() != 0:
        raise ValueError("comment_id must all be unique values")
    df_newindex.index.rename("Comment ID", inplace=True)
    text_to_predict = df_newindex[["comment_text", "question_type"]]
    text_to_predict = text_to_predict.rename(
        columns={"comment_text": "FFT answer", "question_type": "FFT_q_standardised"}
    )
    return df, text_to_predict


def predict_multilabel_ensemble(items):
    # Function which gets preds_dfs for bert, svc, and xgb, and combines them all
    # Process the data
    df, text_to_predict = process_text(items)
    text_to_predict = text_to_predict["FFT answer"]
    # Load models
    bert_model = load_bert_model("multilabel")
    svc_model = load_sklearn_model("final_svc")
    xgb_model = load_sklearn_model("final_xgb")
    # Make preds
    bert_preds = predict_multilabel_bert(
        text_to_predict,
        bert_model,
        labels=minor_cats,
        additional_features=False,
        label_fix=False,
    )
    svc_preds = predict_multilabel_sklearn(
        text_to_predict,
        svc_model,
        labels=minor_cats,
        additional_features=False,
        label_fix=False,
    )
    xgb_preds = predict_multilabel_sklearn(
        text_to_predict,
        xgb_model,
        labels=minor_cats,
        additional_features=False,
        label_fix=False,
    )
    # Combine preds
    preds_list = [bert_preds, svc_preds, xgb_preds]
    combined_preds = combine_predictions(preds_list, labels=minor_cats)
    # Join predicted labels with received data
    combined_preds["comment_id"] = combined_preds.index.astype(str)
    merged = pd.merge(df, combined_preds, how="left", on="comment_id")
    # Fill in anything that got cleaned in preprocessing step
    nulls = merged[merged.labels.isnull()].index
    lnp = pd.Series(
        [["Labelling not possible"]] * len(nulls), index=nulls, dtype=object
    )
    merged.loc[nulls, "labels"] = lnp
    return_df = merged[["comment_id", "labels"]]
    return return_df


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
    df, text_to_predict = process_text(items)
    # Make predictions
    loaded_model = load_bert_model("sentiment")
    preds_df = predict_sentiment_bert(
        text_to_predict, loaded_model, preprocess_text=False, additional_features=True
    )
    # Join predicted labels with received data
    preds_df["comment_id"] = preds_df.index.astype(str)
    merged = pd.merge(df, preds_df, how="left", on="comment_id")
    merged["sentiment"] = merged["sentiment"].fillna("Labelling not possible")
    return_df = merged[["comment_id", "sentiment"]]
    return return_df


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
    parser.add_argument(
        "--target",
        "-t",
        default="ms",
        help="Target of the predictions. m for multilabel, s for sentiment. Defaults to ms for both multilabel and sentiment",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    json_file = os.path.join("data", "data_in", args.json_file[0])
    with open(json_file, "r") as jf:
        json_in = json.load(jf)
    preds_list = []
    if "s" in args.target:
        s_preds = predict_sentiment(json_in)
        preds_list.append(s_preds)
    if "m" in args.target:
        m_preds = predict_multilabel_ensemble(json_in)
        preds_list.append(m_preds)
    if len(preds_list) == 2:
        preds = pd.merge(preds_list[0], preds_list[1], on="comment_id")
    else:
        preds = preds_list[0]
    if not args.local_storage:
        os.remove(json_file)
    json_out = preds.to_dict(orient="records")
    out_path = os.path.join("data", "data_out", args.json_file[0])
    with open(out_path, "w+") as jf:
        json.dump(json_out, jf)


if __name__ == "__main__":
    main()
