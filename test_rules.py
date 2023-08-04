import os
import pickle

from tensorflow.keras.saving import load_model

from pxtextmining.factories.factory_data_load_and_split import (
    load_multilabel_data,
    process_and_split_data,
)
from pxtextmining.factories.factory_model_performance import get_multilabel_metrics
from pxtextmining.factories.factory_write_results import (
    write_model_analysis,
    write_model_preds,
)
from pxtextmining.params import minor_cats, random_state


def test_rules():
    # load sklearn model
    model_path = "test_multilabel/v6_230724/svc/model_0.sav"
    with open(model_path, "rb") as model:
        loaded_model = pickle.load(model)
    path = "test_multilabel/v6_230724/svc/rules"
    additional_features = True
    target = minor_cats
    target_name = "minor_categories"
    df = load_multilabel_data(
        filename="datasets/hidden/multilabel_230719.csv", target=target_name
    )
    X_train, X_test, Y_train, Y_test = process_and_split_data(
        df,
        target=target,
        additional_features=additional_features,
        random_state=random_state,
    )
    training_time = 0
    model_metrics = get_multilabel_metrics(
        X_test,
        Y_test,
        random_state=random_state,
        labels=target,
        model_type="sklearn",
        model=loaded_model,
        training_time=training_time,
        additional_features=additional_features,
        enhance_with_rules=True,
    )
    txtpath = os.path.join(path, "model_0" + ".txt")
    with open(txtpath, "w") as file:
        file.write(model_metrics)
    write_model_preds(
        X_test,
        Y_test,
        loaded_model,
        labels=target,
        additional_features=additional_features,
        path=f"{path}/labels.xlsx",
    )
    write_model_analysis(model_name="model_0", labels=target, dataset=df, path=path)


def test_rules_bert():
    # bert model
    model_path = "test_multilabel/v6_230724/bert/model_0"
    loaded_model = load_model(model_path)
    path = "test_multilabel/v6_230724/bert/rules"
    additional_features = True
    target = minor_cats
    target_name = "minor_categories"
    df = load_multilabel_data(
        filename="datasets/hidden/multilabel_230719.csv", target=target_name
    )
    X_train_val, X_test, Y_train_val, Y_test = process_and_split_data(
        df,
        target=target,
        preprocess_text=False,
        additional_features=additional_features,
        random_state=random_state,
    )
    training_time = 0
    model_metrics = get_multilabel_metrics(
        X_test,
        Y_test,
        random_state=random_state,
        labels=target,
        model_type="bert",
        model=loaded_model,
        training_time=training_time,
        additional_features=additional_features,
        already_encoded=False,
        enhance_with_rules=False,
    )
    txtpath = os.path.join(path, "model_0" + ".txt")
    with open(txtpath, "w") as file:
        file.write(model_metrics)
    model_metrics = get_multilabel_metrics(
        X_test,
        Y_test,
        random_state=random_state,
        labels=target,
        model_type="bert",
        model=loaded_model,
        training_time=training_time,
        additional_features=additional_features,
        already_encoded=False,
        enhance_with_rules=True,
    )
    txtpath = os.path.join(path, "model_0_rules" + ".txt")
    with open(txtpath, "w") as file:
        file.write(model_metrics)
    write_model_preds(
        X_test,
        Y_test,
        loaded_model,
        labels=target,
        additional_features=additional_features,
        path=f"{path}/labels.xlsx",
    )
    write_model_analysis(model_name="model_0", labels=target, dataset=df, path=path)


if __name__ == "__main__":
    test_rules()
    test_rules_bert()
