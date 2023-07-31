import pickle

from pxtextmining.factories.factory_data_load_and_split import (
    load_multilabel_data,
    process_and_split_data,
)
from pxtextmining.factories.factory_model_performance import get_multilabel_metrics
from pxtextmining.factories.factory_write_results import (
    write_model_analysis,
    write_model_preds,
    write_multilabel_models_and_metrics,
)
from pxtextmining.params import minor_cats, random_state

# set params
model_path = "test_multilabel/v6_230724/svc/model_0.sav"
with open(model_path, "rb") as model:
    loaded_model = pickle.load(model)
path = "test_multilabel/v6_230724/svc/rules"


def test_rules():
    additional_features = True
    target = minor_cats
    target_name = "minor_categories"
    target_name = "test"
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
    write_multilabel_models_and_metrics([loaded_model], [model_metrics], path=path)
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
