
import random

from pxtextmining.factories.factory_data_load_and_split import (
    load_multilabel_data, process_and_split_data)
from pxtextmining.factories.factory_model_performance import get_multiclass_metrics
from pxtextmining.factories.factory_pipeline import search_sklearn_pipelines
from pxtextmining.factories.factory_write_results import (
    write_multilabel_models_and_metrics)
from pxtextmining.params import dataset


def run_sentiment_pipeline(
    additional_features=False,
    models_to_try=["svm", "xgb"],
    path="test_multilabel/sentiment",
):
    target_names = ["very positive", "positive", "neutral", "negative", "very negative"]
    random_state = random.randint(1, 999)
    df = load_multilabel_data(filename=dataset, target="sentiment")
    X_train, X_test, Y_train, Y_test = process_and_split_data(
        df,
        target="sentiment",
        additional_features=additional_features,
        random_state=random_state,
    )
    models, training_times = search_sklearn_pipelines(
        X_train,
        Y_train,
        target="sentiment",
        models_to_try=models_to_try,
        additional_features=additional_features,
    )
    model_metrics = []
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        metrics = get_multiclass_metrics(
            X_test,
            Y_test,
            labels=target_names,
            random_state=random_state,
            model=m,
            training_time=t,
        )
        model_metrics.append(metrics)
    write_multilabel_models_and_metrics(models, model_metrics, path)


if __name__ == "__main__":
    run_sentiment_pipeline()
