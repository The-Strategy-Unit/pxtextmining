import random
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split

from pxtextmining.factories.factory_data_load_and_split import (
    bert_data_to_dataset,
    load_multilabel_data,
    process_and_split_data,
)
from pxtextmining.factories.factory_model_performance import (
    get_multilabel_metrics,
    parse_metrics_file,
)
from pxtextmining.factories.factory_pipeline import (
    calculating_class_weights,
    create_bert_model,
    create_bert_model_additional_features,
    create_tf_model,
    create_and_train_svc_model,
    search_sklearn_pipelines,
    train_bert_model,
    train_tf_model,
)
from pxtextmining.factories.factory_write_results import (
    write_multilabel_models_and_metrics,
    write_model_preds,
)
from pxtextmining.helpers.text_preprocessor import tf_preprocessing
from pxtextmining.params import dataset

def run_sentiment_pipeline(
    additional_features=False,
    models_to_try=["svm", "xgb"],
    path="test_multilabel/sentiment",
):
    random_state = random.randint(1, 999)
    df = load_multilabel_data(filename=dataset, target='sentiment')
    X_train, X_test, Y_train, Y_test = process_and_split_data(
        df,
        target='sentiment',
        additional_features=additional_features,
        random_state=random_state
    )
    models, training_times = search_sklearn_pipelines(
        X_train,
        Y_train, target = 'sentiment',
        models_to_try=models_to_try,
        additional_features=additional_features,
    )
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        model_name = models_to_try[i]
        modelpath = os.path.join(path, model_name + '.sav')
        pickle.dump(m, open(modelpath, "wb"))
        txtpath = os.path.join(path, model_name + '.txt')
        with open(txtpath, "w") as file:
            file.write(f"random state = {random_state} \n\n training time = {t}")

    #     model_metrics.append(
    #         get_multilabel_metrics(
    #             X_test,
    #             Y_test,
    #             random_state=random_state,
    #             labels=target,
    #             model_type="sklearn",
    #             model=m,
    #             training_time=t,
    #         )
    #     )
    # write_multilabel_models_and_metrics(models, model_metrics, path=path)

if __name__ == '__main__':
    run_sentiment_pipeline()
