import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

from pxtextmining.factories.factory_data_load_and_split import (
    bert_data_to_dataset,
    load_multilabel_data,
    process_and_split_data,
)
from pxtextmining.factories.factory_model_performance import get_multiclass_metrics
from pxtextmining.factories.factory_pipeline import (
    search_sklearn_pipelines,
    create_bert_model_additional_features,
    train_bert_model,
    create_bert_model,
)
from pxtextmining.factories.factory_write_results import (
    write_multilabel_models_and_metrics,
)
from pxtextmining.params import dataset

random_state = random.randint(1, 999)


def run_sentiment_pipeline(
    additional_features=False,
    models_to_try=["svm", "xgb"],
    path="test_multilabel/sentiment",
):
    """Runs all the functions required to load multiclass data, preprocess it, and split it into training, test and validation sets.
    Creates sklearn model and hyperparameter grid to search, and trains it on the train set.
    Evaluates the performance of trained model with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        models_to_try (list, optional): Which model types to try. Defaults to ["svm", "xgb"].
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    target_names = ["very positive", "positive", "neutral", "negative", "very negative"]
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
            additional_features=additional_features,
        )
        model_metrics.append(metrics)
    write_multilabel_models_and_metrics(models, model_metrics, path)


def run_sentiment_bert_pipeline(
    additional_features=True, path="test_multilabel/sentiment_bert"
):
    """Runs all the functions required to load multiclass data, preprocess it, and split it into training, test and validation sets.
    Creates tf.keras Transformer model with additional layers specific to the classification task, and trains it on the train set.
    Evaluates the performance of trained model with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    print(f"random_state is: {random_state}")
    target_names = ["very positive", "positive", "neutral", "negative", "very negative"]
    df = load_multilabel_data(filename=dataset, target="sentiment")
    X_train_val, X_test, Y_train_val, Y_test = process_and_split_data(
        df,
        target="sentiment",
        additional_features=additional_features,
        preprocess_text=True,
        random_state=random_state,
    )
    Y_train_val_oh = to_categorical(Y_train_val)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val_oh, test_size=0.2, random_state=random_state
    )
    train_dataset = bert_data_to_dataset(
        X_train, Y_train, additional_features=additional_features
    )
    val_dataset = bert_data_to_dataset(
        X_val, Y_val, additional_features=additional_features
    )
    cw = compute_class_weight("balanced", classes=np.unique(Y_train_val), y=Y_train_val)
    class_weights_dict = {}
    for k, v in enumerate(list(cw)):
        class_weights_dict[k] = v
    if additional_features == True:
        model = create_bert_model_additional_features(Y_train, multilabel=False)
    else:
        model = create_bert_model(Y_train, multilabel=False)
    model_trained, training_time = train_bert_model(
        train_dataset,
        val_dataset,
        model,
        class_weights_dict=class_weights_dict,
        epochs=25,
    )
    model_metrics = get_multiclass_metrics(
        X_test,
        Y_test,
        random_state=random_state,
        labels=target_names,
        model=model_trained,
        training_time=training_time,
        additional_features=additional_features,
    )
    write_multilabel_models_and_metrics([model_trained], [model_metrics], path=path)


if __name__ == "__main__":
    # run_sentiment_pipeline(additional_features=False)
    run_sentiment_bert_pipeline(
        additional_features=True, path="test_multilabel/sentiment_bert"
    )
