import os
import random

# import warnings filter
from warnings import simplefilter

from sklearn.model_selection import train_test_split

from pxtextmining.factories.factory_data_load_and_split import (
    bert_data_to_dataset,
    load_multilabel_data,
    process_and_split_data,
)
from pxtextmining.factories.factory_model_performance import get_multilabel_metrics
from pxtextmining.factories.factory_pipeline import (
    calculating_class_weights,
    create_and_train_svc_model,
    create_bert_model,
    create_bert_model_additional_features,
    search_sklearn_pipelines,
    train_bert_model,
)
from pxtextmining.factories.factory_write_results import (
    write_model_analysis,
    write_model_preds,
    write_multilabel_models_and_metrics,
)
from pxtextmining.params import (
    dataset,
    major_cat_dict,
    major_cats,
    merged_minor_cats,
    minor_cats,
    random_state,
)

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def run_sklearn_pipeline(
    additional_features=False,
    target=major_cats,
    models_to_try=("mnb", "knn", "svm", "rfc"),
    path="test_multilabel",
    include_analysis=False,
):
    """Runs all the functions required to load multilabel data, preprocess it, and split it into training and test sets.
    Creates sklearn pipelines and hyperparameters to search, using specified estimators.
    For each estimator type selected, performs a randomized search across the hyperparameters to identify the parameters providing the best
    results on the holdout data within the randomized search.
    Evaluates the performance of the refitted estimator with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        target (list, optional): The target labels, which should be columns in the dataset DataFrame. Defaults to major_cats.
        models_to_try (list, optional): List of the estimators to try. Defaults to ["mnb", "knn", "svm", "rfc"]. Permitted values are "mnb" (Multinomial Naive Bayes), "knn" (K Nearest Neighbours), "svm" (Support Vector Classifier), or "rfc" (Random Forest Classifier).
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    # random_state = random.randint(1, 999)
    if target == major_cats:
        target_name = "major_categories"
    if target == minor_cats:
        target_name = "minor_categories"
    if target == merged_minor_cats:
        target_name = "test"
    df = load_multilabel_data(filename=dataset, target=target_name)
    X_train, X_test, Y_train, Y_test = process_and_split_data(
        df,
        target=target,
        additional_features=additional_features,
        random_state=random_state,
    )
    models, training_times = search_sklearn_pipelines(
        X_train,
        Y_train,
        models_to_try=models_to_try,
        additional_features=additional_features,
    )
    model_metrics = []
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        model_metrics.append(
            get_multilabel_metrics(
                X_test,
                Y_test,
                random_state=random_state,
                labels=target,
                model_type="sklearn",
                model=m,
                training_time=t,
                additional_features=additional_features,
            )
        )
    write_multilabel_models_and_metrics(models, model_metrics, path=path)
    if include_analysis is True:
        for i in range(len(models)):
            model_name = f"model_{i}"
            write_model_preds(
                X_test,
                Y_test,
                models[i],
                labels=target,
                additional_features=additional_features,
                path=f"{path}/{model_name}_labels.xlsx",
            )
            write_model_analysis(model_name, labels=target, dataset=df, path=path)
    print("Pipeline complete")


def run_svc_pipeline(
    additional_features=False,
    target=major_cats,
    path="test_multilabel",
    include_analysis=False,
):
    """Runs all the functions required to load multilabel data, preprocess it, and split it into training and test sets.
    Creates sklearn pipeline using a MultiOutputClassifier and Support Vector Classifier estimator, with specific hyperparameters.
    Fits the pipeline on the training data.
    Evaluates the performance of the refitted estimator with the best hyperparameters on the test set, and saves the model and the performance metrics to a specified folder, together with optional further analysis in the form of Excel files.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        target (list, optional): The target labels, which should be columns in the dataset DataFrame. Defaults to major_cats.
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
        include_analysis (bool, optional): Whether or not to create Excel files including further analysis of the model's performance. Defaults to False. If True, writes two Excel files to the specified folder, one containing the labels and the performance metrics for each label, and one containing the predicted labels and the actual labels for the test set, with the model's probabilities for both.

    """
    # random_state = random.randint(1, 999)
    if target == major_cats:
        target_name = "major_categories"
    if target == minor_cats:
        target_name = "minor_categories"
    if target == merged_minor_cats:
        target_name = "test"
    df = load_multilabel_data(filename=dataset, target=target_name)
    X_train, X_test, Y_train, Y_test = process_and_split_data(
        df,
        target=target,
        additional_features=additional_features,
        random_state=random_state,
    )
    model, training_time = create_and_train_svc_model(
        X_train, Y_train, additional_features=additional_features
    )
    model_metrics = get_multilabel_metrics(
        X_test,
        Y_test,
        random_state=random_state,
        labels=target,
        model_type="sklearn",
        model=model,
        training_time=training_time,
        additional_features=additional_features,
    )
    write_multilabel_models_and_metrics([model], [model_metrics], path=path)
    if include_analysis is True:
        write_model_preds(
            X_test,
            Y_test,
            model,
            labels=target,
            additional_features=additional_features,
            path=f"{path}/labels.xlsx",
        )
        write_model_analysis(model_name="model_0", labels=target, dataset=df, path=path)
    print("Pipeline complete!")


def run_bert_pipeline(
    additional_features=False,
    path="test_multilabel/bert",
    target=major_cats,
    include_analysis=False,
):
    """Runs all the functions required to load multilabel data, preprocess it, and split it into training, test and validation sets.
    Creates tf.keras Transformer model with additional layers specific to the classification task, and trains it on the train set.
    Evaluates the performance of trained model with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    # random_state = random.randint(1, 999)
    print(f"random_state is: {random_state}")
    if target == major_cats:
        target_name = "major_categories"
    if target == minor_cats:
        target_name = "minor_categories"
    if target == merged_minor_cats:
        target_name = "test"
    df = load_multilabel_data(filename=dataset, target=target_name)
    X_train_val, X_test, Y_train_val, Y_test = process_and_split_data(
        df,
        target=target,
        preprocess_text=False,
        additional_features=additional_features,
        random_state=random_state,
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.2, random_state=random_state
    )
    train_dataset = bert_data_to_dataset(
        X_train, Y_train, additional_features=additional_features
    )
    val_dataset = bert_data_to_dataset(
        X_val, Y_val, additional_features=additional_features
    )
    test_dataset = bert_data_to_dataset(
        X_test, Y=None, additional_features=additional_features
    )
    class_weights_dict = calculating_class_weights(Y_train_val)
    if additional_features is True:
        model = create_bert_model_additional_features(Y_train)
    else:
        model = create_bert_model(Y_train)
    model_trained, training_time = train_bert_model(
        train_dataset,
        val_dataset,
        model,
        class_weights_dict=class_weights_dict,
        epochs=25,
    )
    model_metrics = get_multilabel_metrics(
        test_dataset,
        Y_test,
        random_state=random_state,
        labels=target,
        model_type="bert",
        model=model_trained,
        training_time=training_time,
        additional_features=additional_features,
        already_encoded=True,
    )
    write_multilabel_models_and_metrics([model_trained], [model_metrics], path=path)
    if include_analysis is True:
        write_model_preds(
            X_test,
            Y_test,
            model,
            labels=target,
            additional_features=additional_features,
            path=f"{path}/labels.xlsx",
        )
        write_model_analysis(model_name="model_0", labels=target, dataset=df, path=path)
    print("Pipeline complete!")


def run_two_layer_sklearn_pipeline(
    additional_features=True,
    models_to_try=("mnb", "knn", "xgb"),
    path="test_multilabel/230605",
):
    random_state = random.randint(1, 999)
    df_major = load_multilabel_data(filename=dataset, target="major_categories")
    df_minor = load_multilabel_data(filename=dataset, target="minor_categories")
    # major cats first
    X_train, X_test, Y_train, Y_test = process_and_split_data(
        df_major,
        target=major_cats,
        additional_features=additional_features,
        random_state=random_state,
    )
    target = major_cats
    models, training_times = search_sklearn_pipelines(
        X_train,
        Y_train,
        models_to_try=models_to_try,
        additional_features=additional_features,
    )
    svc, svc_time = create_and_train_svc_model(X_train, Y_train)
    models.append(svc)
    training_times.append(svc_time)
    model_metrics = []
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        model_metrics.append(
            get_multilabel_metrics(
                X_test,
                Y_test,
                random_state=random_state,
                labels=target,
                model_type="sklearn",
                model=m,
                training_time=t,
                additional_features=additional_features,
            )
        )
    write_multilabel_models_and_metrics(models, model_metrics, path=path)
    # minor cats
    for k, v in major_cat_dict.items():
        if len(v) > 1:
            print(k)
            target = v
            model_name = k
            minipath = os.path.join(path, model_name)
            X_train, X_test, Y_train, Y_test = process_and_split_data(
                df_minor,
                target=target,
                additional_features=additional_features,
                random_state=random_state,
            )
            models, training_times = search_sklearn_pipelines(
                X_train,
                Y_train,
                models_to_try=["mnb", "knn", "xgb"],
                additional_features=additional_features,
            )
            svc, svc_time = create_and_train_svc_model(X_train, Y_train)
            models.append(svc)
            training_times.append(svc_time)
            model_metrics = []
            for i in range(len(models)):
                m = models[i]
                t = training_times[i]
                model_metrics.append(
                    get_multilabel_metrics(
                        X_test,
                        Y_test,
                        random_state=random_state,
                        labels=target,
                        model_type="sklearn",
                        model=m,
                        training_time=t,
                        additional_features=additional_features,
                    )
                )
            write_multilabel_models_and_metrics(models, model_metrics, path=minipath)


if __name__ == "__main__":
    # run_sklearn_pipeline(
    #     additional_features=True,
    #     target=minor_cats,
    #     models_to_try=["xgb"],
    #     path='test_multilabel/v6_230724/xgb',
    #     include_analysis=True,
    # )
    # run_svc_pipeline(
    #     additional_features=False,
    #     target=minor_cats,
    #     path="test_multilabel/v6_230724/svc_nofeats",
    #     include_analysis=True,
    # )
    # run_svc_pipeline(
    #     additional_features=True,
    #     target=minor_cats,
    #     path="test_multilabel/v6_230724/svc",
    #     include_analysis=True,
    # )
    run_bert_pipeline(
        additional_features=True,
        path="test_multilabel/v6_230724/bert",
        target=minor_cats,
        include_analysis=True,
    )
    # run_sklearn_pipeline(
    #     additional_features=True,
    #     target=minor_cats,
    #     models_to_try=["svm"],
    #     path='test_multilabel/v6_230724/svc_gridsearch',
    #     include_analysis=True,
    # )
    # run_two_layer_sklearn_pipeline()
