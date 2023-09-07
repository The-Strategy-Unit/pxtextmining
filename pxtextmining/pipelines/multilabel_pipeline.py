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
from pxtextmining.factories.factory_predict_unlabelled_text import (
    get_thresholds,
    predict_multilabel_bert,
    predict_multilabel_sklearn,
)
from pxtextmining.factories.factory_write_results import (
    write_model_analysis,
    write_model_preds,
    write_multilabel_models_and_metrics,
)
from pxtextmining.params import (
    dataset,
    major_cats,
    merged_minor_cats,
    minor_cats,
    random_state,
)


def run_sklearn_pipeline(
    additional_features=False,
    target=major_cats,
    models_to_try=("mnb", "knn", "svm", "rfc"),
    path="test_multilabel",
    include_analysis=False,
    custom_threshold=False,
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
    if custom_threshold is True:
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
    else:
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
    threshold_dicts = []
    preds = []
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        if custom_threshold is True:
            val_probs = m.predict_proba(X_val)
            custom_threshold_dict = get_thresholds(Y_val, val_probs, labels=target)
            threshold_dicts.append(custom_threshold_dict)
        else:
            custom_threshold_dict = None
        preds_df = predict_multilabel_sklearn(
            X_test,
            m,
            labels=target,
            additional_features=additional_features,
            label_fix=True,
            enhance_with_rules=False,
            custom_threshold_dict=custom_threshold_dict,
        )
        preds.append(preds_df)
        model_metrics.append(
            get_multilabel_metrics(
                preds_df,
                Y_test,
                random_state=random_state,
                labels=target,
                model=m,
                training_time=t,
            )
        )
    write_multilabel_models_and_metrics(models, model_metrics, path=path)
    if include_analysis is True:
        for i in range(len(models)):
            model_name = f"model_{i}"
            write_model_preds(
                X_test,
                Y_test,
                preds[i],
                labels=target,
                path=f"{path}/{model_name}_labels.xlsx",
            )
            write_model_analysis(
                model_name,
                labels=target,
                dataset=df,
                path=path,
                preds_df=preds[i],
                y_true=Y_test,
                custom_threshold_dict=threshold_dicts[i],
            )
    print("Pipeline complete")


def run_svc_pipeline(
    additional_features=False,
    target=major_cats,
    path="test_multilabel",
    include_analysis=False,
    custom_threshold=False,
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
    if custom_threshold is True:
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
    else:
        X_train, X_test, Y_train, Y_test = process_and_split_data(
            df,
            target=target,
            additional_features=additional_features,
            random_state=random_state,
        )
    model, training_time = create_and_train_svc_model(
        X_train, Y_train, additional_features=additional_features
    )
    if custom_threshold is True:
        val_probs = model.predict_proba(X_val)
        custom_threshold_dict = get_thresholds(Y_val, val_probs, labels=target)
    else:
        custom_threshold_dict = None
    preds_df = predict_multilabel_sklearn(
        X_test,
        model=model,
        labels=target,
        additional_features=additional_features,
        label_fix=True,
        enhance_with_rules=False,
        custom_threshold_dict=custom_threshold_dict,
    )
    model_metrics = get_multilabel_metrics(
        preds_df,
        Y_test,
        labels=target,
        random_state=random_state,
        model=model,
        training_time=training_time,
    )
    write_multilabel_models_and_metrics([model], [model_metrics], path=path)
    if include_analysis is True:
        write_model_preds(
            X_test,
            Y_test,
            preds_df,
            labels=target,
            path=f"{path}/labels.xlsx",
        )
        write_model_analysis(
            model_name="model_0",
            labels=target,
            dataset=df,
            path=path,
            preds_df=preds_df,
            y_true=Y_test,
            custom_threshold_dict=custom_threshold_dict,
        )
    print("Pipeline complete!")


def run_bert_pipeline(
    additional_features=False,
    path="test_multilabel/bert",
    target=major_cats,
    include_analysis=False,
    custom_threshold=False,
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
    if custom_threshold is True:
        val = bert_data_to_dataset(X_val, additional_features=additional_features)
        val_probs = model_trained.predict(val)
        custom_threshold_dict = get_thresholds(Y_val, val_probs, labels=target)
    else:
        custom_threshold_dict = None
    preds_df = predict_multilabel_bert(
        test_dataset,
        model=model_trained,
        labels=target,
        additional_features=additional_features,
        label_fix=True,
        enhance_with_rules=False,
        custom_threshold_dict=custom_threshold_dict,
    )
    model_metrics = get_multilabel_metrics(
        preds_df,
        Y_test,
        labels=target,
        random_state=random_state,
        model=model_trained,
        training_time=training_time,
    )
    write_multilabel_models_and_metrics([model_trained], [model_metrics], path=path)
    if include_analysis is True:
        write_model_preds(
            X_test,
            Y_test,
            preds_df,
            labels=target,
            path=f"{path}/labels.xlsx",
        )
        write_model_analysis(
            model_name="model_0",
            labels=target,
            dataset=df,
            path=path,
            preds_df=preds_df,
            y_true=Y_test,
            custom_threshold_dict=custom_threshold_dict,
        )
    print("Pipeline complete!")


if __name__ == "__main__":
    run_svc_pipeline(
        additional_features=False,
        target=minor_cats,
        path="test_multilabel/0906threshold/svc_noq",
        include_analysis=True,
        custom_threshold=True,
    )
    run_svc_pipeline(
        additional_features=True,
        target=minor_cats,
        path="test_multilabel/0906threshold/svc",
        include_analysis=True,
        custom_threshold=True,
    )
    run_sklearn_pipeline(
        additional_features=True,
        target=minor_cats,
        models_to_try=["xgb", "knn"],
        path="test_multilabel/0906threshold/xgb",
        include_analysis=True,
        custom_threshold=True,
    )
    run_bert_pipeline(
        additional_features=True,
        path="test_multilabel/0906threshold/bert",
        target=minor_cats,
        include_analysis=True,
        custom_threshold=True,
    )
    run_bert_pipeline(
        additional_features=False,
        path="test_multilabel/0906threshold/bert_noq",
        target=minor_cats,
        include_analysis=True,
        custom_threshold=True,
    )
    run_sklearn_pipeline(
        additional_features=True,
        target=minor_cats,
        models_to_try=["svm"],
        path="test_multilabel/0906threshold/svc_gridsearch",
        include_analysis=True,
        custom_threshold=True,
    )
    # run_two_layer_sklearn_pipeline()
