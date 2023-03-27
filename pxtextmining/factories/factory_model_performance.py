import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from pxtextmining.factories.factory_predict_unlabelled_text import (
    fix_no_labels,
    predict_with_bert,
    turn_probs_into_binary,
)


def get_dummy_model(x_train, y_train):
    """Creates dummy model that randomly predicts labels, fitted on the training data.

    Args:
        x_train (pd.DataFrame): Input features.
        y_train (pd.DataFrame): Target values.

    Returns:
        (sklearn.dummy.DummyClassifier): Trained dummy classifier.
    """
    model = DummyClassifier(strategy="uniform")
    model.fit(x_train, y_train)
    return model

def get_multilabel_metrics(
    x_test,
    y_test,
    labels,
    random_state,
    model_type,
    model,
    training_time=None,
    additional_features=False,
    already_encoded=False,
):
    """Creates a string detailing various performance metrics for a multilabel model, which can then be written to
    a text file.

    Args:
        x_test (pd.DataFrame): DataFrame containing test dataset features
        y_test (pd.DataFrame): DataFrame containing test dataset true target values
        labels (list): List containing the target labels
        random_state (int): Seed used to control the shuffling of the data, to enable reproducible results.
        model_type (str): Type of model used. Options are 'bert', 'tf', or 'sklearn'. Defaults to None.
        model (tf.keras or sklearn model): Trained estimator.
        training_time (str, optional): Amount of time taken for model to train. Defaults to None.
        additional_features (bool, optional): Whether or not additional features (e.g. question type) have been included in training the model. Defaults to False.
        already_encoded (bool, optional): Whether or not, if a `bert` model was used, x_test has already been encoded. Defaults to False.

    Raises:
        ValueError: Only model_type 'bert', 'tf' or 'sklearn' are allowed.

    Returns:
        (str): String containing the model architecture/hyperparameters, random state used for the train test split, and performance metrics including: exact accuracy, hamming loss, macro jaccard score, and classification report.
    """

    metrics_string = "\n *****************"
    metrics_string += (
        f"\n Random state seed for train test split is: {random_state} \n\n"
    )
    model_metrics = {}
    # TF Keras models output probabilities with model.predict, whilst sklearn models output binary outcomes
    # Get them both to output the same (binary outcomes) and take max prob as label if no labels predicted at all
    if model_type in ("bert", "tf"):
        if model_type == "bert":
            y_probs = predict_with_bert(
                x_test,
                model,
                additional_features=additional_features,
                already_encoded=already_encoded,
            )
        elif model_type == "tf":
            y_probs = model.predict(x_test)
        binary_preds = turn_probs_into_binary(y_probs)
        y_pred = fix_no_labels(binary_preds, y_probs, model_type="tf")
    elif model_type == "sklearn":
        binary_preds = model.predict(x_test)
        y_probs = np.array(model.predict_proba(x_test))
        y_pred = fix_no_labels(binary_preds, y_probs, model_type="sklearn")
    else:
        raise ValueError('Please select valid model_type. Options are "bert", "tf" or "sklearn"')
    # Calculate various metrics
    model_metrics["exact_accuracy"] = metrics.accuracy_score(y_test, y_pred)
    model_metrics["hamming_loss"] = metrics.hamming_loss(y_test, y_pred)
    model_metrics["macro_jaccard_score"] = metrics.jaccard_score(
        y_test, y_pred, average="macro"
    )
    if model_type in ("bert", "tf"):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        metrics_string += f"\n{model_summary}\n"
    else:
        metrics_string += f"\n{model}\n"
    metrics_string += f"\n\nTraining time: {training_time}\n"
    for k, v in model_metrics.items():
        metrics_string += f"\n{k}: {v}"
    # Classification report
    metrics_string += "\n\n Classification report:\n"
    c_report_str = metrics.classification_report(
        y_test, y_pred, target_names=labels, zero_division=0
    )
    metrics_string += c_report_str
    return metrics_string


def get_accuracy_per_class(y_test, pred):
    """Function to produce accuracy per class for the predicted categories, compared against real values.

    :param pd.Series y_test: Test data (real target values).
    :param pd.Series pred: Predicted target values.

    :return: The computed accuracy per class metrics for the model.
    :rtype: pd.DataFrame
    """
    cm = confusion_matrix(y_test, pred)
    accuracy_per_class = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy_per_class = pd.DataFrame(accuracy_per_class.diagonal())
    accuracy_per_class.columns = ["accuracy"]
    unique, frequency = np.unique(y_test, return_counts=True)
    accuracy_per_class["class"], accuracy_per_class["counts"] = unique, frequency
    accuracy_per_class = accuracy_per_class[["class", "counts", "accuracy"]]
    return accuracy_per_class
