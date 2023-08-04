import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import is_classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model

from pxtextmining.factories.factory_predict_unlabelled_text import (
    predict_multiclass_bert,
    predict_multilabel_bert,
    predict_multilabel_sklearn,
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


def get_multiclass_metrics(
    x_test, y_test, labels, random_state, model, additional_features, training_time=None
):
    """Creates a string detailing various performance metrics for a multiclass model, which can then be written to
    a text file.

    Args:
        x_test (pd.DataFrame): DataFrame containing test dataset features
        y_test (pd.DataFrame): DataFrame containing test dataset true target values
        labels (list): List containing the target labels
        random_state (int): Seed used to control the shuffling of the data, to enable reproducible results.
        model (tf.keras or sklearn model): Trained estimator.
        additional_features (bool, optional): Whether or not additional features (e.g. question type) have been included in training the model. Defaults to False.
        training_time (str, optional): Amount of time taken for model to train. Defaults to None.

    Raises:
        ValueError: Only models built with sklearn or tensorflow are allowed.

    Returns:
        (str): String containing the model architecture/hyperparameters, random state used for the train test split, and classification report.
    """
    metrics_string = "\n *****************"
    metrics_string += (
        f"\n Random state seed for train test split is: {random_state} \n\n"
    )
    # TF Keras models output probabilities with model.predict, whilst sklearn models output binary outcomes
    # Get them both to output the same (binary outcomes) and take max prob as label if no labels predicted at all
    if isinstance(model, Model) is True:
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        metrics_string += f"\n{model_summary}\n"
        y_pred = predict_multiclass_bert(
            x_test,
            model,
            additional_features=additional_features,
            already_encoded=False,
        )
    elif is_classifier(model) is True:
        metrics_string += f"\n{model}\n"
        y_pred = model.predict(x_test)
    else:
        raise ValueError("Model type not recognised")
    # Calculate various metrics
    metrics_string += f"\n\nTraining time: {training_time}\n"
    # Classification report
    metrics_string += "\n\n Classification report:\n"
    c_report_str = metrics.classification_report(
        y_test, y_pred, target_names=labels, zero_division=0
    )
    metrics_string += c_report_str
    return metrics_string


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
    enhance_with_rules=False,
):
    """Creates a string detailing various performance metrics for a multilabel model, which can then be written to
    a text file.

    Args:
        x_test (pd.DataFrame): DataFrame containing test dataset features
        y_test (pd.DataFrame): DataFrame containing test dataset true target values
        labels (list): List containing the target labels
        random_state (int): Seed used to control the shuffling of the data, to enable reproducible results.
        model_type (str): Type of model used. Options are 'bert', or 'sklearn'. Defaults to None.
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
    if model_type == "bert":
        y_pred_df = predict_multilabel_bert(
            x_test,
            model,
            labels=labels,
            additional_features=additional_features,
            label_fix=True,
            enhance_with_rules=enhance_with_rules,
            already_encoded=already_encoded,
        )
    elif model_type == "sklearn":
        y_pred_df = predict_multilabel_sklearn(
            x_test,
            model,
            labels=labels,
            additional_features=additional_features,
            label_fix=True,
            enhance_with_probs=True,
            enhance_with_rules=enhance_with_rules,
        )
    else:
        raise ValueError(
            'Please select valid model_type. Options are "bert" or "sklearn"'
        )
    y_pred = np.array(y_pred_df[labels]).astype("int64")
    # Calculate various metrics
    model_metrics["exact_accuracy"] = metrics.accuracy_score(y_test, y_pred)
    model_metrics["hamming_loss"] = metrics.hamming_loss(y_test, y_pred)
    model_metrics["macro_jaccard_score"] = metrics.jaccard_score(
        y_test, y_pred, average="macro"
    )
    y_probs = y_pred_df.filter(like="Probability", axis=1)
    model_metrics["macro_roc_auc"] = metrics.roc_auc_score(
        y_test, y_probs, multi_class="ovr"
    )
    # Model summary
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

    Args:
        y_test (pd.Series): Test data (real target values).
        pred (pd.Series): Predicted target values.

    Returns:
        (pd.DataFrame): The computed accuracy per class metrics for the model.

    """
    cm = confusion_matrix(y_test, pred)
    accuracy_per_class = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy_per_class = pd.DataFrame(accuracy_per_class.diagonal())
    accuracy_per_class.columns = ["accuracy"]
    unique, frequency = np.unique(y_test, return_counts=True)
    accuracy_per_class["class"], accuracy_per_class["counts"] = unique, frequency
    accuracy_per_class = accuracy_per_class[["class", "counts", "accuracy"]]
    return accuracy_per_class


def parse_metrics_file(metrics_file, labels):
    """Reads performance metrics files that are written by `factory_write_results.write_multilabel_models_and_metrics`.
    Creates a pd.DataFrame with the precision, recall, f1_score, and support for each label, which can be filtered and sorted more easily.

    Args:
        metrics_file (str): Path to the metrics file to be parsed.
        labels (list): List of the target labels used in the metrics file.

    Returns:
        (pd.DataFrame): DataFrame containing the precision, recall, f1_score, and support for each label, as detailed in the performance metrics file.
    """
    with open(metrics_file, "r") as file:
        content = file.readlines()
    for i, line in enumerate(content):
        if line.strip().startswith(labels[0][:10]):
            startline = i
        if line.strip().startswith(labels[-1][:10]):
            endline = i + 1
    lines = [x.strip() for x in content[startline:endline]]
    metrics_dict = {
        "label": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "support": [],
    }
    for each in lines:
        splitted = each.split("      ")
        metrics_dict["label"].append(splitted[0].strip())
        metrics_dict["precision"].append(splitted[1].strip())
        metrics_dict["recall"].append(splitted[2].strip())
        metrics_dict["f1_score"].append(splitted[3].strip())
        metrics_dict["support"].append(splitted[4].strip())
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    return metrics_df


def get_y_score(probs):
    """Converts probabilities into format (n_samples, n_classes) so they can be passed into sklearn roc_auc_score function

    Args:
        probs (np.ndarray): Probability estimates outputted by model

    Returns:
        np.ndarray: Probability estimates in format (n_samples, n_classes)
    """
    if probs.ndim == 3:
        score = np.transpose([pred[:, 1] for pred in probs])
    elif probs.ndim == 2:
        score = probs
    return score


def additional_analysis(preds_df, y_true, labels):
    """For given predictions, returns dataframe containing: macro one-vs-one ROC AUC score, number of True Positives, True Negatives, False Positives, and False Negatives.

    Args:
        preds_df (pd.DataFrame): Dataframe containing predicted labels in one-hot encoded format
        y_true (np.array): One-hot encoded real Y values
        labels (List): List of the target labels

    Returns:
        pd.DataFrame: dataframe containing: macro one-vs-one ROC AUC score, number of True Positives, True Negatives, False Positives, and False Negatives.
    """
    # include threshold?? (later)
    y_score = preds_df.filter(like="Probability", axis=1)
    label_rocs = metrics.roc_auc_score(y_true, y_score, average=None, multi_class="ovo")
    rocs = pd.Series(dict(zip(labels, label_rocs, strict=True)))
    cm = metrics.multilabel_confusion_matrix(y_true, np.array(preds_df[labels]))
    cm_dict = {}
    for i, label in enumerate(labels):
        cm_meaning = {}
        tn, fn = cm[i][0]
        fp, tp = cm[i][1]
        cm_meaning["True Negative"] = tn
        cm_meaning["False Negative"] = fn
        cm_meaning["True Positive"] = tp
        cm_meaning["False Positive"] = fp
        cm_dict[label] = cm_meaning
    df = pd.DataFrame.from_dict(cm_dict, orient="index")
    df["roc_auc_score"] = rocs
    return df
