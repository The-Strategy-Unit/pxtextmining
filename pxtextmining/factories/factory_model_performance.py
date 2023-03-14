import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, accuracy_score
from pxtextmining.helpers.metrics import class_balance_accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras import Sequential, Model
from pxtextmining.factories.factory_predict_unlabelled_text import turn_probs_into_binary, fix_no_labels, predict_with_bert


def get_multilabel_metrics(x_test, y_test, labels, random_state, model_type = None, model = None,
                           training_time = None, x_train = None, y_train = None, additional_features = False,
                           already_encoded = False):
    """Function to produce performance metrics for a multilabel machine learning model.

    :param pd.DataFrame x_test: Test data (predictor).
    :param pd.Series y_test: Test data (target).
    :param pd.DataFrame x_train: Training data (predictor). Defaults to None, only needed if model = None
    :param pd.Series y_train: Training data (target). Defaults to None, only needed if model = None
    :param str model: Trained classifier. Defaults to 'dummy' which instantiates dummy classifier for baseline metrics.

    :return: None
    :rtype: None
    """

    metrics_string = '\n *****************'
    metrics_string += f'\n Random state seed for train test split is: {random_state} \n\n'
    model_metrics = {}
    if model == None:
        model = DummyClassifier(strategy = 'uniform')
        if isinstance(x_train, pd.Series):
            model.fit(x_train, y_train)
        else:
            raise ValueError('For dummy model, x_train and y_train must be provided')
    # TF Keras models output probabilities with model.predict, whilst sklearn models output binary outcomes
    # Get them both to output the same (binary outcomes) and take max prob as label if no labels predicted at all
    if model_type in ('bert', 'tf'):
        if model_type == 'bert':
            y_probs = predict_with_bert(x_test, model, additional_features = additional_features, already_encoded= already_encoded)
        else:
            y_probs = model.predict(x_test)
        binary_preds = turn_probs_into_binary(y_probs)
        y_pred = fix_no_labels(binary_preds, y_probs, model_type = 'tf')
    else:
        binary_preds = model.predict(x_test)
        y_probs = np.array(model.predict_proba(x_test))
        y_pred = fix_no_labels(binary_preds, y_probs, model_type = 'sklearn')
    c_report_str = metrics.classification_report(y_test, y_pred,
                                            target_names = labels, zero_division=0)
    model_metrics['exact_accuracy'] = metrics.accuracy_score(y_test, y_pred)
    model_metrics['hamming_loss'] = metrics.hamming_loss(y_test, y_pred)
    model_metrics['macro_jaccard_score'] = metrics.jaccard_score(y_test, y_pred, average = 'macro')
    if isinstance(model, (Sequential, Model)):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        metrics_string += f'\n{model_summary}\n'
    else:
        metrics_string += f'\n{model}\n'
    metrics_string += f'\n\nTraining time: {training_time}\n'
    for k,v in model_metrics.items():
        metrics_string += f'\n{k}: {v}'
    metrics_string += '\n\n Classification report:\n'
    metrics_string += c_report_str
    # per_class_jaccard = zip(labels,metrics.jaccard_score(y_test, y_pred, average = None, zero_division = 0))
    # metrics_string += '\nper class Jaccard score:'
    # for k,v in per_class_jaccard:
    #     print(f'{k}: {v}')
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
